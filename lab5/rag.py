
from pathlib import Path
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "documents"
STORE = ROOT / "vector_store.json"
MODEL = "llama3.2"

def chunk_text(text, source, max_chars=320, overlap=50):
    "Розбиття тексту на фрагменти зі збереженням назви джерела."
    paragraphs = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > max_chars:
            chunks.append({"text": current.strip(), "source": source})
            current = current[-overlap:] + " " + paragraph if overlap else paragraph
        else:
            current = (current + " " + paragraph).strip()
    if current:
        chunks.append({"text": current.strip(), "source": source})
    return chunks

def build_index():
    chunks = []
    for path in sorted(DOCS.glob("*.txt")):
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise RuntimeError("У папці documents немає TXT-документів.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform([c["text"] for c in chunks]).toarray()
    payload = {
        "chunks": chunks,
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "vectors": matrix.tolist()
    }
    STORE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Індекс створено: {len(chunks)} фрагментів із {len(list(DOCS.glob('*.txt')))} документів.")

def load_index():
    if not STORE.exists():
        build_index()
    data = json.loads(STORE.read_text(encoding="utf-8"))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, vocabulary=data["vocabulary"])
    vectorizer.fit([" ".join(data["vocabulary"].keys())])
    # Відновлюємо IDF, збережені під час індексації.
    vectorizer.idf_ = np.asarray(data["idf"])
    return data, vectorizer

def retrieve(question, top_k=3):
    data, vectorizer = load_index()
    q_vec = vectorizer.transform([question]).toarray()
    scores = cosine_similarity(q_vec, np.asarray(data["vectors"]))[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [(data["chunks"][i], float(scores[i])) for i in order]

def ask_ollama(prompt):
    "Генерація відповіді через локальну модель Ollama."
    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        print("\n[Ollama] Відповідь отримана від Llama 3.2.")

        return response.json()["response"].strip()

    except Exception as e:
        print(f"\n[Ollama] Помилка підключення: {e}")
        print("[Ollama] Використовується демонстраційний режим.")

        return None
def answer(question, top_k=3):
    results = retrieve(question, top_k)
    # Додаткова перевірка покриття змістових слів запиту.
    # Вона допомагає не приймати збіг лише за загальним словом (наприклад, Python)
    # за доказ наявності відповіді на запит про автора та рік створення.
    stopwords = {"що", "таке", "для", "чого", "які", "яка", "який", "чим",
                 "як", "в", "у", "на", "і", "та", "чи", "це", "є", "про",
                 "від", "між", "потрібна", "потрібен"}
    terms = [t.lower() for t in re.findall(r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9]+", question)
             if len(t) > 2 and t.lower() not in stopwords]
    corpus_terms = set()
    for item in json.loads(STORE.read_text(encoding="utf-8"))["chunks"]:
        corpus_terms.update(t.lower() for t in re.findall(
            r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9]+", item["text"]))
    coverage = sum(t in corpus_terms for t in terms) / max(1, len(terms))
    if not results or results[0][1] < 0.08 or (len(terms) >= 2 and coverage < 0.5):
        return "У наданій базі документів немає достатньо інформації для відповіді.", results
    context = "\n\n".join(
        f"[Джерело: {item['source']}]\n{item['text']}" for item, score in results if score > 0
    )
    prompt = f"""Відповідай українською лише на основі контексту.
Якщо контекст не містить відповіді, прямо скажи про це. Не вигадуй фактів.
Контекст:
{context}
Запитання: {question}
Відповідь:"""
    generated = ask_ollama(prompt)
    if generated:
        return generated, results
    # Демонстраційний fallback: повертає релевантні речення з retrieved-контексту.
    sentences = []
    for item, score in results:
        if score > 0:
            sentences.extend(re.split(r"(?<=[.!?])\s+", item["text"]))
    selected = [s for s in sentences if s.strip()][:3]
    return (" ".join(selected) if selected else
            "У контексті не знайдено достатньо інформації для відповіді."), results

if __name__ == "__main__":
    print("Проста RAG-система. Для виходу введіть: exit")
    while True:
        q = input("\nЗапитання: ").strip()
        if q.lower() in {"exit", "quit", "вихід"}:
            break
        response, found = answer(q)
        print("\nЗнайдені фрагменти:")
        for chunk, score in found:
            print(f"- {chunk['source']} | score={score:.3f}: {chunk['text']}")
        print("\nВідповідь:", response)
