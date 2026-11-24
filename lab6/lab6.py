
import csv
from pathlib import Path
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = "llama3.2"
URL = "http://localhost:11434/api/generate"
CATEGORIES = ["Основи Python", "Керування потоком", "Функції", "Структури даних"]

EXAMPLES = [
    ("Python є мовою програмування високого рівня.", "Основи Python"),
    ("Оператор for перебирає елементи послідовності.", "Керування потоком"),
    ("Функція оголошується за допомогою def.", "Функції"),
    ("Словник зберігає пари ключ—значення.", "Структури даних"),
    ("Оператор if перевіряє логічну умову.", "Керування потоком"),
    ("Метод append додає елемент до списку.", "Структури даних"),
]

def ask_ollama(prompt):
    try:
        response = requests.post(URL, json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0}
        }, timeout=180)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException as error:
        raise RuntimeError(
            "Ollama недоступна. Запустіть Ollama та виконайте: ollama pull llama3.2"
        ) from error

def get_category(answer):
    for category in CATEGORIES:
        if category.lower() in answer.lower():
            return category
    return "Не визначено"

def prompt_engineering(text):
    prompt = f"""Ти класифікуєш навчальні тексти з Python.
Обери одну категорію: {", ".join(CATEGORIES)}.
Поверни лише назву категорії без пояснень.
Текст: {text}
Категорія:"""
    return get_category(ask_ollama(prompt))

def build_rag_index():
    documents = []
    for path in sorted(DATA.glob("*.txt")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([text for _, text in documents])
    return documents, vectorizer, matrix

def retrieve(text, documents, vectorizer, matrix, top_k=3):
    query_vector = vectorizer.transform([text])
    scores = cosine_similarity(query_vector, matrix).flatten()
    indices = scores.argsort()[::-1][:top_k]
    return "\n\n".join(
        f"Джерело: {documents[i][0]} (similarity={scores[i]:.3f})\n{documents[i][1]}"
        for i in indices
    )

def rag(text, documents, vectorizer, matrix):
    context = retrieve(text, documents, vectorizer, matrix)
    prompt = f"""Використовуючи лише наведений контекст, класифікуй текст.
Можливі категорії: {", ".join(CATEGORIES)}.
Поверни лише назву категорії.
Контекст:
{context}
Текст:
{text}
Категорія:"""
    return get_category(ask_ollama(prompt))

def few_shot(text):
    examples = "\n\n".join(
        f"Текст: {sample}\nКатегорія: {label}" for sample, label in EXAMPLES
    )
    prompt = f"""Класифікуй навчальні тексти з Python за прикладами.
{examples}

Можливі категорії: {", ".join(CATEGORIES)}.
Поверни лише назву категорії.
Текст: {text}
Категорія:"""
    return get_category(ask_ollama(prompt))

def main():
    with open(DATA / "test_data.csv", encoding="utf-8-sig", newline="") as f:
        tests = list(csv.DictReader(f))
    documents, vectorizer, matrix = build_rag_index()
    results = []
    for item in tests:
        text, correct = item["text"], item["correct_category"]
        p = prompt_engineering(text)
        r = rag(text, documents, vectorizer, matrix)
        fs = few_shot(text)
        row = {
            "id": item["id"], "text": text, "correct": correct,
            "prompt": p, "rag": r, "few_shot": fs,
            "prompt_correct": p == correct, "rag_correct": r == correct,
            "few_shot_correct": fs == correct
        }
        results.append(row)
        print(f"\n#{item['id']} | Еталон: {correct}\nPrompt: {p}\nRAG: {r}\nFew-shot: {fs}")

    out = ROOT / "results.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nПІДСУМКИ")
    for key, title in [
        ("prompt_correct", "Prompt Engineering"),
        ("rag_correct", "RAG"),
        ("few_shot_correct", "Імітація Fine-tuning")
    ]:
        count = sum(row[key] for row in results)
        print(f"{title}: {count}/{len(results)} = {count/len(results)*100:.1f}%")
    print(f"\nРезультати збережено: {out}")

if __name__ == "__main__":
    main()
