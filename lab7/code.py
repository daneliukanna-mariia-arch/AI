

import ast
import operator
import re
from datetime import date, timedelta




_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Безпечно обчислює арифметичний вираз без eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Завеликий показник степеня.")
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Підтримуються лише числа та арифметичні операції.")


def calculator(expression):
    """Інструмент Calculator: приймає математичний вираз, повертає число."""
    expression = expression.replace(",", ".").strip()
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


def date_planner(start_text, days):
    """Інструмент DatePlanner: формує календарний план від сьогодні."""
    start = date.fromisoformat(start_text) if start_text else date.today()
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range(days)
    ]



class StudentAssistantAgent:
    def __init__(self):
        self.memory = []  # Історія реплік у межах поточного запуску
        self.last_topic = None

    def remember(self, role, text):
        self.memory.append({"role": role, "text": text})

    def _topic_from_text(self, text):
        low = text.lower()
        topics = {
            "Python": ["python", "пайтон"],
            "Java": ["java", "джава"],
            "математика": ["математик", "алгебр", "геометр"],
            "AI": ["штучний інтелект", "ai-агент", "нейромереж"],
        }
        for topic, keywords in topics.items():
            if any(word in low for word in keywords):
                return topic
        return None

    def _make_plan(self, intent):
        plans = {
            "calculator": [
                "Виділити математичний вираз із запиту.",
                "Передати вираз інструменту Calculator.",
                "Отримати числовий результат і пояснити його.",
            ],
            "average": [
                "Знайти числа в запиті.",
                "Обчислити їх суму інструментом Calculator.",
                "Поділити суму на кількість чисел.",
                "Пояснити результат користувачу.",
            ],
            "study_plan": [
                "Визначити навчальну тему та тривалість підготовки.",
                "Розподілити матеріал на послідовні дні.",
                "Сформувати календар за допомогою DatePlanner.",
            ],
            "explain": [
                "Визначити тему запитання.",
                "Підібрати коротке пояснення з навчальної довідки.",
                "Надати відповідь з урахуванням попереднього контексту.",
            ],
            "topic_memory": [
                "Визначити навчальну тему в повідомленні.",
                "Зберегти тему в пам'яті агента.",
                "Підтвердити, що тема враховуватиметься в наступних запитах.",
            ],
            "unknown": [
                "Проаналізувати запит.",
                "Перевірити, чи є достатньо інформації для відповіді.",
                "Повідомити про обмеження, якщо запит не підтримується.",
            ],
        }
        return plans[intent]

    def respond(self, user_text):
        self.remember("user", user_text)
        low = user_text.lower()
        detected_topic = self._topic_from_text(user_text)
        if detected_topic:
            self.last_topic = detected_topic
        topic = detected_topic or self.last_topic

        # Визначення наміру та потрібного інструмента
        numbers = re.findall(r"(?<![A-Za-zА-Яа-я])\d+(?:[.,]\d+)?", user_text)
        intent = "unknown"
        tool_name = "Не використовується"

        if any(word in low for word in ["середн", "average"]):
            intent, tool_name = "average", "Calculator"
        elif any(word in low for word in ["обчисли", "порахуй", "скільки буде", "розрахуй"]):
            intent, tool_name = "calculator", "Calculator"
        elif any(word in low for word in ["план підготовки", "план навчання", "підготуй план"]):
            intent, tool_name = "study_plan", "DatePlanner"
        elif any(word in low for word in ["поясни", "що таке", "розкажи"]):
            intent = "explain"
        elif detected_topic and any(word in low for word in ["вивчаю", "вчу", "вивчаємо"]):
            intent = "topic_memory"

        plan = self._make_plan(intent)
        result = None

        try:
            if intent == "average":
                if len(numbers) < 2:
                    raise ValueError("Для середнього арифметичного потрібно щонайменше два числа.")
                values = [float(x.replace(",", ".")) for x in numbers]
                total = calculator(" + ".join(map(str, values)))
                result = total / len(values)
                answer = (
                    f"Середнє арифметичне чисел {values} дорівнює "
                    f"{result:.2f}. Воно обчислюється як сума чисел, "
                    f"поділена на їх кількість."
                )
            elif intent == "calculator":
                # Беремо вираз після двокрапки, якщо вона є, інакше — рядок запиту.
                expression = user_text.split(":", 1)[1] if ":" in user_text else user_text
                expression = expression.strip()
                expression = re.sub(
                    r"^(обчисли|порахуй|розрахуй|скільки буде)\s*",
                    "", expression, flags=re.IGNORECASE
                )
                expression = expression.replace(",", ".")
                # Залишаємо лише символи математичного виразу.
                expression = re.sub(r"[^0-9+\-*/().\s]", "", expression)
                result = calculator(expression)
                answer = f"Результат обчислення: {result}."
            elif intent == "study_plan":
                days_match = re.search(r"(\d+)\s*(дн|день|дні|днів)", low)
                days = int(days_match.group(1)) if days_match else 5
                days = max(1, min(days, 14))
                dates = date_planner(None, days)
                topic_name = topic or "обрану тему"
                tasks = [
                    "Ознайомитися з основними поняттями",
                    "Опрацювати приклади",
                    "Виконати практичні завдання",
                    "Повторити матеріал і пройти самоперевірку",
                ]
                schedule = [
                    f"{d}: {tasks[min(i, len(tasks)-1)]}"
                    for i, d in enumerate(dates)
                ]
                answer = f"План підготовки з теми «{topic_name}» на {days} дн.:\n" + "\n".join(schedule)
                result = schedule
            elif intent == "topic_memory":
                answer = f"Добре! Зберіг тему «{topic}» у пам'яті й враховуватиму її в наступних запитах."
            elif intent == "explain":
                if topic == "Python":
                    answer = "Python — це високорівнева інтерпретована мова програмування загального призначення."
                elif topic == "Java":
                    answer = "Java — об'єктноорієнтована мова програмування, код якої виконується на JVM."
                elif topic == "математика":
                    answer = "Математика вивчає кількісні відношення, структури та просторові форми."
                elif topic == "AI":
                    answer = "AI-агент — система, яка отримує мету, планує дії та може використовувати інструменти."
                else:
                    answer = "Уточніть тему запитання — наприклад, Python, Java, математика або AI."
            else:
                answer = (
                    "Я поки не вмію виконувати цей тип запиту. "
                    "Підтримую короткі пояснення з базових тем, арифметичні "
                    "обчислення, середнє арифметичне та простий план підготовки."
                )
        except (ValueError, ZeroDivisionError, SyntaxError) as error:
            answer = f"Не вдалося виконати запит: {error}"
            result = "Помилка"

        self.remember("assistant", answer)
        return {
            "plan": plan,
            "tool": tool_name,
            "result": result,
            "answer": answer,
            "memory_size": len(self.memory),
        }

    def show_history(self):
        if not self.memory:
            print("Історія поки порожня.")
            return
        print("\n--- Історія діалогу ---")
        for item in self.memory:
            label = "Користувач" if item["role"] == "user" else "Агент"
            print(f"{label}: {item['text']}")


def main():
    agent = StudentAssistantAgent()
    print("Помічник студента. Команди: /history — історія, /exit — вихід.")
    print("Приклади: «Поясни, що таке Python»; «Обчисли: (12+18)/2».")
    while True:
        try:
            user_text = input("\nВи: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершення роботи.")
            break
        if user_text.lower() in {"/exit", "exit", "вихід"}:
            print("До побачення!")
            break
        if user_text.lower() == "/history":
            agent.show_history()
            continue
        if not user_text:
            continue

        response = agent.respond(user_text)
        print("\nПлан агента:")
        for step_number, step in enumerate(response["plan"], start=1):
            print(f"  {step_number}. {step}")
        print(f"Інструмент: {response['tool']}")
        if response["result"] is not None:
            print(f"Результат інструмента: {response['result']}")
        print(f"Відповідь: {response['answer']}")
        print(f"Кількість збережених реплік: {response['memory_size']}")


if __name__ == "__main__":
    main()
