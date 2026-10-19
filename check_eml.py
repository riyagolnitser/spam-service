from pathlib import Path
import email
from email import policy

base = Path("spam_service")

for folder in ["easy_ham/easy_ham", "hard_ham/hard_ham", "spam_2/spam_2"]:
    files = [f for f in (base / folder).iterdir() if f.is_file()]
    print("\n" + folder)
    print("Всего файлов:", len(files))

    if files:
        sample = files[0]
        print("Пример файла:", sample.name)
        try:
            with open(sample, "rb") as f:
                msg = email.message_from_binary_file(f, policy=policy.default)
            print("Subject:", str(msg.get("Subject", ""))[:120])
            print("Content-Type:", msg.get_content_type())
        except Exception as e:
            print("Ошибка чтения:", e)