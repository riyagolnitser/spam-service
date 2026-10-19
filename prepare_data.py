# -*- coding: utf-8 -*-
"""
Подготовка единого датасета для бинарной классификации spam/ham.
Источники:
1. SpamAssassin .eml: easy_ham, hard_ham, spam_2
2. messages.csv (LingSpam-подобный датасет)

Результат:
- data/processed/full_dataset.csv
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv
- data_card.md
"""

from pathlib import Path
import email
from email import policy
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE = Path(".")
SPAMASSASSIN = BASE / "spam_service"
OUTPUT = BASE / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = SCRIPT_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip().lower()


def parse_eml(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        subject = str(msg.get("Subject", ""))

        plain_parts, html_parts = [], []

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    try:
                        plain_parts.append(part.get_content())
                    except Exception:
                        pass
                elif ctype == "text/html":
                    try:
                        html_parts.append(part.get_content())
                    except Exception:
                        pass
        else:
            ctype = msg.get_content_type()
            try:
                content = msg.get_content()
            except Exception:
                content = ""

            if ctype == "text/plain":
                plain_parts.append(content)
            elif ctype == "text/html":
                html_parts.append(content)

        body = "\n".join(plain_parts).strip()
        if not body:
            body = clean_text("\n".join(html_parts))

        return clean_text(subject + "\n" + body)

    except Exception:
        return ""


def load_spamassassin() -> pd.DataFrame:
    sources = {
        "easy_ham/easy_ham": 0,
        "hard_ham/hard_ham": 0,
        "spam_2/spam_2": 1,
    }

    rows = []

    for folder, label in sources.items():
        folder_path = SPAMASSASSIN / folder
        files = [f for f in folder_path.iterdir() if f.is_file()]
        print(f"SpamAssassin {folder}: {len(files)} файлов")

        for file in files:
            text = parse_eml(file)
            if text:
                rows.append({
                    "text": text,
                    "label_spam": label,
                    "source": "spamassassin",
                })

    return pd.DataFrame(rows)


def load_messages_csv() -> pd.DataFrame:
    path = SPAMASSASSIN / "messages.csv"
    df = pd.read_csv(path)

    df["text"] = (
        df["subject"].fillna("").astype(str)
        + "\n"
        + df["message"].fillna("").astype(str)
    )
    df["text"] = df["text"].apply(clean_text)
    df["label_spam"] = df["label"].astype(int)
    df["source"] = "messages_csv"

    return df[["text", "label_spam", "source"]]


def main():
    print("Загрузка SpamAssassin...")
    df_eml = load_spamassassin()

    print("Загрузка messages.csv...")
    df_csv = load_messages_csv()

    df = pd.concat([df_eml, df_csv], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(df)

    print(f"\nДо удаления дубликатов: {before}")
    print(f"После удаления дубликатов: {after}")
    print(f"Удалено дубликатов: {before - after}")

    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    print("\nИтоговый размер:", df.shape)
    print("\nБаланс классов:")
    print(df["label_spam"].value_counts())
    print("\nПо источникам:")
    print(df.groupby(["source", "label_spam"]).size())

    df.to_csv(OUTPUT / "full_dataset.csv", index=False, encoding="utf-8-sig")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label_spam"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_spam"],
    )

    train_df.to_csv(OUTPUT / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT / "test.csv", index=False, encoding="utf-8-sig")

    data_card = f"""# Data Card

## Источники данных

1. **SpamAssassin**
   - `easy_ham`: 2551 письмо, метка 0
   - `hard_ham`: 250 писем, метка 0
   - `spam_2`: 1397 писем, метка 1

2. **messages.csv**
   - 2893 письма
   - 2412 писем с меткой 0
   - 481 письмо с меткой 1

## Обработка

- Извлечены тема и тело письма.
- HTML очищен от тегов, скриптов и стилей.
- Текст приведён к нижнему регистру.
- Удалены пустые записи.
- Удалены дубликаты по нормализованному тексту.
- Данные разделены стратифицированно: train 70%, validation 15%, test 15%.

## Итоговый датасет

- Всего записей после очистки: {len(df)}
- Спам: {int((df['label_spam'] == 1).sum())}
- Не-спам: {int((df['label_spam'] == 0).sum())}
- Доля спама: {(df['label_spam'] == 1).mean():.4f}

## Разделение

- Train: {len(train_df)}
- Validation: {len(val_df)}
- Test: {len(test_df)}

## Ограничения

- Датасеты на английском языке.
- Возможны различия в стиле и качестве текста между источниками.
- Тестовый набор изолирован и не используется для подбора гиперпараметров или порога.
"""

    (BASE / "data_card.md").write_text(data_card, encoding="utf-8")

    print("\nФайлы сохранены:")
    print(OUTPUT / "full_dataset.csv")
    print(OUTPUT / "train.csv")
    print(OUTPUT / "val.csv")
    print(OUTPUT / "test.csv")
    print(BASE / "data_card.md")


if __name__ == "__main__":
    main()