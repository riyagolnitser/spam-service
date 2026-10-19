import pandas as pd

path = "spam_service/messages.csv"
df = pd.read_csv(path)

print("Размер:", df.shape)
print("\nКолонки:", df.columns.tolist())

print("\nПропуски:")
print(df.isna().sum())

print("\nРаспределение label:")
print(df["label"].value_counts(dropna=False))
print("\nДоля классов:")
print(df["label"].value_counts(normalize=True, dropna=False))

df["text"] = df["subject"].fillna("") + "\n" + df["message"].fillna("")
df["text_clean"] = df["text"].str.strip().str.lower()

print("\nПустые тексты:", (df["text_clean"] == "").sum())
print("Дубликаты по subject + message:", df.duplicated(subset=["subject", "message"]).sum())
print("Дубликаты по очищенному тексту:", df.duplicated(subset=["text_clean"]).sum())

print("\nПримеры спама:")
print(df.loc[df["label"] == 1, "text"].head(3).to_string())

print("\nПримеры не-спама:")
print(df.loc[df["label"] == 0, "text"].head(3).to_string())