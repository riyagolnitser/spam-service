import pandas as pd

path = "spam_service/messages.csv"

df = pd.read_csv(path, nrows=10)

print("Колонки:", df.columns.tolist())
print("Форма первых 10 строк:", df.shape)
print(df.head().to_string())

print("\nТипы данных:")
print(df.dtypes)

print("\nПримеры значений по колонкам:")
for col in df.columns:
    print(f"\n--- {col} ---")
    for value in df[col].head(3):
        print(str(value)[:300])