# Spam Classification API

ML-сервис для бинарной классификации писем: `spam` / `ham`.

## Задача

Модель определяет, является ли письмо спамом. Приоритет — минимизация ложноположительных срабатываний: важно не отправить легитимное письмо в спам.

## Данные

Использованы два источника:

- SpamAssassin: `easy_ham`, `hard_ham`, `spam_2`
- `messages.csv`

После очистки и удаления дубликатов: 6 837 писем.

- Spam: 1 792
- Ham: 5 045

## Модель

- TF-IDF: слова и биграммы
- Ручные признаки: длина темы, CAPS, восклицательные знаки, ссылки, HTML
- Logistic Regression
- Порог классификации подобран при условии Recall(spam) >= 0.90

## Результаты на тесте

- Precision(spam): 1.0000
- Recall(spam): 0.9294
- F1(spam): 0.9634
- Ложноположительные срабатывания: 0

## Запуск сервиса

```bash
pip install -r requirements.txt
python -m uvicorn service.app.main:app --reload --port 8000
```

Swagger:

http://127.0.0.1:8000/docs

## Пример запроса

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Congratulations! You have won a free iPhone. Click http://example.com now!\"}"
```

## Docker

```bash
docker build -t spam-service .
docker run -p 8000:8000 spam-service
```