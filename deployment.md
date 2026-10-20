# Запуск и публикация Spam Classification API

## 1. Статус развёртывания

Приложение опубликовано на Render как Docker Web Service.

Подтверждено:

- Репозиторий загружен с GitHub.
- Docker-образ успешно собран.
- Зависимости установлены.
- FastAPI-приложение запущено на `0.0.0.0:8000`.
- Render выдал статус `Your service is live`.
- Публичный `/health` подтвердил загрузку модели.
- Публичный `/predict` успешно обработал запрос классификации.

Дата проверки: 20 октября 2026 года.

## 2. Публичные адреса

| Назначение | Адрес |
|---|---|
| Главная страница | https://spam-classification-api-2zf0.onrender.com |
| Swagger UI | https://spam-classification-api-2zf0.onrender.com/docs |
| Проверка состояния | https://spam-classification-api-2zf0.onrender.com/health |
| OpenAPI | https://spam-classification-api-2zf0.onrender.com/openapi.json |

Эндпоинт классификации:

```text
POST [https://spam-classification-api-2zf0.onrender.com/predict](https://spam-classification-api-2zf0.onrender.com/predict)
```

Репозиторий:

```text
[https://github.com/riyagolnitser/spam-service](https://github.com/riyagolnitser/spam-service)
```

Репозиторий закрытый. Доступ к исходному коду предоставляется отдельно.

## 3. Состав Docker-образа

В образ включены:

- зависимости из `requirements.txt`;
- сохранённая модель `spam_model.joblib`;
- код приложения из папки `service/`.

Исходные датасеты и скрипты обучения не копируются в образ командами Dockerfile.

Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY spam_model.joblib .
COPY service ./service

EXPOSE 8000

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. Логика работы сервиса

1. При запуске загружается `spam_model.joblib`.
2. Из артефакта извлекаются TF-IDF-векторизатор, классификатор и порог.
3. Клиент отправляет JSON с полем `text`.
4. Приложение рассчитывает текстовые и дополнительные числовые признаки.
5. Модель возвращает оценку вероятности спама.
6. Вероятность сравнивается с сохранённым порогом.
7. Клиент получает JSON с классом, вероятностью и флагом ручной проверки.

Текущая реализация API не полностью повторяет предобработку подготовки данных. Это зафиксированное ограничение, требующее дальнейшего исправления.

## 5. Проверка публичного сервиса

### Проверка состояния

PowerShell:

```powershell
Invoke-RestMethod `
    -Uri "https://spam-classification-api-2zf0.onrender.com/health"
```

Полученный ответ:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Проверка классификации

```powershell
$body = @{
    text = "Congratulations! You have won a free iPhone. Click now to claim your prize!"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://spam-classification-api-2zf0.onrender.com/predict" `
    -Method Post `
    -Body $body `
    -ContentType "application/json"
```

Полученный ответ:

```json
{
  "label": "spam",
  "label_id": 1,
  "confidence": 0.999879758628561,
  "spam_probability": 0.999879758628561,
  "needs_human_review": false
}
```

Эта проверка подтверждает работоспособность опубликованного API. Она не заменяет оценку качества на тестовом наборе.

## 6. Локальный запуск без Docker

Требования:

- Python 3.12.
- Файл модели в корне проекта.
- Установленные зависимости.

Команды выполняются из корня проекта:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn service.app.main:app --reload --port 8000
```

После запуска:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

Терминал с сервером необходимо оставить открытым. Для остановки нажать `Ctrl + C`.

## 7. Локальный запуск через Docker

Инструкция подготовлена для повторного запуска. На личном ноутбуке Docker-сборка не проверялась; успешная сборка и запуск выполнены на Render.

```powershell
docker build -t spam-service .
docker run --rm -p 8000:8000 --name spam-api spam-service
```

Если порт 8000 занят:

```powershell
docker run --rm -p 8001:8000 --name spam-api spam-service
```

Swagger во втором случае:

```text
http://127.0.0.1:8001/docs
```

## 8. Параметры публикации на Render

- Тип сервиса: Web Service.
- Репозиторий: `riyagolnitser/spam-service`.
- Ветка: `main`.
- Среда: Docker.
- Dockerfile: в корне репозитория.
- Команда запуска: из Dockerfile.
- Адрес прослушивания: `0.0.0.0`.
- Порт приложения: `8000`.

## 9. Зависимости и воспроизводимость

В логах успешной облачной сборки установлены:

- NumPy 2.3.4.
- SciPy 1.16.3.

Ранее в локальном окружении обучения были зафиксированы:

- NumPy 2.1.3.
- SciPy 1.17.0.

Приложение успешно загрузило модель и выполнило проверочный запрос при указанном различии версий. Полная эквивалентность локальных и облачных предсказаний не проверялась.

Для улучшения воспроизводимости необходимо согласовать версии окружения обучения и сервиса, затем повторить проверку.

## 10. Демонстрация работы

Скриншоты успешного развёртывания на Render, Swagger UI и ответов API представлены в презентации проекта.

Работоспособность сервиса также можно проверить по публичным адресам из раздела 2.