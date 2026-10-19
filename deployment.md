# Deployment

## Локальный запуск

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Запустить FastAPI:

```bash
python -m uvicorn service.app.main:app --reload --port 8000
```

3. Открыть Swagger:

http://127.0.0.1:8000/docs

4. Проверить healthcheck:

http://127.0.0.1:8000/health

5. Отправить POST-запрос на `/predict`.

## Скриншоты

- Скриншот терминала с успешным запуском Uvicorn
- Скриншот Swagger `/docs`
- Скриншот запроса `/predict`
- Скриншот ответа JSON
- Скриншот `/health`