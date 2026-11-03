
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY artifacts/activity_ml/best_model.pt \
     /app/artifacts/activity_ml/best_model.pt

ENV HAR_MODEL_PATH=/app/artifacts/activity_ml/best_model.pt

EXPOSE 8000

CMD ["uvicorn", "activity_ml.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]