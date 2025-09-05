FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .

EXPOSE 8080

CMD ["uvicorn", "src.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
