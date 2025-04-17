FROM python:3.9-slim

# Güvenlik için non-root kullanıcı
RUN useradd -m appuser
USER appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Portları aç
EXPOSE 5000 8000

CMD ["python", "predict.py"]