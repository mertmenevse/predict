FROM python:3.9-slim

# Gerekli sistem paketlerini yükle
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Uygulama dizini
WORKDIR /app

# Gerekli Python paketlerini yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Kullanıcı oluştur (opsiyonel ama önerilir)
RUN useradd -m appuser
USER appuser

# Varsayılan komut
CMD ["python", "predict.py"]