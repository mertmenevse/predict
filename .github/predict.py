from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter
import logging

# Flask uygulamasını başlat
app = Flask(__name__)

# Loglama yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modeli yükle
try:
    model = lgb.Booster(model_file='model.txt')
    logger.info("Model başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yükleme hatası: {str(e)}")
    raise SystemExit(1)

# Prometheus metrikleri
start_http_server(8000)  # Prometheus metrikleri için port
REQUEST_COUNTER = Counter('model_requests_total', 'Toplam Tahmin İsteği Sayısı')
PREDICTION_GAUGE = Gauge('model_predictions', 'Model Tahmin Değerleri')
ERROR_COUNTER = Counter('model_errors_total', 'Toplam Hata Sayısı')
LATENCY_HISTOGRAM = Gauge('model_request_latency_seconds', 'İstek Gecikme Süresi')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        REQUEST_COUNTER.inc()

        # Gelen veriyi al ve doğrula
        if not request.is_json:
            ERROR_COUNTER.inc()
            return jsonify({"error": "Geçersiz veri formatı", "status": "error"}), 400

        input_data = request.json

        # Zorunlu alan kontrolü
        required_fields = ['Gender', 'Age', 'Price per Unit',
                           'Product Category_Beauty',
                           'Product Category_Clothing',
                           'Product Category_Electronics']

        if not all(field in input_data for field in required_fields):
            ERROR_COUNTER.inc()
            return jsonify({"error": "Eksik veri alanları", "status": "error"}), 400

        # DataFrame'e çevir
        df = pd.DataFrame([input_data])

        # Tahmin yap
        prediction = model.predict(df)[0]

        # Metrikleri güncelle
        PREDICTION_GAUGE.set(prediction)
        LATENCY_HISTOGRAM.set(time.time() - start_time)

        return jsonify({
            "prediction": float(prediction),
            "status": "success",
            "model_version": "1.0.0"
        })

    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Tahmin hatası: {str(e)}")
        return jsonify({
            "error": "İç sunucu hatası",
            "status": "error",
            "details": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "ready": True
    })


@app.route('/metrics', methods=['GET'])
def custom_metrics():
    # Prometheus istemcisi otomatik olarak /metrics endpoint'ini yönettiği için
    # Bu sadece örnek bir custom endpoint
    return jsonify({
        "active_requests": REQUEST_COUNTER._value.get()
    })


if __name__ == '__main__':
    # Her iki portu da açık bırakmak için
    from threading import Thread

    Thread(target=lambda: app.run(host='0.0.0.0', port=5000)).start()
