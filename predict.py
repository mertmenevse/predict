
import lightgbm as lgb
import pandas as pd
from prometheus_client import start_http_server, Gauge
from sklearn.metrics import mean_squared_error

# Modeli yükle
model = lgb.Booster(model_file="model.txt")

# Prometheus metrikleri
RMSE_GAUGE = Gauge("model_rmse", "Model RMSE Metric")

# Metrik sunucusunu başlat
start_http_server(8000)

def predict(input_data):
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    # Örnek RMSE güncellemesi (gerçek kullanımda gerçek y_true lazım)
    RMSE_GAUGE.set(pred)  
    return pred

if __name__ == "__main__":
    sample = {
        "Gender": 1,
        "Age": 30,
        "Price per Unit": 50,
        "Product Category_Beauty": 1,
        "Product Category_Clothing": 0,
        "Product Category_Electronics": 0
    }
    print("Tahmin:", predict(sample))
