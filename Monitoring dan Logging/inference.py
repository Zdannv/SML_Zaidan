
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import time
import psutil
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = FastAPI(title="Wine Quality API")

model = mlflow.pyfunc.load_model("/Users/macbook/Documents/SMSML_Zaidan/Membangun_model/mlruns/614510730523264548/7410ba5bdc3d4b93ba786be5e5e2c7f2/artifacts/model")

REQUEST_COUNT = Counter("prediction_requests_total","Total prediction requests")
LATENCY = Histogram("prediction_latency_seconds","Prediction latency")
CPU = Gauge("cpu_usage_percent","CPU usage")
MEM = Gauge("memory_usage_percent","Memory usage")
PRED = Counter("prediction_result_total","Prediction result",["quality"])

class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def home():
    return {"status":"running"}

@app.get("/metrics")
def metrics():
    CPU.set(psutil.cpu_percent())
    MEM.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
def predict(data: Wine):
    REQUEST_COUNT.inc()
    start=time.time()
    df=pd.DataFrame([data.model_dump()])
    df.columns = [c.replace("_", " ") for c in df.columns]
    pred=model.predict(df)[0]
    PRED.labels(quality=str(pred)).inc()
    LATENCY.observe(time.time()-start)
    return {"prediction": int(pred)}
