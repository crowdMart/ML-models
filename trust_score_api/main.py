from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("trust_score_model.pkl")

# Define input format
class TrustScoreRequest(BaseModel):
    total_deliveries: int
    on_time_deliveries: int
    avg_delivery_delay: float
    cancelled_deliveries: int
    customer_rating_avg: float
    complaints_count: int
    fraud_flags: int
    days_active: int
    high_priority_jobs_done: int
    pod_scan_miss_rate: float

app = FastAPI()

@app.post("/trust-score")
def predict_trust_score(data: TrustScoreRequest):
    input_data = np.array([
        data.total_deliveries,
        data.on_time_deliveries,
        data.avg_delivery_delay,
        data.cancelled_deliveries,
        data.customer_rating_avg,
        data.complaints_count,
        data.fraud_flags,
        data.days_active,
        data.high_priority_jobs_done,
        data.pod_scan_miss_rate
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    return {"trust_score": round(float(prediction), 2)}
