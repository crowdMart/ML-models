from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import datetime
import joblib

# ──────────────────────────────────────────────
# 🔹 Load ML Model + CSV Data
# ──────────────────────────────────────────────
trust_model = joblib.load("trust_score_model.pkl")
drivers_df = pd.read_csv("Data_drivers.csv")
parcels_df = pd.read_csv("Data_parcel.csv")

app = FastAPI(title="Walmart Hackathon APIs")

# ──────────────────────────────────────────────
# 🔸 TRUST SCORE SECTION
# ──────────────────────────────────────────────

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

@app.post("/trust-score")
def predict_trust_score(data: TrustScoreRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = trust_model.predict(input_df)[0]
    return {"trust_score": round(float(prediction), 2)}

# ──────────────────────────────────────────────
# 🔸 DELIVERY MATCHING SECTION
# ──────────────────────────────────────────────

class MatchRequest(BaseModel):
    driver_id: str
    top_k: int = 5

def time_to_minutes(t):
    return int(datetime.strptime(t, "%H:%M").hour) * 60 + int(datetime.strptime(t, "%H:%M").minute)

def match_parcels(driver_id, top_k=5):
    driver = drivers_df[drivers_df["driver_id"] == driver_id].iloc[0]
    scores = []

    for _, parcel in parcels_df.iterrows():
        dist_to_pickup = geodesic((driver.driver_lat, driver.driver_lon), (parcel.pickup_lat, parcel.pickup_lon)).km
        delivery_distance = geodesic((parcel.pickup_lat, parcel.pickup_lon), (parcel.drop_lat, parcel.drop_lon)).km
        dest_alignment = geodesic((driver.driver_dest_lat, driver.driver_dest_lon), (parcel.drop_lat, parcel.drop_lon)).km

        driver_start = time_to_minutes(driver.available_from)
        driver_end = time_to_minutes(driver.available_until)
        fits_time = 1 if parcel.expected_delivery_time <= (driver_end - driver_start) else 0
        priority = parcel.priority

        score = (
            (dist_to_pickup * 0.3) +
            (delivery_distance * 0.2) +
            (dest_alignment * 0.3) -
            (priority * 5) -
            (fits_time * 3)
        )

        scores.append({"parcel_id": parcel.parcel_id, "score": round(score, 2)})

    scores = sorted(scores, key=lambda x: x["score"])
    return scores[:top_k]

@app.post("/predict-match")
def get_top_matches(request: MatchRequest):
    result = match_parcels(request.driver_id, request.top_k)
    return {"matches": result}
