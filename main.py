# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# ──────────────────────────────────────────────
# 🔹 Initialize App & Load ML Models
# ──────────────────────────────────────────────
app = FastAPI(title="Walmart Hackathon APIs 🚚")

# Load ML Models
trust_model = joblib.load("trust_score_model.pkl")
pricing_model = joblib.load("dynamic_pricing_model.pkl")
customer_anomaly_model = joblib.load("customer_anomaly_model.pkl")
pod_model = joblib.load("pod_placement_model.pkl")
incentive_model = joblib.load("incentive_model.pkl")

# Load CSVs
drivers_df = pd.read_csv("Data_drivers.csv")
parcels_df = pd.read_csv("Data_parcel.csv")
full_df = pd.read_csv("delivery_locations_india.csv")

# ──────────────────────────────────────────────
# 🔸 Root Health Check
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Walmart Hackathon API is running 🚀"}

# ──────────────────────────────────────────────
# 🔸 TRUST SCORE API
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
# 🔸 DELIVERY MATCHING API
# ──────────────────────────────────────────────
class MatchRequest(BaseModel):
    driver_id: str
    top_k: int = 5

def time_to_minutes(t):
    dt = datetime.strptime(t, "%H:%M")
    return dt.hour * 60 + dt.minute

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

# ──────────────────────────────────────────────
# 🔸 DYNAMIC PRICING API
# ──────────────────────────────────────────────
class PricingRequest(BaseModel):
    distance_km: float
    parcel_weight_kg: float
    traffic_level: str
    delivery_time_slot: str
    is_urgent: int
    weather_condition: str
    base_price: float

@app.post("/predict-price")
def predict_dynamic_price(data: PricingRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = pricing_model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}

# ──────────────────────────────────────────────
# 🔸 CUSTOMER ANOMALY DETECTION API
# ──────────────────────────────────────────────
class CustomerAnomalyRequest(BaseModel):
    total_orders: int
    return_count: int
    avg_return_time_days: float
    value_returned_total: float
    high_value_items_returned: int
    reported_wrong_item: int
    refund_disputes_raised: int
    customer_rating_by_sellers: float
    fraud_flag_previous: int

@app.post("/detect-customer-anomaly")
def detect_customer_anomaly(data: CustomerAnomalyRequest):
    input_df = pd.DataFrame([data.dict()])
    prob = customer_anomaly_model.predict_proba(input_df)[0][1]
    is_anomaly = prob >= 0.82
    return {
        "is_anomaly": is_anomaly,
        "confidence": round(float(prob), 4)
    }

# ──────────────────────────────────────────────
# 🔸 POD PLACEMENT OPTIMIZER API
# ──────────────────────────────────────────────
class Location(BaseModel):
    lat: float
    lon: float

class PodPlacementRequest(BaseModel):
    city: Optional[str] = None
    locations: Optional[List[Location]] = None
    num_pods: int = 5

@app.post("/recommend-pods")
def recommend_dynamic_pods(data: PodPlacementRequest):
    if data.city:
        coords = full_df[full_df["city"].str.lower() == data.city.lower()][['drop_lat', 'drop_lon']]
        if coords.empty:
            return {"error": f"No data found for city: {data.city}"}
        if len(coords) < data.num_pods:
            return {"error": f"Not enough data points in {data.city} to suggest {data.num_pods} pods."}
    elif data.locations and len(data.locations) >= data.num_pods:
        coords = pd.DataFrame([{"drop_lat": loc.lat, "drop_lon": loc.lon} for loc in data.locations])
    else:
        return {"error": "Please provide a valid 'city' name or at least 'num_pods' locations."}

    kmeans = KMeans(n_clusters=data.num_pods, random_state=42)
    kmeans.fit(coords)

    centers = kmeans.cluster_centers_
    return {
        "recommended_pods": [
            {
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "google_maps_url": f"https://www.google.com/maps?q={round(float(lat), 6)},{round(float(lon), 6)}"
            }
            for lat, lon in centers
        ]
    }

# ──────────────────────────────────────────────
# 🔸 INCENTIVE RECOMMENDATION API
# ──────────────────────────────────────────────
class IncentiveRequest(BaseModel):
    total_deliveries: int
    on_time_rate: float
    avg_delivery_rating: float
    high_priority_deliveries: int
    hours_active_today: float
    current_zone_demand: float
    completed_back_to_back_jobs: int
    late_night_jobs_done: int

@app.post("/recommend-incentive")
def recommend_incentive(data: IncentiveRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = incentive_model.predict(input_df)[0]
    return {"recommended_incentive": round(float(prediction), 2)}
