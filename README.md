# 🚀 Walmart Hackathon – AIML API System

Welcome to the official repository for our Walmart Hackathon submission! This project showcases a microservice-based AI/ML backend system that optimizes delivery partner evaluation and intelligent parcel assignment.

---

## 📦 Features Built

---

### 1. **Trust Score Model API**

#### 🔹 Endpoint

`POST /trust-score`

#### 🔍 Purpose

Predicts a delivery partner's trustworthiness on a scale of 0 to 100 based on performance and behavioral data. This helps:

* Prioritize reliable drivers
* De-prioritize underperformers
* Drive smarter assignment logic

#### 📊 Features Used

* Total Deliveries
* On-Time Deliveries
* Avg. Delivery Delay
* Cancelled Deliveries
* Customer Rating
* Complaint Count
* Fraud Flags
* Days Active
* High-Priority Jobs Done
* Pod Scan Miss Rate

#### 🧐 ML Model

* **Random Forest Regressor**
* R² Score: **0.994**
* MAE: **0.34**

#### 🧪 Sample Request

```json
{
  "total_deliveries": 120,
  "on_time_deliveries": 110,
  "avg_delivery_delay": 2.5,
  "cancelled_deliveries": 4,
  "customer_rating_avg": 4.3,
  "complaints_count": 1,
  "fraud_flags": 0,
  "days_active": 190,
  "high_priority_jobs_done": 30,
  "pod_scan_miss_rate": 0.05
}
```

#### ✅ Sample Response

```json
{
  "trust_score": 89.57
}
```

---

### 2. **Delivery Matching Model API**

#### 🔹 Endpoint

`POST /predict-match`

#### 🔍 Purpose

Matches a driver to the most suitable parcel deliveries based on location, intent, availability, and parcel priority.

#### 📊 Inputs

* Driver's Current Location
* Driver’s Destination
* Time Window
* Parcel Pickup/Drop Coordinates
* Parcel Priority

#### ⚙️ Scoring Logic

* Lower score = better match
* Factors: pickup distance, route alignment, time fit, and delivery urgency

#### 🧪 Sample Request

```json
{
  "driver_id": "D003",
  "top_k": 5
}
```

#### ✅ Sample Response

```json
{
  "matches": [
    {"parcel_id": "P018", "score": 4.57},
    {"parcel_id": "P004", "score": 5.13},
    {"parcel_id": "P020", "score": 5.99}
  ]
}
```

---

### 3. **Dynamic Pricing Model API**

#### 🔹 Endpoint

`POST /predict-price`

#### 🔍 Purpose

Recommends optimal delivery pricing based on demand, parcel characteristics, traffic, weather, and urgency.

#### 📊 Features Used

* Distance
* Weight
* Traffic Level
* Delivery Time Slot
* Weather Condition
* Urgency Flag
* Base Price

#### 🧐 ML Model

* **XGBoost Regressor**
* R² Score: **0.98**

#### 🧪 Sample Request

```json
{
  "distance_km": 12.5,
  "parcel_weight_kg": 3.2,
  "traffic_level": "High",
  "delivery_time_slot": "Evening",
  "is_urgent": 1,
  "weather_condition": "Rainy",
  "base_price": 50
}
```

#### ✅ Sample Response

```json
{
  "predicted_price": 72.85
}
```

---

### 4. **Customer Anomaly Detection API**

#### 🔹 Endpoint

`POST /detect-customer-anomaly`

#### 🔍 Purpose

Detects potential fraud-prone customers using behavioral data and return history.

#### 📊 Features Used

* Total Orders
* Returns
* Return Time
* Value Returned
* Disputes
* High-value Returns
* Seller Ratings
* Historical Flags

#### 🧐 ML Model

* **Logistic Regression**
* ROC AUC: **0.91**

#### ☟️ Threshold

Only returns `true` if the anomaly confidence is ≥ **82%**

#### 🧪 Sample Request

```json
{
  "total_orders": 55,
  "return_count": 12,
  "avg_return_time_days": 2.5,
  "value_returned_total": 8900,
  "high_value_items_returned": 5,
  "reported_wrong_item": 3,
  "refund_disputes_raised": 2,
  "customer_rating_by_sellers": 3.2,
  "fraud_flag_previous": 1
}
```

#### ✅ Sample Response

```json
{
  "is_anomaly": true,
  "confidence": 0.8742
}
```

---

### 5. **Pod Placement Optimizer API**

#### 🔹 Endpoint

`POST /recommend-pods`

#### 🔍 Purpose

Uses unsupervised clustering to suggest optimal pod drop locations based on drop-off density.

#### 🔗 Google Maps Links Included!

Each pod recommendation links to the exact spot.

#### 📊 Inputs

* `city`: Name of Indian city (optional)
* `locations`: List of coordinates (optional)
* `num_pods`: Number of pods (default = 5)

#### 🧐 ML Model

* **KMeans Clustering**
* Trained on **10,000+ delivery points across India**

#### 🧪 Sample Request

```json
{
  "city": "Bangalore",
  "num_pods": 4
}
```

#### ✅ Sample Response

```json
{
  "recommended_pods": [
    {
      "lat": 12.96123,
      "lon": 77.59445,
      "google_maps_url": "https://www.google.com/maps?q=12.96123,77.59445"
    },
    ...
  ]
}
```

---

### 6. **Incentive Recommendation Model API**

#### 🔹 Endpoint

`POST /recommend-incentive`

#### 🔍 Purpose

Recommends incentive amount based on delivery performance, current demand, and job difficulty.

#### 📊 Features Used

* Total Deliveries
* On-Time Rate
* Avg. Rating
* High-Priority Jobs
* Active Hours
* Zone Demand
* Back-to-Back Jobs
* Late-Night Deliveries

#### 🧐 ML Model

* **Gradient Boosting Regressor**
* R² Score: **0.965**

#### 🧪 Sample Request

```json
{
  "total_deliveries": 85,
  "on_time_rate": 0.93,
  "avg_delivery_rating": 4.7,
  "high_priority_deliveries": 15,
  "hours_active_today": 7.5,
  "current_zone_demand": 0.82,
  "completed_back_to_back_jobs": 12,
  "late_night_jobs_done": 5
}
```

#### ✅ Sample Response

```json
{
  "recommended_incentive": 375.75
}
```

---

## 🛠️ How to Run

1. Clone the repository

2. Ensure the following files exist:

   * `main.py` (FastAPI code)
   * All `*_model.pkl` files
   * `Data_drivers.csv`
   * `Data_parcel.csv`
   * `delivery_locations_india.csv`

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the server:

```bash
uvicorn main:app --reload
```

5. Open in browser: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

> Built with ❤️ for the Walmart Hackathon by a team passionate about AI in logistics.
