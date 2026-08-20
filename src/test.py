import joblib
import pandas as pd

model = joblib.load("models/delivery_delay_model.pkl")

new_delivery = pd.DataFrame([{
    "delivery_id": 25001,
    "delivery_partner": "delhivery",
    "package_type": "electronics",
    "vehicle_type": "bike",
    "delivery_mode": "express",
    "region": "west",
    "weather_condition": "rainy",
    "distance_km": 250,
    "package_weight_kg": 20,
    "expected_time_hours": 8,
    "delivery_rating": 4,
    "delivery_cost": 1200
}])

prediction = model.predict(new_delivery)[0]

print("Delivery Delay Prediction:", prediction)

if prediction == "yes":
    print("Result: Delivery is likely to be delayed.")
else:
    print("Result: Delivery is likely to be on time.")