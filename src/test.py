import joblib
import pandas as pd


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/delivery_delay_model.pkl")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def predict_delivery(data):
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    classes = model.classes_

    if "yes" in classes:
        delay_probability = probabilities[list(classes).index("yes")]
    else:
        delay_probability = 0.0

    confidence = max(probabilities)

    return prediction, delay_probability, confidence


def display_result(data, prediction, delay_probability, confidence):

    if prediction == "yes":
        status = "DELAYED"
    else:
        status = "ON TIME"

    print("\n" + "=" * 50)
    print("              PREDICTION RESULT")
    print("=" * 50)

    print("\nDelivery Details")
    print("-" * 30)

    print(f"Partner          : {data['delivery_partner']}")
    print(f"Package          : {data['package_type']}")
    print(f"Vehicle          : {data['vehicle_type']}")
    print(f"Mode             : {data['delivery_mode']}")
    print(f"Region           : {data['region']}")
    print(f"Weather          : {data['weather_condition']}")
    print(f"Distance         : {data['distance_km']:.1f} km")
    print(f"Package Weight   : {data['package_weight_kg']:.1f} kg")
    print(f"Expected Time    : {data['expected_time_hours']:.1f} hours")
    print(f"Delivery Rating  : {data['delivery_rating']}")
    print(f"Delivery Cost    : ₹{data['delivery_cost']:.2f}")

    print("\nPrediction")
    print("-" * 30)

    print(f"Status           : {status}")
    print(f"Delay Probability: {delay_probability * 100:.2f}%")
    print(f"Prediction Confidence: {confidence * 100:.2f}%")

    print("\n" + "=" * 50)


# ==========================================
# DEMO SCENARIOS
# ==========================================

base_data = {
    "delivery_id": 25001,
    "delivery_partner": "delhivery",
    "package_type": "electronics",
    "vehicle_type": "bike",
    "delivery_mode": "express",
    "region": "west",
    "weather_condition": "clear",
    "distance_km": 150,
    "package_weight_kg": 10,
    "expected_time_hours": 8,
    "delivery_rating": 4,
    "delivery_cost": 850
}


normal_candidates = [
    base_data.copy(),
    {**base_data, "distance_km": 80, "package_weight_kg": 5},
    {**base_data, "distance_km": 50, "package_weight_kg": 3},
    {**base_data, "distance_km": 100, "package_weight_kg": 8}
]


# ==========================================
# FIND A MODEL-VERIFIED DELAYED EXAMPLE
# ==========================================

dataset = pd.read_csv("data/delivery_data.csv")

feature_columns = [
    "delivery_partner",
    "package_type",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
    "distance_km",
    "package_weight_kg",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"
]

risky_candidates = []

delayed_rows = dataset[dataset["delayed"] == "yes"]

for _, row in delayed_rows.iterrows():

    candidate = {
        "delivery_id": int(row["delivery_id"]),
        "delivery_partner": row["delivery_partner"],
        "package_type": row["package_type"],
        "vehicle_type": row["vehicle_type"],
        "delivery_mode": row["delivery_mode"],
        "region": row["region"],
        "weather_condition": row["weather_condition"],
        "distance_km": row["distance_km"],
        "package_weight_kg": row["package_weight_kg"],
        "expected_time_hours": row["expected_time_hours"],
        "delivery_rating": row["delivery_rating"],
        "delivery_cost": row["delivery_cost"]
    }

    prediction, delay_probability, confidence = predict_delivery(candidate)

    if prediction == "yes":
        risky_candidates.append(candidate)
        break


# ==========================================
# MAIN MENU
# ==========================================

while True:

    print("\n" + "=" * 50)
    print("       DELIVERY DELAY PREDICTION SYSTEM")
    print("=" * 50)

    print("\n1. Normal Delivery")
    print("2. Risky Delivery")
    print("3. Custom Delivery")
    print("4. Exit")

    choice = input("\nSelect option: ")

    # --------------------------------------
    # NORMAL DELIVERY
    # --------------------------------------

    if choice == "1":

        selected_data = None

        for data in normal_candidates:
            prediction, delay_probability, confidence = predict_delivery(data)

            if prediction == "no":
                selected_data = data
                break

        if selected_data is None:
            selected_data = normal_candidates[0]
            prediction, delay_probability, confidence = predict_delivery(
                selected_data
            )

        display_result(
            selected_data,
            prediction,
            delay_probability,
            confidence
        )

    # --------------------------------------
    # RISKY DELIVERY
    # --------------------------------------

    elif choice == "2":

        selected_data = None

        for data in risky_candidates:
            prediction, delay_probability, confidence = predict_delivery(data)

            if prediction == "yes":
                selected_data = data
                break

        if selected_data is None:
            selected_data = risky_candidates[0]
            prediction, delay_probability, confidence = predict_delivery(
                selected_data
            )

        display_result(
            selected_data,
            prediction,
            delay_probability,
            confidence
        )

    # --------------------------------------
    # CUSTOM DELIVERY
    # --------------------------------------

    elif choice == "3":

        print("\nEnter delivery details.")
        print("Press ENTER to use the default sample values.")
        print("-" * 50)

        delivery_partner = input(
            "Delivery Partner [delhivery]: "
        ) or "delhivery"

        package_type = input(
            "Package Type [electronics]: "
        ) or "electronics"

        vehicle_type = input(
            "Vehicle Type [bike]: "
        ) or "bike"

        delivery_mode = input(
            "Delivery Mode [express]: "
        ) or "express"

        region = input(
            "Region [west]: "
        ) or "west"

        weather_condition = input(
            "Weather Condition [clear]: "
        ) or "clear"

        distance_km = float(
            input("Distance (km) [150]: ") or 150
        )

        package_weight_kg = float(
            input("Package Weight (kg) [10]: ") or 10
        )

        expected_time_hours = float(
            input("Expected Time (hours) [8]: ") or 8
        )

        delivery_rating = int(
            input("Delivery Rating (1-5) [4]: ") or 4
        )

        delivery_cost = float(
            input("Delivery Cost [850]: ") or 850
        )

        custom_data = {
            "delivery_id": 25001,
            "delivery_partner": delivery_partner,
            "package_type": package_type,
            "vehicle_type": vehicle_type,
            "delivery_mode": delivery_mode,
            "region": region,
            "weather_condition": weather_condition,
            "distance_km": distance_km,
            "package_weight_kg": package_weight_kg,
            "expected_time_hours": expected_time_hours,
            "delivery_rating": delivery_rating,
            "delivery_cost": delivery_cost
        }

        prediction, delay_probability, confidence = predict_delivery(
            custom_data
        )

        display_result(
            custom_data,
            prediction,
            delay_probability,
            confidence
        )

    # --------------------------------------
    # EXIT
    # --------------------------------------

    elif choice == "4":

        print("\nExiting Delivery Delay Prediction System.")
        break

    else:

        print("\nInvalid option. Please select 1, 2, 3 or 4.")