import joblib
import pandas as pd


MODEL_PATH = "models/delivery_delay_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


def get_text_input(prompt, default):
    """Get a text value from the user."""
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    return value


def get_float_input(prompt, default, minimum=None):
    """Get a valid floating-point value."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip()

        if value == "":
            return default

        try:
            value = float(value)

            if minimum is not None and value < minimum:
                print(f"Please enter a value >= {minimum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


def get_integer_input(prompt, default, minimum=None, maximum=None):
    """Get a valid integer value."""
    while True:
        value = input(f"{prompt} [{default}]: ").strip()

        if value == "":
            return default

        try:
            value = int(value)

            if minimum is not None and value < minimum:
                print(f"Please enter a value >= {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Please enter a value <= {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def predict_delivery(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probabilities = model.predict_proba(df)[0]

    classes = model.classes_

    if "yes" in classes:
        delay_probability = probabilities[
            list(classes).index("yes")
        ]
    else:
        delay_probability = 0.0

    confidence = max(probabilities)

    return prediction, delay_probability, confidence



def get_risk_level(delay_probability):

    if delay_probability >= 0.70:
        return "HIGH RISK"

    elif delay_probability >= 0.40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


def display_result(
    data,
    prediction,
    delay_probability,
    confidence
):

    if prediction == "yes":
        status = "DELAYED"
    else:
        status = "ON TIME"

    risk_level = get_risk_level(delay_probability)

    print("\n" + "=" * 55)
    print("              PREDICTION RESULT")
    print("=" * 55)

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

    print(f"Delivery Cost    : Rs.{data['delivery_cost']:.2f}")

    print("\nPrediction")
    print("-" * 30)

    print(f"Status              : {status}")

    print(
        f"Delay Probability   : "
        f"{delay_probability * 100:.2f}%"
    )

    print(
        f"Prediction Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(f"Risk Level          : {risk_level}")

    print("\n" + "=" * 55)


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

    {
        **base_data,
        "distance_km": 80,
        "package_weight_kg": 5
    },

    {
        **base_data,
        "distance_km": 50,
        "package_weight_kg": 3
    },

    {
        **base_data,
        "distance_km": 100,
        "package_weight_kg": 8
    }
]



dataset = pd.read_csv("data/delivery_data.csv")

risky_candidates = []

delayed_rows = dataset[
    dataset["delayed"] == "yes"
]


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

    prediction, delay_probability, confidence = (
        predict_delivery(candidate)
    )

    if prediction == "yes":
        risky_candidates.append(candidate)
        break


while True:

    print("\n" + "=" * 55)
    print("       DELIVERY DELAY PREDICTION SYSTEM")
    print("=" * 55)

    print("\n1. Normal Delivery")
    print("2. Risky Delivery")
    print("3. Custom Delivery")
    print("4. Exit")

    choice = input("\nSelect option: ").strip()


    if choice == "1":

        selected_data = None

        for data in normal_candidates:

            prediction, delay_probability, confidence = (
                predict_delivery(data)
            )

            if prediction == "no":
                selected_data = data
                break

        if selected_data is None:

            selected_data = normal_candidates[0]

            prediction, delay_probability, confidence = (
                predict_delivery(selected_data)
            )

        display_result(
            selected_data,
            prediction,
            delay_probability,
            confidence
        )


    elif choice == "2":

        if not risky_candidates:

            print("\nNo model-verified risky delivery found.")

            continue

        selected_data = risky_candidates[0]

        prediction, delay_probability, confidence = (
            predict_delivery(selected_data)
        )

        display_result(
            selected_data,
            prediction,
            delay_probability,
            confidence
        )


    elif choice == "3":

        print("\nEnter delivery details.")
        print("Press ENTER to use the default value.")
        print("-" * 55)

        delivery_partner = get_text_input(
            "Delivery Partner",
            "delhivery"
        )

        package_type = get_text_input(
            "Package Type",
            "electronics"
        )

        vehicle_type = get_text_input(
            "Vehicle Type",
            "bike"
        )

        delivery_mode = get_text_input(
            "Delivery Mode",
            "express"
        )

        region = get_text_input(
            "Region",
            "west"
        )

        weather_condition = get_text_input(
            "Weather Condition",
            "clear"
        )

        distance_km = get_float_input(
            "Distance (km)",
            150,
            minimum=0
        )

        package_weight_kg = get_float_input(
            "Package Weight (kg)",
            10,
            minimum=0
        )

        expected_time_hours = get_float_input(
            "Expected Time (hours)",
            8,
            minimum=0
        )

        delivery_rating = get_integer_input(
            "Delivery Rating (1-5)",
            4,
            minimum=1,
            maximum=5
        )

        delivery_cost = get_float_input(
            "Delivery Cost",
            850,
            minimum=0
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

        prediction, delay_probability, confidence = (
            predict_delivery(custom_data)
        )

        display_result(
            custom_data,
            prediction,
            delay_probability,
            confidence
        )

    elif choice == "4":

        print("\nExiting Delivery Delay Prediction System.")
        break

    else:

        print(
            "\nInvalid option. "
            "Please select 1, 2, 3 or 4."
        )