
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model (preprocessing + regressor pipeline, serialized with joblib)
model = joblib.load("superkart_model.joblib")

# The set of "raw" features callers (frontend / API clients) are expected to supply.
EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# The model pipeline was trained on two additional engineered features (Price_per_Weight,
# Premium_Product_Flag - see the "Data Preprocessing" section of the training notebook).
# Rather than pushing that feature-engineering burden onto every API caller, we derive them
# here server-side from the raw fields above, using the same fixed threshold used at training
# time. This keeps the external API contract simple and stable even if the model's internal
# feature set evolves.
PREMIUM_MRP_THRESHOLD = 170  # must match PREMIUM_MRP_THRESHOLD in the training notebook


def _add_engineered_features(df):
    df = df.copy()
    if "Price_per_Weight" not in df.columns:
        df["Price_per_Weight"] = df["Product_MRP"] / df["Product_Weight"]
    if "Premium_Product_Flag" not in df.columns:
        df["Premium_Product_Flag"] = (df["Product_MRP"] > PREMIUM_MRP_THRESHOLD).astype(int)
    return df

# If the model's final step is a bagged ensemble (Random Forest / Bagging), it exposes
# `estimators_` - the spread of predictions across individual trees gives an approximate
# 95% prediction interval, which we surface in the API response when available.
_final_estimator = model.named_steps[model.steps[-1][0]] if hasattr(model, "named_steps") else None
_supports_interval = _final_estimator is not None and hasattr(_final_estimator, "estimators_")


def _predict_with_interval(input_df):
    point = model.predict(input_df).tolist()[0]
    if not _supports_interval:
        return point, None, None
    transformed = model.named_steps["columntransformer"].transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    tree_preds = np.array([tree.predict(transformed) for tree in _final_estimator.estimators_])
    lower = float(np.percentile(tree_preds, 2.5, axis=0)[0])
    upper = float(np.percentile(tree_preds, 97.5, axis=0)[0])
    return point, lower, upper


# Define a route for the home page
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction System"


# Simple health check route - useful for container orchestration / uptime checks
@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


# Define an endpoint to predict sales for a single product
@superkart_api.post("/v1/predict")
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Validate that every expected feature is present in the payload
    missing = [f for f in EXPECTED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required field(s): {missing}"}), 400

    try:
        # Extract relevant features from the input data, in the expected order
        sample = {feature: data[feature] for feature in EXPECTED_FEATURES}
        input_data = pd.DataFrame([sample])
        input_data = _add_engineered_features(input_data)

        # Make a prediction using the trained model pipeline (plus an approximate 95% interval, if supported)
        prediction, lower, upper = _predict_with_interval(input_data)

        response_payload = {"Sales": prediction}
        if lower is not None:
            response_payload["Sales_95pct_CI_Lower"] = lower
            response_payload["Sales_95pct_CI_Upper"] = upper

        # Return the prediction as a JSON response
        return jsonify(response_payload)
    except Exception as exc:
        return jsonify({"error": f"Failed to generate prediction: {exc}"}), 400


# Define an endpoint to predict sales for a batch of products
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' found in the request."}), 400

    file = request.files["file"]

    try:
        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(file)

        missing = [f for f in EXPECTED_FEATURES if f not in input_data.columns]
        if missing:
            return jsonify({"error": f"Missing required column(s): {missing}"}), 400

        # Make predictions for the batch data (deriving the same engineered features used at training time)
        input_data = _add_engineered_features(input_data[EXPECTED_FEATURES])
        predictions = model.predict(input_data).tolist()

        # Create an output dictionary mapping row index (as string) to predicted sales
        result = {str(idx): pred for idx, pred in enumerate(predictions)}
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Failed to generate batch predictions: {exc}"}), 400


# Allows the app to also be run directly (e.g. `python app.py`) for quick local testing,
# in addition to being served by Gunicorn in the Docker container (see Dockerfile below).
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
