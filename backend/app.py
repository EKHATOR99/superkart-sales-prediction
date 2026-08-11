
# Import necessary libraries
import os
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Cap request body size (~5 MB is generous for a single JSON payload or a batch CSV of a few
# thousand rows) to reduce exposure to oversized-payload abuse. This is a minimal hardening step,
# not a substitute for a real API gateway / auth layer in production.
superkart_api.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

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
# Overridable via an environment variable so it can be tuned per-deployment without a code
# change/rebuild, while still defaulting to the exact value used at training time (170).
PREMIUM_MRP_THRESHOLD = float(os.environ.get("PREMIUM_MRP_THRESHOLD", 170))

# The categorical values the model actually saw during training. Used purely for input
# *validation/warnings* below - the model itself still works on unseen categories via the
# preprocessing pipeline's OneHotEncoder(handle_unknown="ignore"), which zero-encodes them.
# A category outside this set (e.g. a new Store_Type the model has never seen) is still handled
# gracefully by the encoder, but is also surfaced as a "warnings" field in the response so the
# caller knows their input included something the model wasn't trained on.
KNOWN_CATEGORIES = {
    "Product_Sugar_Content": {"Low Sugar", "Regular", "No Sugar"},
    "Store_Size": {"Small", "Medium", "High"},
    "Store_Location_City_Type": {"Tier 1", "Tier 2", "Tier 3"},
    "Store_Type": {"Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"},
    "Product_Id_char": {"FD", "DR", "NC"},
    "Product_Type_Category": {"Perishables", "Non Perishables"},
}

# Semantic/range checks for the numeric fields, informed by the ranges observed during training
# (see the EDA / Outlier Detection sections of the training notebook). These catch clearly
# invalid input (e.g. negative weight) that "field is present" validation alone would miss.
NUMERIC_RANGES = {
    "Product_Weight": (0, None),
    "Product_Allocated_Area": (0, 1),
    "Product_MRP": (0, None),
    "Store_Age_Years": (0, None),
}


def _add_engineered_features(df):
    df = df.copy()
    if "Price_per_Weight" not in df.columns:
        df["Price_per_Weight"] = df["Product_MRP"] / df["Product_Weight"]
    if "Premium_Product_Flag" not in df.columns:
        df["Premium_Product_Flag"] = (df["Product_MRP"] > PREMIUM_MRP_THRESHOLD).astype(int)
    return df


def _validate_semantic(record):
    """Row-level (dict) semantic validation. Returns (errors, warnings) lists of strings."""
    errors, warnings = [], []

    for field, (lo, hi) in NUMERIC_RANGES.items():
        value = record.get(field)
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be numeric, got {record.get(field)!r}.")
            continue
        if lo is not None and value < lo:
            errors.append(f"'{field}' must be >= {lo}, got {value}.")
        if hi is not None and value > hi:
            errors.append(f"'{field}' must be <= {hi}, got {value}.")

    for field, known_values in KNOWN_CATEGORIES.items():
        value = record.get(field)
        if value is not None and value not in known_values:
            warnings.append(
                f"'{field}' value {value!r} was not seen during training (known values: "
                f"{sorted(known_values)}). The model will still return a prediction, but treat "
                f"it with extra caution - this field's contribution is effectively dropped."
            )

    return errors, warnings


def _validate_dataframe(df):
    """Batch (DataFrame) version of the same checks. Returns (errors, warnings) lists of strings."""
    errors, warnings = [], []

    for field, (lo, hi) in NUMERIC_RANGES.items():
        col = pd.to_numeric(df[field], errors="coerce")
        if col.isna().any():
            errors.append(f"'{field}' contains non-numeric value(s).")
            continue
        if lo is not None and (col < lo).any():
            errors.append(f"'{field}' contains value(s) below the minimum allowed ({lo}).")
        if hi is not None and (col > hi).any():
            errors.append(f"'{field}' contains value(s) above the maximum allowed ({hi}).")

    for field, known_values in KNOWN_CATEGORIES.items():
        unseen = sorted(set(df[field].unique()) - known_values)
        if unseen:
            warnings.append(
                f"'{field}' contains value(s) not seen during training: {unseen}. "
                f"Predictions for those rows will still be returned, but with reduced confidence."
            )

    return errors, warnings


# If the model's final step is a bagged ensemble (Random Forest / Bagging), it exposes
# `estimators_` - the spread of predictions across individual trees gives an approximate,
# empirical prediction interval, which we surface in the API response when available.
# Named "Sales_95pct_Empirical_*" rather than "Sales_95pct_CI_*" deliberately: "CI" (confidence
# interval) implies a formally calibrated statistical guarantee, which this is not - it's the
# empirical spread across the trees in the forest, which the training notebook's own "Prediction
# Confidence Intervals" section shows has ~93% actual coverage against a 95% nominal target
# (close, but not exact). The field name makes that distinction explicit in the API contract
# itself, not just in prose documentation someone might not read.
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

    # Semantic/range validation, on top of the "is the field present" check above.
    semantic_errors, semantic_warnings = _validate_semantic(data)
    if semantic_errors:
        return jsonify({"error": "; ".join(semantic_errors)}), 400

    try:
        # Extract relevant features from the input data, in the expected order
        sample = {feature: data[feature] for feature in EXPECTED_FEATURES}
        input_data = pd.DataFrame([sample])
        input_data = _add_engineered_features(input_data)

        # Make a prediction using the trained model pipeline (plus an approximate,
        # empirically-derived interval, if supported by the final estimator)
        prediction, lower, upper = _predict_with_interval(input_data)

        response_payload = {"Sales": prediction}
        if lower is not None:
            response_payload["Sales_95pct_Empirical_Lower"] = lower
            response_payload["Sales_95pct_Empirical_Upper"] = upper
        if semantic_warnings:
            response_payload["warnings"] = semantic_warnings

        # Return the prediction as a JSON response
        return jsonify(response_payload)
    except Exception:
        # Don't leak internal exception details (stack traces, file paths, library internals)
        # back to the caller - log server-side instead, return a generic message.
        # `superkart_api.logger` writes to stderr, which Gunicorn/Docker captures.
        superkart_api.logger.exception("Failed to generate a single prediction")
        return jsonify({"error": "Failed to generate prediction due to an internal error."}), 400


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

        # Same semantic validation as the single-prediction endpoint, applied column-wise across
        # the whole batch.
        semantic_errors, semantic_warnings = _validate_dataframe(input_data[EXPECTED_FEATURES])
        if semantic_errors:
            return jsonify({"error": "; ".join(semantic_errors)}), 400

        # Make predictions for the batch data (deriving the same engineered features used at training time)
        engineered = _add_engineered_features(input_data[EXPECTED_FEATURES])
        predictions = model.predict(engineered).tolist()

        # Create an output dictionary mapping row index (as string) to predicted sales
        result = {str(idx): pred for idx, pred in enumerate(predictions)}
        if semantic_warnings:
            # Batch responses stay a flat {row: prediction} map for backward compatibility, so
            # batch-level warnings are returned as a small sidecar key rather than restructuring
            # every row - callers that don't expect it can simply ignore it.
            result["_warnings"] = semantic_warnings
        return jsonify(result)
    except Exception:
        superkart_api.logger.exception("Failed to generate batch predictions")
        return jsonify({"error": "Failed to generate batch predictions due to an internal error."}), 400


# Allows the app to also be run directly (e.g. `python app.py`) for quick local testing,
# in addition to being served by Gunicorn in the Docker container (see Dockerfile below).
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
