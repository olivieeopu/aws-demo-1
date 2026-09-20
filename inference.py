inference.py

import json
import os

import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"

CLASS_NAMES = [
    "Poor",
    "Standard",
    "Good"
]


def model_fn(model_dir):

    model = joblib.load(
        os.path.join(model_dir, "model.joblib")
    )

    scaler = joblib.load(
        os.path.join(model_dir, "scaler.joblib")
    )

    encoder = joblib.load(
        os.path.join(model_dir, "encoder.joblib")
    )

    feature_names = joblib.load(
        os.path.join(model_dir, "feature_names.joblib")
    )

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": feature_names,
    }


def input_fn(request_body, request_content_type):

    if request_content_type != JSON_CONTENT_TYPE:
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    payload = json.loads(request_body)

    return pd.DataFrame(payload["instances"])


def predict_fn(input_data, artifacts):

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    feature_names = artifacts["feature_names"]

    # ----------------------------
    # Feature Engineering
    # ----------------------------

    input_data["Debt_to_Income_Ratio"] = np.where(
        input_data["Annual_Income"] > 0,
        input_data["Outstanding_Debt"]
        / input_data["Annual_Income"],
        0,
    )

    input_data["Risk_Score"] = (
        0.5 * input_data["Delay_from_due_date"]
        + 0.3 * input_data["Num_of_Delayed_Payment"]
        + 0.2 * input_data["Num_Credit_Inquiries"]
    )

    input_data["Financial_Exposure"] = (
        input_data["Num_Bank_Accounts"]
        + input_data["Num_Credit_Card"]
        + input_data["Num_of_Loan"]
    )

    # ----------------------------
    # One Hot Encoding
    # ----------------------------

    categorical_cols = [
        "Occupation",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
    ]

    encoded = encoder.transform(
        input_data[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_cols
        ),
        index=input_data.index,
    )

    final_df = pd.concat(
        [
            input_data.drop(columns=categorical_cols),
            encoded_df,
        ],
        axis=1,
    )

    # hanya feature yang dipakai model
    final_df = final_df.reindex(
        columns=feature_names,
        fill_value=0,
    )

    # ----------------------------
    # Scaling
    # ----------------------------

    scaled = scaler.transform(final_df)

    predictions = model.predict(scaled)

    probabilities = model.predict_proba(scaled)

    labels = [
        CLASS_NAMES[int(x)]
        for x in predictions
    ]

    return {
        "predictions": predictions.tolist(),
        "labels": labels,
        "probabilities": probabilities.tolist(),
    }


def output_fn(prediction, accept_content_type):

    if accept_content_type == JSON_CONTENT_TYPE:

        return (
            json.dumps(prediction),
            JSON_CONTENT_TYPE,
        )

    raise ValueError(
        f"Unsupported accept type: {accept_content_type}"
    )