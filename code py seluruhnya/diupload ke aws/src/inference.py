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
        os.path.join(
            model_dir,
            "model.joblib"
        )
    )

    scaler = joblib.load(
        os.path.join(
            model_dir,
            "scaler.joblib"
        )
    )

    feature_names = joblib.load(
        os.path.join(
            model_dir,
            "feature_names.joblib"
        )
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }


def input_fn(
    request_body,
    request_content_type
):

    if request_content_type != JSON_CONTENT_TYPE:

        raise ValueError(
            f"Unsupported content type: "
            f"{request_content_type}"
        )

    payload = json.loads(
        request_body
    )

    return pd.DataFrame(
        payload["instances"]
    )


def predict_fn(
    input_data,
    artifacts
):

    model = artifacts["model"]

    scaler = artifacts["scaler"]

    feature_names = artifacts[
        "feature_names"
    ]

    input_data = input_data[
        feature_names
    ]

    input_scaled = scaler.transform(
        input_data
    )

    probs = model.predict_proba(
        input_scaled
    )

    predictions = model.predict(
        input_scaled
    )

    labels = [
        CLASS_NAMES[int(i)]
        for i in predictions
    ]

    return {
        "predictions":
            predictions.tolist(),

        "labels":
            labels,

        "probabilities":
            probs.tolist()
    }


def output_fn(
    prediction,
    accept_content_type
):

    if (
        accept_content_type
        ==
        JSON_CONTENT_TYPE
    ):

        return (
            json.dumps(
                prediction
            ),
            JSON_CONTENT_TYPE
        )

    raise ValueError(
        f"Unsupported accept type: "
        f"{accept_content_type}"
    )