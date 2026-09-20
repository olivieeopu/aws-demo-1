from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

import os
import joblib

import mlflow
import mlflow.sklearn

# Data
ingestion = DataIngestion(
    "data/data_B.csv"
)

df = ingestion.load_data()

# Preprocess
preprocessor = DataPreprocessor()

X_train, X_test, y_train, y_test = (
    preprocessor.preprocess(df)
)

# Train
trainer = ModelTrainer()

best_model, best_model_name = (
    trainer.train(
        X_train,
        y_train
    )
)

# eval

evaluator = ModelEvaluator()

# MLFLOW LOGGING
mlflow.set_experiment(
    "Credit Score Prediction"
)

results = evaluator.evaluate(
    best_model,
    X_test,
    y_test
)

# LOG ALL EXPERIMENTS

experiments = [
    (
        "Logistic Regression",
        trainer.best_lr,
        trainer.study_lr
    ),
    (
        "Random Forest",
        trainer.best_rf,
        trainer.study_rf
    ),
    (
        "XGBoost",
        trainer.best_xgb,
        trainer.study_xgb
    )
]

for model_name, model, study in experiments:

    exp_results = evaluator.evaluate(
        model,
        X_test,
        y_test
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "accuracy",
            exp_results["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            exp_results["precision"]
        )

        mlflow.log_metric(
            "recall",
            exp_results["recall"]
        )

        mlflow.log_metric(
            "f1_score",
            exp_results["f1"]
        )

        mlflow.log_metric(
            "roc_auc",
            exp_results["roc_auc"]
        )

        mlflow.log_metric(
            "best_cv_f1",
            study.best_value
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

with mlflow.start_run(
    run_name=f"Best_{best_model_name}"
):

    # Metrics
    mlflow.log_metric(
        "accuracy",
        results["accuracy"]
    )

    mlflow.log_metric(
        "precision",
        results["precision"]
    )

    mlflow.log_metric(
        "recall",
        results["recall"]
    )

    mlflow.log_metric(
        "f1_score",
        results["f1"]
    )

    mlflow.log_metric(
        "roc_auc",
        results["roc_auc"]
    )

    mlflow.log_metric(
        "best_cv_f1",
        trainer.best_score
    )

    # Parameters
    mlflow.log_param(
        "best_model_name",
        trainer.best_model_name
    )

    mlflow.log_params(
        trainer.best_params
    )

    # Model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model"
    )



# save model
os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    preprocessor.scaler,
    "models/scaler.pkl"
)

joblib.dump(
    preprocessor.encoder,
    "models/encoder.pkl"
)

joblib.dump(
    preprocessor.feature_names,
    "models/feature_names.pkl"
)

# final out
print(
    f"\nFinal Best Model: "
    f"{best_model_name}"
)

print(
    "\nFiles saved:"
)

print(
    "- models/best_model.pkl"
)

print(
    "- models/scaler.pkl"
)

print(
    "- models/encoder.pkl"
)

print(
    "- models/feature_names.pkl"
)