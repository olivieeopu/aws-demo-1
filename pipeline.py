import sys
import os
import tarfile
import joblib

sys.path.append("src")

from src.data import (
    DataIngestion,
    DataPreprocessor
)

from src.model import (
    ModelTrainer
)

from src.evaluate import (
    ModelEvaluator
)


ARTIFACT_DIR = "model_artifact"
MODEL_FILENAME = "model.joblib"
TARBALL_PATH = os.path.join(
    ARTIFACT_DIR,
    "model.tar.gz"
)


def main():

    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    print("\n=== LOAD DATA ===")

    ingestion = DataIngestion(
        "../data/data_B.csv"
    )

    df = ingestion.load_data()

    print("\n=== PREPROCESSING ===")

    preprocessor = DataPreprocessor()

    X_train, X_test, y_train, y_test = (
        preprocessor.preprocess(df)
    )

    print("\n=== TRAINING ===")

    trainer = ModelTrainer()

    model = trainer.train(
        X_train,
        y_train
    )

    print("\n=== EVALUATION ===")

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(
        model,
        X_test,
        y_test
    )

    print("\n=== FINAL RESULTS ===")

    print(
        f"Best CV F1: "
        f"{trainer.best_score:.4f}"
    )

    print(
        "\nBest Parameters:"
    )

    print(
        trainer.best_params
    )

    print(
        "\nTest Metrics:"
    )

    print(
        results
    )

    print("\n=== SAVE ARTIFACTS ===")

    # MODEL
    joblib.dump(
        model,
        os.path.join(
            ARTIFACT_DIR,
            MODEL_FILENAME
        )
    )

    # PREPROCESSOR OBJECTS
    joblib.dump(
        preprocessor.encoder,
        os.path.join(
            ARTIFACT_DIR,
            "encoder.joblib"
        )
    )

    joblib.dump(
        preprocessor.scaler,
        os.path.join(
            ARTIFACT_DIR,
            "scaler.joblib"
        )
    )

    joblib.dump(
        preprocessor.feature_names,
        os.path.join(
            ARTIFACT_DIR,
            "feature_names.joblib"
        )
    )

    print(
        f"Saved model to "
        f"{ARTIFACT_DIR}"
    )

    print(
        "\n=== PACKAGE MODEL ==="
    )

    with tarfile.open(
        TARBALL_PATH,
        "w:gz"
    ) as tar:

        tar.add(
            os.path.join(
                ARTIFACT_DIR,
                MODEL_FILENAME
            ),
            arcname=MODEL_FILENAME
        )

        tar.add(
            os.path.join(
                ARTIFACT_DIR,
                "encoder.joblib"
            ),
            arcname="encoder.joblib"
        )

        tar.add(
            os.path.join(
                ARTIFACT_DIR,
                "scaler.joblib"
            ),
            arcname="scaler.joblib"
        )

        tar.add(
            os.path.join(
                ARTIFACT_DIR,
                "feature_names.joblib"
            ),
            arcname="feature_names.joblib"
        )

        tar.add(
            "src",
            arcname="code"
        )

    print(
        f"Packaged: "
        f"{TARBALL_PATH}"
    )

    print(
        "\nNext steps:"
    )

    print(
        "1. Upload model.tar.gz to S3"
    )

    print(
        "2. Create SageMaker Endpoint"
    )


if __name__ == "__main__":
    main()