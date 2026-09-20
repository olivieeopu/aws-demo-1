from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import pandas as pd


class ModelEvaluator:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

        print("\n===== EVALUATION =====")

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            "\n===== CLASSIFICATION REPORT ====="
        )

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Poor",
                    "Standard",
                    "Good"
                ]
            )
        )

        print(
            "\n===== CONFUSION MATRIX ====="
        )

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
    