from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

class ModelEvaluator:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                average="macro"
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                average="macro"
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                average="macro"
            ),
            "roc_auc": roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        }