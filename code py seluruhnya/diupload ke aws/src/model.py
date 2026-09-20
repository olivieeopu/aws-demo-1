import optuna
import numpy as np

from optuna.samplers import TPESampler

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from sklearn.metrics import (
    f1_score
)

from xgboost import XGBClassifier


class ModelTrainer:

    def __init__(self):

        self.RANDOM_STATE = 42

        self.best_model = None
        self.best_score = None
        self.best_params = None

        self.study = None

    def objective(
        self,
        trial
    ):

        model = XGBClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                400
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                3,
                10
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.01,
                0.3
            ),
            subsample=trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),
            gamma=trial.suggest_float(
                "gamma",
                0,
                5
            ),
            eval_metric="mlogloss",
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        scores = []

        for train_idx, val_idx in self.skf.split(
            self.X_train,
            self.y_train
        ):

            X_train_fold = self.X_train[
                train_idx
            ]

            X_val_fold = self.X_train[
                val_idx
            ]

            y_train_fold = self.y_train.iloc[
                train_idx
            ]

            y_val_fold = self.y_train.iloc[
                val_idx
            ]

            sample_weight_fold = (
                self.sample_weights[
                    train_idx
                ]
            )

            model.fit(
                X_train_fold,
                y_train_fold,
                sample_weight=sample_weight_fold
            )

            y_pred = model.predict(
                X_val_fold
            )

            score = f1_score(
                y_val_fold,
                y_pred,
                average="macro"
            )

            scores.append(score)

        return np.mean(scores)

    def train(
        self,
        X_train,
        y_train
    ):

        self.X_train = X_train
        self.y_train = y_train

        self.skf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.RANDOM_STATE
        )

        classes = np.unique(
            y_train
        )

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )

        class_weight_dict = dict(
            zip(
                classes,
                weights
            )
        )

        self.sample_weights = np.array([
            class_weight_dict[y]
            for y in y_train
        ])

        print(
            "\n=== OPTUNA TUNING START ==="
        )

        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(
                seed=42
            )
        )

        self.study.optimize(
            self.objective,
            n_trials=20
        )

        self.best_score = (
            self.study.best_value
        )

        self.best_params = (
            self.study.best_params
        )

        self.best_model = XGBClassifier(
            **self.best_params,
            eval_metric="mlogloss",
            random_state=self.RANDOM_STATE
        )

        self.best_model.fit(
            X_train,
            y_train,
            sample_weight=self.sample_weights
        )

        print(
            f"\nBest CV F1: "
            f"{self.best_score:.4f}"
        )

        print(
            "\nBest Parameters:"
        )

        print(
            self.best_params
        )

        return self.best_model