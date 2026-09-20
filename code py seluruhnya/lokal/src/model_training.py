import optuna
import numpy as np
import random

from optuna.samplers import TPESampler

random.seed(42)
np.random.seed(42)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier
)

from xgboost import XGBClassifier

class ModelTrainer:

    def __init__(self):

        self.RANDOM_STATE = 42

        self.best_lr = None
        self.best_rf = None
        self.best_xgb = None

        self.best_model = None
        self.best_model_name = None

        self.study_lr = None
        self.study_rf = None
        self.study_xgb = None

    # LOGISTIC REGRESSION OBJECTIVE
    def objective_lr(self, trial):

        model = LogisticRegression(
            C=trial.suggest_float(
                "C",
                0.001,
                10,
                log=True
            ),
            solver='lbfgs',
            max_iter=3000,
            class_weight=trial.suggest_categorical(
                "class_weight",
                [None, 'balanced']
            ),
            random_state=self.RANDOM_STATE
        )

        score = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=self.skf,
            scoring='f1_macro',
            n_jobs=-1
        ).mean()

        return score

    # RANDOM FOREST OBJECTIVE
    def objective_rf(self, trial):

        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                400
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                5,
                30
            ),
            min_samples_split=trial.suggest_int(
                "min_samples_split",
                2,
                10
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf",
                1,
                5
            ),
            max_features=trial.suggest_categorical(
                "max_features",
                ['sqrt', 'log2']
            ),
            class_weight=trial.suggest_categorical(
                "class_weight",
                [None, 'balanced']
            ),
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        score = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=self.skf,
            scoring='f1_macro',
            n_jobs=-1
        ).mean()

        return score

    # XGBOOST OBJECTIVE
    def objective_xgb(self, trial):

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
            eval_metric='mlogloss',
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        f1_scores = []

        for train_idx, val_idx in self.skf.split(
            self.X_train,
            self.y_train
        ):

            X_train_fold = self.X_train[train_idx]
            X_val_fold = self.X_train[val_idx]

            y_train_fold = self.y_train.iloc[train_idx]
            y_val_fold = self.y_train.iloc[val_idx]

            sample_weight_fold = (
                self.sample_weights[train_idx]
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
                average='macro'
            )

            f1_scores.append(score)

        return np.mean(f1_scores)

    # TRAIN
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

        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weight_dict = dict(
            zip(classes, weights)
        )

        self.sample_weights = np.array([
            class_weight_dict[y]
            for y in y_train
        ])

        print(
            "\n=== CLASS WEIGHT + OPTUNA START ==="
        )

        self.study_lr = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )

        self.study_lr.optimize(
            self.objective_lr,
            n_trials=20
        )

        self.study_rf = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)

        )

        self.study_rf.optimize(
            self.objective_rf,
            n_trials=20
        )

        self.study_xgb = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )

        self.study_xgb.optimize(
            self.objective_xgb,
            n_trials=20
        )

        # BUILD MODELS

        self.best_lr = LogisticRegression(
            **self.study_lr.best_params,
            solver='lbfgs',
            max_iter=3000,
            random_state=self.RANDOM_STATE
        )

        self.best_rf = RandomForestClassifier(
            **self.study_rf.best_params,
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        self.best_xgb = XGBClassifier(
            **self.study_xgb.best_params,
            eval_metric='mlogloss',
            random_state=self.RANDOM_STATE
        )


        # FIT MODELS

        self.best_lr.fit(
            self.X_train,
            self.y_train
        )

        self.best_rf.fit(
            self.X_train,
            self.y_train
        )

        self.best_xgb.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weights
        )

        scores = {
            "Logistic Regression":
                self.study_lr.best_value,

            "Random Forest":
                self.study_rf.best_value,

            "XGBoost":
                self.study_xgb.best_value
        }

        self.best_model_name = max(
            scores,
            key=scores.get
        )

        self.best_score = scores[
            self.best_model_name
        ]

        if self.best_model_name == \
                "Logistic Regression":

            self.best_model = self.best_lr

        elif self.best_model_name == \
                "Random Forest":

            self.best_model = self.best_rf

        else:

            self.best_model = self.best_xgb

        print("\n=== BEST RESULTS ===")

        for name, score in scores.items():

            print(
                f"{name}: "
                f"{score:.4f}"
            )

        print(
            f"\nBest Model: "
            f"{self.best_model_name}"
        )

        print(
            f"Best CV F1: "
            f"{self.best_score:.4f}"
        )

        if self.best_model_name == "Logistic Regression":
            self.best_params = self.study_lr.best_params
        elif self.best_model_name == "Random Forest":
            self.best_params = self.study_rf.best_params
        else:
            self.best_params = self.study_xgb.best_params

        return (
            self.best_model,
            self.best_model_name
        )