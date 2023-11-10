import os
from datetime import datetime

import mlflow
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from tezzautoml.models import xgboost_model, lgbm_model


class AutoML:
    def __init__(
        self,
        train_data: pd.DataFrame,
        target: str,
        log: bool = False,
        n_trials: int = 10,
        task: str = "classification",
        mlflow_experiment_name: str = "automl",
        fast_mode: bool = False,
        tracking_uri: str = "http://127.0.0.1:8080",
    ):
        self.best_model = None
        self.train_data = train_data
        self.target = target
        self.log = log
        self.n_trials = n_trials
        self.task = task
        self.X = self.train_data.drop(self.target, axis=1)
        self.y = self.train_data[self.target]
        self.mlflow_experiment_name = mlflow_experiment_name
        self.fast_mode = fast_mode
        self.tracking_uri = tracking_uri
        self.mlflow_callback = MLflowCallback(
            tracking_uri=self.tracking_uri,
            metric_name="f1_score" if self.task == "classification" else "mse",
            create_experiment=True,
        )
        mlflow.set_tracking_uri(self.tracking_uri)

    def kfold_cv(self, model, kfold):
        for train_index, val_index in kfold.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            if self.task == "classification":
                return f1_score(y_val, y_pred), model
            else:
                return mean_squared_error(y_val, y_pred), model

    def train_test_split(self, model):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if self.task == "classification":
            return f1_score(y_val, y_pred), model
        else:
            return mean_squared_error(y_val, y_pred), model

    def save_model(self, model, algorithm="XGBoost"):
        model_dir = "../model"
        os.makedirs(model_dir, exist_ok=True)
        if algorithm == "XGBoost":
            file_name = os.path.join(model_dir, "model.xgb")
            model.save_model(file_name)
        elif algorithm == "LightGBM":
            file_name = os.path.join(model_dir, "model.lgbm")
            model.booster_.save_model(file_name)
        else:
            file_name = os.path.join(model_dir, "model.pkl")
        return file_name

    def fit(self):
        mlfc = self.mlflow_callback

        @mlfc.track_in_mlflow()
        def _objective(trial: optuna.Trial):
            model = None
            algo_to_use = trial.suggest_categorical("model", ["XGBoost", "LightGBM"])
            if algo_to_use == "XGBoost":
                model = xgboost_model(trial, self.task)
            elif algo_to_use == "LightGBM":
                model = lgbm_model(trial, self.task)
            metric = list()
            if self.fast_mode:
                score, model = self.train_test_split(model)
            else:
                if self.task == "classification":
                    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                else:
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                score, model = self.kfold_cv(model, kfold)
            metric.append(score)
            file_name = self.save_model(model, algo_to_use)
            trial.set_user_attr('model', model)
            mlflow.log_artifact(file_name)
            return sum(metric) / len(metric)

        study = optuna.create_study(
            direction="maximize" if self.task == "classification" else "minimize",
            study_name=self.mlflow_experiment_name
            + "_finetuning_"
            + str(datetime.now()),
        )
        study.optimize(_objective, n_trials=self.n_trials, callbacks=[mlfc])
        self.best_model = study.best_trial.user_attrs['model']

    def predict(self, X):
        """
        Make predictions using the best model.

        :param X: DataFrame or array-like, features for making predictions
        :return: Predicted values
        """
        return self.best_model.predict(X)
