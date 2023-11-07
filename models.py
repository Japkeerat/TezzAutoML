import xgboost as xgb
import lightgbm as lgbm


def xgboost_model(trial, task):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "objective": "reg:squarederror" if task == "regression" else "binary:logistic",
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 9, 81),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }
    if task == "classification":
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1, **param_grid)
    else:
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **param_grid)
    return model


def lgbm_model(trial, task):
    param_grid = {
        "num_leaves": trial.suggest_int("num_leaves", 9, 81),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
    }
    if task == "classification":
        model = lgbm.LGBMClassifier(random_state=42, n_jobs=-1, **param_grid)
    else:
        model = lgbm.LGBMRegressor(random_state=42, n_jobs=-1, **param_grid)
    return model
