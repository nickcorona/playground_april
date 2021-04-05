import optuna
import xgboost as xgb


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial, dt, dv, params, num_boost_round, early_stopping_rounds):
    # TODO: figure out how to add the following hperparaters conditional on grow_policy: max_leaves
    params = params.copy()
    params["lambda"] = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
    params["alpha"] = trial.suggest_float("alpha", 1e-8, 2.0, log=True)
    params["max_depth"] = trial.suggest_int("max_depth", 1, 15)
    params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    params["grow_policy"] = trial.suggest_categorical(
        "grow_policy", ["depthwise", "lossguide"]
    )
    params["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 150)
    params["subsample"] = trial.suggest_float("subsample", 0.3, 1.0, log=True)
    params["colsample_bytree"] = trial.suggest_float(
        "colsample_bytree", 0.3, 1.0, log=True
    )

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, f"valid-{params['eval_metric']}"
    )
    bst = xgb.train(
        params,
        dt,
        evals=[(dt, "training"), (dv, "valid")],
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[pruning_callback],
        verbose_eval=False,
    )
    return bst.best_score


{
    "lambda": 0.32504607983010503,
    "alpha": 0.0350944365077051,
    "max_depth": 11,
    "gamma": 1.9524613583443957e-06,
    "grow_policy": "depthwise",
    "min_child_weight": 47,
    "subsample": 0.9118731804951301,
}

