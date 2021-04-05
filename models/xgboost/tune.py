import json
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import models.xgboost.objective

reload(models.xgboost.objective)
from models.xgboost.objective import objective
import numpy as np
import optuna
import pandas as pd
import phik
import seaborn as sns
from category_encoders import OrdinalEncoder
from helpers import encode_dates, loguniform, similarity_encode

from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

import xgboost as xgb

df = pd.read_csv(
    r"data\train.csv", parse_dates=[], index_col=[], delimiter=",", low_memory=False,
)

PROFILE = False
if PROFILE:
    profile = ProfileReport(df)
    profile.to_file("pandas_profiling_report.html")

TARGET = "target"
print(f"Missing targets: {df[TARGET].isnull().sum()}")
print(f"% missing: {df[TARGET].isnull().sum() / len(df):.0%}")

DROP_MISSING = False
if DROP_MISSING:
    df = df.dropna(subset=[TARGET])

DROP_FEATURES = []
y = df[TARGET].replace(np.nan, 0)
X = df.drop([TARGET, *DROP_FEATURES], axis=1,)

obj_cols = X.select_dtypes("object").columns
nunique = X[obj_cols].nunique()
prop_unique = (X[obj_cols].nunique() / len(df)).sort_values(
    ascending=False
)  # in order of most unique to least
unique = pd.concat([prop_unique, nunique], axis=1)
unique.columns = [
    "proportion",
    "nunique",
]
print(unique)

ENCODE = False
if ENCODE:
    X = similarity_encode(
        X, encode_columns=[], n_prototypes=4, preran=False, drop_original=True,
    )

LENGTH_ENCODE = False
if LENGTH_ENCODE:
    len_encode = ["URL"]
    for col in len_encode:
        X[f"{col}_len"] = X[col].apply(len)
        X = X.drop(col, axis=1)

CATEGORIZE = True
if CATEGORIZE:
    X[obj_cols] = X[obj_cols].astype("category")
    enc = OrdinalEncoder()
    X = enc.fit_transform(X)

DATE_ENCODE = False
if DATE_ENCODE:
    X = encode_dates(X, "date")

sns.displot(y)
plt.title("Distribution")
plt.show()

SEED = 0
SAMPLE_SIZE = 10000

Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = xgb.DMatrix(Xt, yt)
np.random.seed(SEED)
sample_idx = np.random.choice(Xt.index, size=SAMPLE_SIZE)
Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
ds = xgb.DMatrix(Xs, ys)
dv = xgb.DMatrix(Xv, yv)

OBJECTIVE = "binary:logistic"
METRIC = "auc"
MAXIMIZE = True
EARLY_STOPPING_ROUNDS = 50
MAX_ROUNDS = 20000
REPORT_ROUNDS = 100

params = {
    "objective": OBJECTIVE,
    "eval_metric": METRIC,
    "verbosity": 0,
    "nthread": 6,
    "tree_method": "gpu_hist",
    # "num_classes": 3,
    # "tweedie_variance_power": 1.3,
}

model = xgb.train(
    params,
    dt,
    evals=[(dt, "training"), (dv, "valid")],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

xgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()

TUNE_ETA = False
best_etas = {"learning_rate": [], "score": []}
if TUNE_ETA:
    for _ in range(30):
        eta = loguniform(-3, 1)
        best_etas["learning_rate"].append(eta)
        params["learning_rate"] = eta
        model = xgb.train(
            params,
            dt,
            evals=[(dt, "training"), (dv, "valid")],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_etas["score"].append(model.best_score)

    best_eta_df = pd.DataFrame.from_dict(best_etas)
    lowess_data = lowess(best_eta_df["score"], best_eta_df["learning_rate"],)

    rounded_data = lowess_data.copy()
    rounded_data[:, 1] = rounded_data[:, 1].round(4)
    rounded_data = rounded_data[::-1]  # reverse to find first best
    # maximize or minimize metric
    if MAXIMIZE:
        best = np.argmax
    else:
        best = np.argmin
    best_eta = rounded_data[best(rounded_data[:, 1]), 0]

    # plot relationship between learning rate and performance, with an eta selected just before diminishing returns
    # use log scale as it's easier to observe the whole graph
    sns.lineplot(x=lowess_data[:, 0], y=lowess_data[:, 1])
    plt.xscale("log")
    plt.axvline(best_eta, color="orange")
    plt.title("Smoothed relationship between learning rate and metric.")
    plt.xlabel("learning rate")
    plt.ylabel(METRIC)
    plt.show()

    print(f"Good learning rate: {best_eta}")
    params["learning_rate"] = best_eta
else:
    # best learning rate once run
    params["learning_rate"] = 0.03407666624199937

model = xgb.train(
    params,
    dt,
    evals=[(dt, "training"), (dv, "valid")],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

DROP_CORRELATED = False
if DROP_CORRELATED:
    threshold = 0.75
    corr = Xt.phik_matrix()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    upper = upper.stack()
    high_upper = upper[(abs(upper) > threshold)]
    abs_high_upper = abs(high_upper).sort_values(ascending=False)
    pairs = abs_high_upper.index.to_list()
    correlation = len(pairs) > 0
    print(f"Correlated features: {pairs if correlation else None}")

    correlated_features = set()
    if correlation:
        # drop correlated features
        best_score = model.best_score
        print(f"starting score: {best_score:.4f}")
        drop_dict = {pair: [] for pair in pairs}
        for pair in pairs:
            for feature in pair:
                drop_set = correlated_features.copy()
                drop_set.add(feature)
                Xt, Xv, yt, yv = train_test_split(
                    X.drop(drop_set, axis=1), y, random_state=SEED
                )
                Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
                dt = xgb.DMatrix(Xt, yt, silent=True,)
                dv = xgb.DMatrix(Xv, yv, silent=True,)
                drop_model = xgb.train(
                    params,
                    dt,
                    evals=[(dt, "training"), (dv, "valid")],
                    num_boost_round=MAX_ROUNDS,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                drop_dict[pair].append(drop_model.best_score)
            if MAXIMIZE:
                pair_best = np.max(drop_dict[pair])
                if pair_best > best_score:
                    drop_feature = pair[np.argmax(drop_dict[pair])]
                    best_score = pair_best
                    correlated_features.add(drop_feature)
            else:
                pair_best = np.min(drop_dict[pair])
                if pair_best < best_score:
                    drop_feature = pair[np.argmin(drop_dict[pair])]
                    best_score = pair_best
                    correlated_features.add(drop_feature)
        print(
            f"dropped features: {correlated_features if len(correlated_features) > 0 else None}"
        )
        print(f"ending score: {best_score:.4f}")
else:
    correlated_features = {}

correlation_elimination = len(correlated_features) > 0
if correlation_elimination:
    X = X.drop(correlated_features, axis=1)
    Xt, Xv, yt, yv = train_test_split(
        X, y, random_state=SEED
    )  # split into train and validation set
    Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
    dt = xgb.DMatrix(Xt, yt, silent=True)
    ds = xgb.DMatrix(Xs, ys, silent=True)
    dv = xgb.DMatrix(Xv, yv, silent=True)

    model = xgb.train(
        params,
        dt,
        evals=[(dt, "training"), (dv, "valid")],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

# decide which unimportant features to drop to improve the model
DROP_UNIMPORTANT = False
if DROP_UNIMPORTANT:
    sorted_features = sorted(model.get_score().items(), key=lambda kv: kv[1])
    sorted_features = [x[0] for x in sorted_features]
    best_score = model.best_score
    print(f"starting score: {best_score:.4f}")
    unimportant_features = []
    for feature in sorted_features:
        unimportant_features.append(feature)
        Xt, Xv, yt, yv = train_test_split(
            X.drop(unimportant_features, axis=1), y, random_state=SEED
        )
        Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
        dt = xgb.DMatrix(Xt, yt, silent=True)
        dv = xgb.DMatrix(Xv, yv, silent=True)

        drop_model = xgb.train(
            params,
            dt,
            evals=[(dt, "training"), (dv, "valid")],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        score = drop_model.best_score
        if MAXIMIZE:
            if score < best_score:
                del unimportant_features[-1]  # remove from drop list
                print(f"Dropping {feature} worsened score to {score:.4f}.")
                break
            else:
                best_score = score
        else:
            if score > best_score:
                del unimportant_features[-1]  # remove from drop list
                print(f"Dropping {feature} worsened score to {score:.4f}.")
                break
            else:
                best_score = score
    print(f"ending score: {best_score:.4f}")
    print(
        f"dropped features: {unimportant_features if len(unimportant_features) > 0 else None}"
    )

    # redefine
    Xt, Xv, yt, yv = train_test_split(
        X.drop(unimportant_features, axis=1), y, random_state=SEED
    )
    Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
    dt = xgb.DMatrix(Xt, yt, silent=True)
    dv = xgb.DMatrix(Xv, yv, silent=True)
else:
    unimportant_features = []

feature_elimination = len(unimportant_features) > 0

if feature_elimination:
    X = X.drop(unimportant_features, axis=1)
    Xt, Xv, yt, yv = train_test_split(
        X, y, random_state=SEED
    )  # split into train and validation set
    Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
    dt = xgb.DMatrix(Xt, yt, silent=True)
    ds = xgb.DMatrix(Xs, ys, silent=True)
    dv = xgb.DMatrix(Xv, yv, silent=True)

    model = xgb.train(
        params,
        dt,
        evals=[(dt, "training"), (dv, "valid")],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

DROP_FEATURES.extend(correlated_features)
DROP_FEATURES.extend(unimportant_features)
print(DROP_FEATURES)

xgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()
import os

TUNE_HYPER = False
if TUNE_HYPER:
    study = optuna.create_study(
        storage="sqlite:///xgboost.db",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        direction="maximize",
        study_name="xgboost",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, dt, dv, params, MAX_ROUNDS, EARLY_STOPPING_ROUNDS
        ),
        n_trials=100,
    )
    study.best_trial.params
    dt = xgb.DMatrix(Xt, yt, silent=True)
    ds = xgb.DMatrix(Xs, ys, silent=True)
    dv = xgb.DMatrix(Xv, yv, silent=True)

    score = study.best_value
    best_params = study.best_params
    best_params.update(params)
    # best_params["num_boost_rounds"] = model.best_iteration
    print("Best params:", best_params)
    print(f"  {METRIC} = {score}")
    print("Params:")
    print(json.dumps(best_params, indent=4))
    print(DROP_FEATURES)
else:
    dt = xgb.DMatrix(Xt, yt, silent=True)
    ds = xgb.DMatrix(Xs, ys, silent=True)
    dv = xgb.DMatrix(Xv, yv, silent=True)
    # 0.894158
    best_params = {
        "alpha": 0.11324980045343246,
        "colsample_bytree": 0.3121715844355898,
        "gamma": 3.751285293316015e-05,
        "grow_policy": "depthwise",
        "lambda": 0.015007461107671404,
        "max_depth": 15,
        "min_child_weight": 41,
        "subsample": 0.7744365539734944,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "nthread": 6,
        "tree_method": "gpu_hist",
        "learning_rate": 0.03407666624199937,
    }

model = xgb.train(
    best_params,
    dt,
    evals=[(dt, "training"), (dv, "valid")],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)
score = model.best_score
print(f"{METRIC}: {score:.4f}")


xgb.plot_importance(
    model, grid=False, max_num_features=20, importance_type="gain",
)
figure_path = Path("figures")
figure_path.mkdir(exist_ok=True)
plt.savefig(figure_path / "xgboost_importance.png")

# test score
TEST = True
if TEST:
    df_test = pd.read_csv("data/test.csv")
    df_test[obj_cols] = df_test[obj_cols].astype("category")
    df_test = enc.transform(df_test)
    X_test = df_test.drop(DROP_FEATURES, axis=1)
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    submission = pd.Series(y_pred, index=df_test["id"])
    submission.name = "target"
    submission.to_csv("submission/xgboost.csv")
