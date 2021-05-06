import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik
import seaborn as sns
from helpers import encode_dates, loguniform, similarity_encode
from optuna.integration._lightgbm_tuner.optimize import LightGBMTunerCV
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

import lightgbm as lgb

df = pd.read_csv(
    r"data\train.csv",
    parse_dates=[],
    index_col=["PassengerId"],
    delimiter=",",
    low_memory=False,
)

PROFILE = False
if PROFILE:
    profile = ProfileReport(df)
    profile.to_file("pandas_profiling_report.html")

TARGET = "Survived"
print(f"Missing targets: {df[TARGET].isnull().sum()}")
print(f"% missing: {df[TARGET].isnull().sum() / len(df):.0%}")

DROP_MISSING = False
if DROP_MISSING:
    df = df.dropna(subset=[TARGET])

DROP_FEATURES = []
y = df[TARGET].replace(np.nan, 0)
X = df.drop([TARGET, *DROP_FEATURES], axis=1,)


def preprocess(df):
    df = df.copy()
    df["ticket_number"] = (
        df["Ticket"]
        .str.split(" ")
        .dropna()
        .apply(lambda x: x[-1])
        .replace({"": np.nan})
        .astype(float)
    )
    df["ticket_prefix"] = df["Ticket"].str.extract(r"([A-Za-z.\d\/]+) ")
    df["family_size"] = df["SibSp"] + df["Parch"]
    df["age*pclass"] = df["Age"] * df["Pclass"]
    df["deck"] = df["Cabin"].str.extract(r"([A-Z])")
    df["cabin_number"] = df["Cabin"].str.extract(r"(\d+)").astype(float)
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1, errors="ignore")
    return df


X = preprocess(X)

CATEGORIZE = True
if CATEGORIZE:
    obj_cols = X.select_dtypes("object").columns
    X[obj_cols] = X[obj_cols].astype("category")

DATE_ENCODE = False
if DATE_ENCODE:
    X = encode_dates(X, "date")

PLOT_TARGET = False
if PLOT_TARGET:
    sns.displot(y)
    plt.title("Distribution")
    plt.show()

SEED = 0
SAMPLE_SIZE = 10000

Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, free_raw_data=False)
np.random.seed(SEED)
sample_idx = np.random.choice(Xt.index, size=SAMPLE_SIZE)
Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
ds = lgb.Dataset(Xs, ys)
dv = lgb.Dataset(Xv, yv, free_raw_data=False)

OBJECTIVE = "binary"
METRIC = "binary_error"
MAXIMIZE = False
EARLY_STOPPING_ROUNDS = 200
MAX_ROUNDS = 20000
REPORT_ROUNDS = 100

params = {
    "objective": OBJECTIVE,
    "metric": METRIC,
    "verbose": -1,
    "n_jobs": 6,
    "num_classes": 1,
}

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

lgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()

TUNE_ETA = True
best_etas = {"learning_rate": [], "score": []}
if TUNE_ETA:
    for _ in range(30):
        eta = loguniform(-3, 1)
        best_etas["learning_rate"].append(eta)
        params["learning_rate"] = eta
        model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_etas["score"].append(model.best_score["valid"][METRIC])

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
    params["learning_rate"] = 0.1206840078080975

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

DROP_CORRELATED = True
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
        best_score = model.best_score["valid"][METRIC]
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
                dt = lgb.Dataset(Xt, yt, silent=True,)
                dv = lgb.Dataset(Xv, yv, silent=True,)
                drop_model = lgb.train(
                    params,
                    dt,
                    valid_sets=[dt, dv],
                    valid_names=["training", "valid"],
                    num_boost_round=MAX_ROUNDS,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                drop_dict[pair].append(drop_model.best_score["valid"][METRIC])
            if MAXIMIZE:
                pair_best = np.max(drop_feature[pair])
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
    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)

    model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

# decide which unimportant features to drop to improve the model
DROP_UNIMPORTANT = True
if DROP_UNIMPORTANT:
    sorted_features = [
        feature
        for _, feature in sorted(
            zip(model.feature_importance(importance_type="gain"), dt.feature_name),
            reverse=False,
        )
    ]
    best_score = model.best_score["valid"][METRIC]
    print(f"starting score: {best_score:.4f}")
    unimportant_features = []
    for feature in sorted_features:
        unimportant_features.append(feature)
        Xt, Xv, yt, yv = train_test_split(
            X.drop(unimportant_features, axis=1), y, random_state=SEED
        )
        Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
        dt = lgb.Dataset(Xt, yt, silent=True)
        dv = lgb.Dataset(Xv, yv, silent=True)

        drop_model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        score = drop_model.best_score["valid"][METRIC]
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
    dt = lgb.Dataset(Xt, yt, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)
else:
    unimportant_features = []

feature_elimination = len(unimportant_features) > 0

if feature_elimination:
    X = X.drop(unimportant_features, axis=1)
    Xt, Xv, yt, yv = train_test_split(
        X, y, random_state=SEED
    )  # split into train and validation set
    Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)

    model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

DROP_FEATURES.extend(correlated_features)
DROP_FEATURES.extend(unimportant_features)
print(DROP_FEATURES)

lgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()

TUNE_HYPER = True
if TUNE_HYPER:
    # TODO: switch to cross-validation
    from optuna.integration.lightgbm import LightGBMTuner
    from sklearn.model_selection import KFold

    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)
    auto_booster = LightGBMTuner(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        # return_cvbooster=True,
    )
    auto_booster.run()

    score = auto_booster.best_score
    best_params = auto_booster.best_params
    model = auto_booster.get_best_booster()
    best_params["num_boost_rounds"] = model.best_iteration
    print("Best params:", best_params)
    print(f"  {METRIC} = {score}")
    print("Params:")
    print(json.dumps(best_params, indent=4))
    print(DROP_FEATURES)
else:
    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)
    # best error: 0.21356
    best_params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbose": -1,
        "n_jobs": 6,
        "num_classes": 1,
        "learning_rate": 0.1206840078080975,
        "feature_pre_filter": False,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "num_leaves": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "min_child_samples": 20,
        "num_boost_rounds": 304,
    }


model = lgb.train(
    best_params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)
score = model.best_score["valid"][METRIC]
print(f"{METRIC}: {score:.4f}")


lgb.plot_importance(
    model, grid=False, max_num_features=20, importance_type="gain", figsize=(10, 5)
)
plt.show()
figure_path = Path("figures")
figure_path.mkdir(exist_ok=True)
plt.savefig(figure_path / "feature_importance.png")

# test score
TEST = True
if TEST:
    df_test = pd.read_csv("data/test.csv", index_col=["PassengerId"])
    X_test = df_test.drop(DROP_FEATURES, axis=1)
    X_test = preprocess(X_test)
    X_test[obj_cols] = X_test[obj_cols].astype("category")
    y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
    submission = pd.Series(y_pred, index=df_test.index)
    submission.name = "Survived"
    submission.to_csv("submission/lightgbm.csv")
