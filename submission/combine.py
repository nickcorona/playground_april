import pandas as pd


lightgbm = pd.read_csv("submission/lightgbm.csv", index_col="id")
xgboost = pd.read_csv("submission/xgboost.csv", index_col="id")

ensemble = (lightgbm + xgboost) / 2

ensemble.to_csv("submission/ensemble.csv")