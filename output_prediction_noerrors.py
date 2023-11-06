import numpy as np
import pandas as pd
import xgboost
import pickle
from tools.get_prediction_data import get_gen_forecast, get_weather_forecast


# data preparation
## load basic data
test = get_weather_forecast().iloc[:, 1:].values  # (24 x ?)
gens_prediction = get_gen_forecast().iloc[:, 1:].values  # (24 x ?)
test = np.concatenate([test, gens_prediction], axis=1)
dtest = xgboost.DMatrix(test)


# final prediction
with open('checkpoints/weather+pred_prediction.xgb', 'rb') as f:
    predictor = pickle.load(f)
print('xgboost model load complete.')

y_test = predictor.predict(dtest)
print('prediction process finally complete. Result:\n')
print(y_test)