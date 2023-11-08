import numpy as np
import pandas as pd
import xgboost
import pickle
from tools.get_prediction_data import get_gen_forecast, get_weather_forecast


# data preparation
## load basic data
today = pd.Timestamp.now().strftime('%Y-%m-%d')
test = get_weather_forecast().iloc[:, 1:].values
gens_prediction = get_gen_forecast().iloc[:, 1:].values


## load errors prediction
with open(f'data_prediction/{today}-errors_prediction.pickle', 'rb') as f:
    errors_prediction = pickle.load(f)[-24:, :]
print('data preparation complete.')


# final prediction
test = np.concatenate([test, gens_prediction, errors_prediction], axis=1)
dtest = xgboost.DMatrix(test)
with open('checkpoints/weather+pred+error_prediction.xgb', 'rb') as f:
    predictor = pickle.load(f)
print('xgboost model load complete.')

y_pred = predictor.predict(dtest)
print('prediction process finally complete. Result:\n')
print(y_pred)