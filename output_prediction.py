import numpy as np
import pandas as pd
import xgboost
import pickle
from tools.get_prediction_data import get_gen_forecast, get_weather_forecast


today = pd.Timestamp.now().strftime('%Y-%m-%d')
test = get_weather_forecast().iloc[:, 1:].values
gens_prediction = get_gen_forecast().iloc[:, 1:].values


# # time setup
# today = pd.Timestamp.now().strftime('%Y-%m-%d')  # ex. 2023-10-27
# time_start = pd.to_datetime(f'{today} 01:00:00').tz_localize('Asia/Seoul') + pd.Timedelta(days=1)
# time_end = time_start + pd.Timedelta(hours=23)
# day_slicer = lambda data: data.loc[(data['time'] >= time_start) & (data['time'] <= time_end), :].reset_index(drop=True)
# print('time setup complete.')


# # data preparation
# ## load basic data
# test = pd.read_csv('data_prediction/weathers_forecasts_17.csv').sort_values(by=['time'])  # 10. 26. 16:00~10. 27. 15:00 <- 10. 23. 16:00~10. 26. 15:00
# test['time'] = pd.to_datetime(test['time']).dt.tz_convert('Asia/Seoul')
# test = day_slicer(test).iloc[:, 1:].values

# gens_prediction = pd.read_csv('data_prediction/gen_forecasts_17.csv').sort_values(by='time')  # 10. 26. 16:00~10. 27. 15:00
# gens_prediction['time'] = pd.to_datetime(gens_prediction['time']).dt.tz_convert('Asia/Seoul')
# gens_prediction = day_slicer(gens_prediction).iloc[:, 1:].values

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

y_test = predictor.predict(dtest)
print('prediction process finally complete. Result:\n')
print(y_test)