import numpy as np
import pandas as pd
from tools.get_prediction_data import get_gen_forecast, get_weather_forecast, get_gen
import torch
from tools.DARNN import QARNN
device = torch.device('cuda')
day_delta = lambda x: pd.Timedelta(days=x)


print('prediction process just started.')
# data preparation
## load basic data
today = pd.Timestamp.now()
x1 = get_weather_forecast(today-day_delta(3), today+day_delta(1)).iloc[:, 1:].values
temp = get_gen_forecast(today-day_delta(3), today+day_delta(1)).iloc[:, 1:].values
x1 = np.concatenate([x1, temp], axis=1)  # 120 x 18
del temp

x2 = get_gen(today-day_delta(3), today-day_delta(1)).iloc[:, -1].values
x2 = x2.reshape(-1, 1)
print('data load complete.')

## data preprocessing
x1 = torch.FloatTensor(x1).view(1, *x1.shape).to(device)
x2 = torch.FloatTensor(x2).view(1, *x2.shape).to(device)
print('data preparation complete.')


# final prediction
y_pred = np.zeros((24, ))
model = QARNN(72, 48, 18, 32, 32).to(device)
model.load_state_dict(torch.load('checkpoints/qarnn_best.pt'))
print('model load complete.')

model.eval()
with torch.no_grad():
    output = torch.relu(model(x1, x2).squeeze(dim=0))
    output = output[31:43].cpu().numpy()
    y_pred[-17:-5] = output
print('prediction process finally complete. Result:\n')
print(y_pred)