import pandas as pd
import pickle
from tools.get_prediction_data import get_bid_results

import torch
from tools.DLinear import Model as Dlinear
device = torch.device('cuda')


# data preparation
today = pd.Timestamp.now().strftime('%Y-%m-%d')
errors_x = get_bid_results().iloc[:, 1:].values
errors_x = torch.FloatTensor(errors_x).reshape(1, -1, 5).to(device)


# errors prediction
class Config:
    def __init__(self, seq_len=10, pred_len=2, individual=True, enc_in=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in  # channels
        return

configs = Config(seq_len=72, pred_len=48, individual=True, enc_in=5)
dlinear = Dlinear(configs).to(device)
dlinear.load_state_dict(torch.load('checkpoints/errors_to_errors_1.pt'))
print('dliear model load complete.')

with torch.no_grad():
    errors_prediction = dlinear(errors_x).squeeze(dim=0)
    errors_prediction = errors_prediction.cpu().numpy()


# saving
with open(f'data_prediction/{today}-errors_prediction.pickle', 'wb') as f:
    pickle.dump(errors_prediction, f)
print('errors prediction and saving process complete.')