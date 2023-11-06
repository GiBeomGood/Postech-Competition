import pandas as pd
import numpy as np
from tqdm import trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, window_size, forecast_size):
        super(CustomDataset, self).__init__()

        data = torch.FloatTensor(data)
        self.x = []; x_append = self.x.append
        self.y = []; y_append = self.y.append
        self.length = data.shape[0] - window_size - forecast_size + 1

        for i in trange(self.length):
            x_append(data[i:i+window_size, :])
            y_append(data[i+window_size:i+window_size+forecast_size, -1])


    def __len__(self):
        return self.length

    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

def get_loaders(window_size=16, forecast_size=24, val_size=0.15, test_size=0.15):
    train = pd.read_csv('data/weather_actual.csv').drop(columns=['time'])
    temp = pd.read_csv('data/gens.csv').amount
    train = pd.concat([train, temp], axis=1).values
    del temp
    length = train.shape[0]

    train_dataset = CustomDataset(train[:-int(length * (val_size+test_size))], window_size, forecast_size)
    val_dataset = CustomDataset(train[-int(length * (val_size+test_size)): -int(length * test_size)], window_size, forecast_size)
    test_dataset = CustomDataset(train[-int(length * test_size):], window_size, forecast_size)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
