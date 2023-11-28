import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder



def read_data():
    table = pd.read_pickle(
        r".\yeo7_100_Numcm_and_demographic_data.pkl"
    )
    x = table.CM.values  # CM
    scaler = StandardScaler()
    x_list = []
    for xi in x:
        xi = np.asarray(xi.reshape(-1,1), dtype=np.float32)
        xi[xi<=0] = np.nan
        xi = scaler.fit_transform(xi)
        xi[np.isnan(xi)] = 0
        x_list.append(xi)
    x = np.asarray(x_list)

    y = table.GENDER.values  # gender tagging
    y = encode_gender(y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long).reshape(-1)

    return x, y


def encode_gender(y):

    le = LabelEncoder()
    y = le.fit_transform(y)
    return y


class GenderDataset(Dataset):
    def __init__(self):
        self.x, self.y = read_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



