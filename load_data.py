import torch
import torch.nn as nn


def read_data():
    x = []  # CM
    y = []  # gender tagging
    y = encode_gender(y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    return x, y


def encode_gender(y):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(y)
    return y
