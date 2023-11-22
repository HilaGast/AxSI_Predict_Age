import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

torch.manual_seed(0)


def read_data():
    table = pd.read_pickle(
        r"C:\Users\hilag\PycharmProjects\pythonProject\AxSI_Predict_Age\yeo7_100_ADDcm_and_demographic_data.pkl"
    )
    x = table.CM.values  # CM
    x = np.asarray([np.asarray(xi, dtype=np.float32) for xi in x])
    y = table.GENDER.values  # gender tagging
    y = encode_gender(y)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long).reshape(-1)

    return x, y


def encode_gender(y):
    from sklearn.preprocessing import LabelEncoder

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


def batch_norm_cnn():
    IMAGE_SIZE = 100
    BATCH_SIZE = 16
    TEST_SIZE = 0.2
    composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    dataset = GenderDataset()
    train_dataset, test_dataset = train_test_split(dataset, test_size=TEST_SIZE)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN_batch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 5
    accuracy_list, cost_list = train_model(model, criterion, train_data_loader, test_data_loader, optimizer, n_epochs)


class CNN_batch(nn.Module):
    def __init__(self, out1=50, out2=10, out3=5, number_of_classes=2):
        super(CNN_batch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out1)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out2)
        self.conv3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=out3)
        self.fc1 = nn.Linear(in_features=out3 * 100 * 100, out_features=number_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


def train_model(model, criterion, train_loader, validation_loader, optimizer, n_epochs):
    cost_list = []
    accuracy_list = []
    N_test = len(validation_loader)
    for epoch in range(n_epochs):
        COST = 0
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            x = x.unsqueeze(1)
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += [loss.data]
        cost_list.append(COST)
        correct = 0
        # perform a prediction on the validationdata
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
    return accuracy_list, cost_list


def plot_cost_accuracy(accuracy_list, cost_list):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel("epoch", color=color)
    ax1.set_ylabel("Cost", color=color)
    ax1.tick_params(axis="y", color=color)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("accuracy", color=color)
    ax2.set_xlabel("epoch", color=color)
    ax2.plot(accuracy_list, color=color)
    ax2.tick_params(axis="y", color=color)
    fig.tight_layout()
    plt.show()
