import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, out1=500, out2=100, out3=50, IMAGE_SIZE=100, number_of_classes=2):
        super(CNN, self).__init__()
        self.drop = nn.Dropout2d(p=0.1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=out3 * IMAGE_SIZE * IMAGE_SIZE, out_features=number_of_classes)

    def forward(self, x):
        x = self.drop(self.conv1(x))
        x = torch.relu(x)
        x = self.drop(self.conv2(x))
        x = torch.relu(x)
        x = self.drop(self.conv3(x))
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class FFN(nn.Module):
    def __init__(self, Layers, p=0):
        super(FFN, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
    # Prediction function
    def forward(self, x):
        L = len(self.hidden)
        for (l,linear_transform) in zip(range(L),self.hidden):
            if l < L - 1:
                x = self.drop(torch.relu(linear_transform(x)))
            else:
                x = torch.sigmoid(linear_transform(x))
        return x


# class GCN(nn.Module):
#     def __init__(self):