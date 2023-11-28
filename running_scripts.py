import torch
from sklearn.model_selection import train_test_split
from plotting import plot_cost_accuracy
from load_data import GenderDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader


def batch_norm_ffn():
    from training import train_model_FFN
    from models import FFN

    IMAGE_SIZE = 100
    BATCH_SIZE = 64
    TEST_SIZE = 0.3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    dataset = GenderDataset()
    dataset.x = dataset.x.to(DEVICE)
    dataset.y = dataset.y.to(DEVICE)
    train_dataset, test_dataset = train_test_split(dataset, test_size=TEST_SIZE)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    Layers = [IMAGE_SIZE * IMAGE_SIZE, 5000, 1000,500, 100, 2]
    model = FFN(Layers, p=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = 50
    accuracy_list, cost_list = train_model_FFN(model, criterion, train_data_loader, test_data_loader, optimizer, n_epochs)
    plot_cost_accuracy(accuracy_list, cost_list)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_norm_ffn()