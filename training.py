import torch


def train_model_FFN(model, criterion, train_loader, validation_loader, optimizer, n_epochs):
    cost_list = []
    accuracy_list = []
    N_test = len(validation_loader)
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        COST = 0
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            x = x.squeeze(2)
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.data
        cost_list.append(COST.cpu())
        correct = 0
        # perform a prediction on the validationdata
        for x_test, y_test in validation_loader:
            model.eval()
            x_test = x_test.squeeze(2)
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
    return accuracy_list, cost_list