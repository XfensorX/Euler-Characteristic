import torch


def calculate_loss(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)

    return total_loss / len(data_loader.dataset)
