import torch


def evaluate(model, dataloader, criterion, device):

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = torch.max(output, 1)
            total += batch_size
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / total 
    accuracy = 100 * correct / total
    return avg_loss, accuracy