import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELossForClassification(nn.Module):
    def __init__(self, num_classes):
        super(MSELossForClassification, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        predictions_soft = F.softmax(predictions, dim=1)
        return F.mse_loss(predictions_soft, targets_one_hot)


class MAELossForClassification(nn.Module):
    def __init__(self, num_classes):
        super(MAELossForClassification, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        predictions_soft = F.softmax(predictions, dim=1)
        return F.l1_loss(predictions_soft, targets_one_hot)


class HuberLossForClassification(nn.Module):
    def __init__(self, num_classes, delta=1.0):
        super(HuberLossForClassification, self).__init__()
        self.num_classes = num_classes
        self.delta = delta
    
    def forward(self, predictions, targets):
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        predictions_soft = F.softmax(predictions, dim=1)
        return F.smooth_l1_loss(predictions_soft, targets_one_hot, beta=self.delta)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    pass


class BinaryCrossEntropyLoss(nn.BCEWithLogitsLoss):
    pass


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, predictions, targets):
        errors = targets - predictions
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)


class customLossFunction(nn.Module):
    def __init__(self):
        super(customLossFunction, self).__init__()
    
    def forward(self, predictions, targets):
        pass


def get_loss_function(loss_name, num_classes=None, **kwargs):

    loss_functions = {
        'mse': lambda: MSELossForClassification(num_classes),
        'mae': lambda: MAELossForClassification(num_classes),
        'huber': lambda: HuberLossForClassification(num_classes, **kwargs),
        'cross_entropy': lambda: nn.CrossEntropyLoss(**kwargs),
        'ce': lambda: nn.CrossEntropyLoss(**kwargs),
        'binary_ce': lambda: nn.BCEWithLogitsLoss(**kwargs),
        'quantile': lambda: QuantileLoss(**kwargs),
        'custom': lambda: customLossFunction(),
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()]()