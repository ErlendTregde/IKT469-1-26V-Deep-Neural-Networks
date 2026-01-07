import torch
import torch.nn as nn

class ShallowNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu', dropout=0.2):
        super(ShallowNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_name(self):
        return "ShallowNetwork"