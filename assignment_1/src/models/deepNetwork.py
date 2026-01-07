import torch
import torch.nn as nn

class DeepNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.2):

        super(DeepNetwork, self).__init__()
        
        # Choose activation function
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
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_name(self):
        """Get model name for saving"""
        return "DeepNetwork"
