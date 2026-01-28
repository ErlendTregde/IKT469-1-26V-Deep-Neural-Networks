import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCNN(nn.Module):
    """
    Deep CNN with 8-12 convolutional layers
    Plain architecture (no skip connections)
    For CIFAR-10: Input 3x32x32 -> Output 10 classes
    """
    
    def __init__(self, num_classes=10, dropout=0.2):
        super(DeepCNN, self).__init__()
        
        # Conv block 1 (2 layers)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # Conv block 2 (2 layers)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # Conv block 3 (3 layers)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        
        # Conv block 4 (3 layers)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # After 4 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Store metadata
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
    
    def forward(self, x):
        # Conv block 1 (2 layers)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2 (2 layers)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv block 3 (3 layers)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = F.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Conv block 4 (3 layers)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = F.relu(x)
        x = self.pool(x)  # 4x4 -> 2x2
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_name(self):
        """Get model name for saving"""
        return "DeepCNN"