# models/mixtureOfExpert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet
from .inception import InceptionNet
from .squeezenet import SqueezeNet
from .custom import custom_model

class Router(nn.Module):
    """
    Simple gating/router network that takes an image and produces weights for each expert.
    """
    def __init__(self, input_channels=1, num_experts=4, hidden_dim=128):
        super(Router, self).__init__()
        
        # Simple CNN to process the image and produce expert weights
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        logits = self.fc(x)
        weights = F.softmax(logits, dim=1)  # g_i(x)
        
        return weights


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts with trainable router/gating.
    
    Implements: z(x) = Σ g_i(x) * z_i(x)
    Prediction: argmax(z(x))
    """
    def __init__(self, num_classes=10, freeze_experts=True):
        super(MixtureOfExperts, self).__init__()
        
        # K >= 3 experts
        self.experts = nn.ModuleList([
            ResNet(num_classes=num_classes),      # Expert A: ResNet-like
            InceptionNet(num_classes=num_classes), # Expert B: Inception-like
            SqueezeNet(num_classes=num_classes),   # Expert C: Fire/Squeeze-like
            custom_model(num_classes=num_classes)  # Expert D: SuperNetwork
        ])
        
        self.num_experts = len(self.experts)
        self.freeze_experts = freeze_experts
        
        # Optionally freeze expert parameters (train only the router)
        if freeze_experts:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False
        
        # Initialize router/gate: g(x) -> softmax
        self.router = Router(num_experts=self.num_experts)

    def forward(self, x, return_gate_weights=False):
        # Router produces mixture weights: g_i(x) using softmax
        gate_weights = self.router(x)  # Shape: (batch_size, num_experts)
        
        # Get predictions from ALL experts: z_i(x)
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # Shape: (batch_size, num_classes)
            expert_outputs.append(output)
        
        # Stack expert outputs: (batch_size, num_experts, num_classes)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Expand gate weights for broadcasting: (batch_size, num_experts, 1)
        gate_weights_expanded = gate_weights.unsqueeze(2)
        
        # Combine: z(x) = Σ g_i(x) * z_i(x)
        mixed_output = (gate_weights_expanded * expert_outputs).sum(dim=1)
        
        if return_gate_weights:
            return mixed_output, gate_weights  # Return weights for analysis
        return mixed_output


    def analyze_expert_usage(self, dataloader, device):
        """Analyze which experts are used for the dataset."""
        self.eval()
        expert_counts = torch.zeros(self.num_experts)
        expert_weights_sum = torch.zeros(self.num_experts)
        total_samples = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _, gate_weights = self.forward(inputs, return_gate_weights=True)
                
                # Count which expert has highest weight for each sample
                selected_experts = torch.argmax(gate_weights, dim=1)
                for expert_idx in range(self.num_experts):
                    expert_counts[expert_idx] += (selected_experts == expert_idx).sum().item()
                
                # Sum up all weights
                expert_weights_sum += gate_weights.sum(dim=0).cpu()
                total_samples += inputs.size(0)
        
        print(f"\n{'='*60}")
        print("Expert Usage Analysis:")
        print(f"{'='*60}")
        expert_names = ["ResNet", "Inception", "SqueezeNet", "Custom"]
        for i in range(self.num_experts):
            percentage = (expert_counts[i] / total_samples) * 100
            avg_weight = expert_weights_sum[i] / total_samples
            print(f"Expert {i} ({expert_names[i]}):")
            print(f"  Selected as top expert: {expert_counts[i]:.0f}/{total_samples} ({percentage:.2f}%)")
            print(f"  Average weight: {avg_weight:.4f}")
        print(f"{'='*60}\n")


    
    def load_expert_weights(self, expert_idx, checkpoint_path, device):
        """Load pretrained weights for a specific expert."""
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.experts[expert_idx].load_state_dict(state_dict)
        print(f"Loaded weights for expert {expert_idx} from {checkpoint_path}")
