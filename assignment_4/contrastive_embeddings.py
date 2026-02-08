import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random


class ContrastiveDataset(Dataset):

    def __init__(self, base_dataset, num_pairs_per_sample=2):
        self.base_dataset = base_dataset
        self.num_pairs_per_sample = num_pairs_per_sample
        
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.labels = list(self.label_to_indices.keys())
    
    def __len__(self):
        return len(self.base_dataset) * self.num_pairs_per_sample
    
    def __getitem__(self, idx):
        anchor_idx = idx % len(self.base_dataset)
        anchor_img, anchor_label = self.base_dataset[anchor_idx]
        
        is_positive = random.random() > 0.5
        
        if is_positive:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
            pair_img, _ = self.base_dataset[positive_idx]
            label = 1.0
        else:
            negative_labels = [l for l in self.labels if l != anchor_label]
            negative_label = random.choice(negative_labels)
            negative_idx = random.choice(self.label_to_indices[negative_label])
            pair_img, _ = self.base_dataset[negative_idx]
            label = 0.0
        
        return anchor_img, pair_img, torch.tensor(label, dtype=torch.float32), anchor_label


class ContrastiveEmbeddingNet(nn.Module):

    def __init__(self, input_dim=784, embedding_dim=32):
        super(ContrastiveEmbeddingNet, self).__init__()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )
    
    def forward(self, x):
        return self.embedding_net(x)
    
    def get_embedding(self, x):
        embedding = self.forward(x)
        return F.normalize(embedding, p=2, dim=1)


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):

        distance = F.pairwise_distance(embedding1, embedding2)
        

        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_positive + loss_negative)
        return loss


def train_contrastive(model, train_loader, num_epochs=10, lr=0.001, margin=1.0, device='cpu'):

    model = model.to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print("Training Contrastive Embedding Network...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (anchor, pair, label, _) in enumerate(train_loader):
            # Flatten images
            anchor = anchor.view(anchor.size(0), -1).to(device)
            pair = pair.view(pair.size(0), -1).to(device)
            label = label.to(device)
            
            # Forward pass
            embedding1 = model.get_embedding(anchor)
            embedding2 = model.get_embedding(pair)
            
            loss = criterion(embedding1, embedding2, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            distance = F.pairwise_distance(embedding1, embedding2)
            predicted = (distance < margin/2).float()  # If distance < threshold, predict similar
            correct += (predicted == label).sum().item()
            total += label.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
    
    return losses


def extract_contrastive_embeddings(model, data_loader, device='cpu'):

    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (anchor, _, _, anchor_label) in enumerate(data_loader):
            anchor = anchor.view(anchor.size(0), -1).to(device)
            embedding = model.get_embedding(anchor)
            embeddings.append(embedding.cpu().numpy())
            labels.append(anchor_label.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels


def visualize_contrastive_embeddings(embeddings, labels, title="Contrastive Embeddings", save_path=None):

    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding visualization to {save_path}")
    
    plt.show()


def visualize_embedding_distances(model, test_dataset, device='cpu', num_samples=100, save_path=None):

    model.eval()
    positive_distances = []
    negative_distances = []
    

    label_to_indices = {}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    labels = list(label_to_indices.keys())
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Random anchor
            anchor_label = random.choice(labels)
            anchor_idx = random.choice(label_to_indices[anchor_label])
            anchor_img, _ = test_dataset[anchor_idx]
            anchor_img = anchor_img.view(1, -1).to(device)
            anchor_emb = model.get_embedding(anchor_img)
            
            # Positive pair
            positive_idx = random.choice(label_to_indices[anchor_label])
            positive_img, _ = test_dataset[positive_idx]
            positive_img = positive_img.view(1, -1).to(device)
            positive_emb = model.get_embedding(positive_img)
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb).item()
            positive_distances.append(pos_dist)
            
            # Negative pair
            negative_labels = [l for l in labels if l != anchor_label]
            negative_label = random.choice(negative_labels)
            negative_idx = random.choice(label_to_indices[negative_label])
            negative_img, _ = test_dataset[negative_idx]
            negative_img = negative_img.view(1, -1).to(device)
            negative_emb = model.get_embedding(negative_img)
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb).item()
            negative_distances.append(neg_dist)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(positive_distances, bins=30, alpha=0.5, label='Positive pairs (same class)', color='green')
    plt.hist(negative_distances, bins=30, alpha=0.5, label='Negative pairs (different class)', color='red')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Embedding Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distance distribution to {save_path}")
    
    plt.show()
    
    print(f"\nDistance Statistics:")
    print(f"Positive pairs - Mean: {np.mean(positive_distances):.4f}, Std: {np.std(positive_distances):.4f}")
    print(f"Negative pairs - Mean: {np.mean(negative_distances):.4f}, Std: {np.std(negative_distances):.4f}")


def run_contrastive_experiment(dataset_name='CIFAR10', embedding_dim=32, 
                               num_epochs=10, batch_size=256, margin=1.0, save_dir='results'):

    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == 'CIFAR10':
        train_dataset_base = datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=transform)
        test_dataset_base = datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=transform)
        input_dim = 3072  # 32x32x3
    elif dataset_name == 'FashionMNIST':
        train_dataset_base = datasets.FashionMNIST(root='./data', train=True, 
                                                    download=True, transform=transform)
        test_dataset_base = datasets.FashionMNIST(root='./data', train=False, 
                                                   download=True, transform=transform)
        input_dim = 784  # 28x28x1
    elif dataset_name == 'MNIST':
        train_dataset_base = datasets.MNIST(root='./data', train=True, 
                                            download=True, transform=transform)
        test_dataset_base = datasets.MNIST(root='./data', train=False, 
                                           download=True, transform=transform)
        input_dim = 784  # 28x28x1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dataset = ContrastiveDataset(train_dataset_base, num_pairs_per_sample=2)
    test_dataset = ContrastiveDataset(test_dataset_base, num_pairs_per_sample=1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Training pairs: {len(train_dataset)}")
    print(f"Test pairs: {len(test_dataset)}")
    print(f"Input dimension: {input_dim}")
    
    model = ContrastiveEmbeddingNet(input_dim=input_dim, embedding_dim=embedding_dim)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nEmbedding dimension: {embedding_dim}")
    print(f"Contrastive margin: {margin}")
    
    losses = train_contrastive(model, train_loader, num_epochs=num_epochs, 
                              lr=0.001, margin=margin, device=device)
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Training Loss')
    plt.grid(True)
    loss_path = os.path.join(save_dir, 'contrastive_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved loss plot to {loss_path}")
    plt.show()
    
    print("\nExtracting embeddings from test set...")
    base_test_loader = DataLoader(test_dataset_base, batch_size=batch_size, 
                                   shuffle=False, num_workers=0)
    
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in base_test_loader:
            data = data.view(data.size(0), -1).to(device)
            embedding = model.get_embedding(data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(target.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    print(f"Embedding shape: {embeddings.shape}")
    
    emb_path = os.path.join(save_dir, 'contrastive_embeddings.png')
    visualize_contrastive_embeddings(embeddings, labels, 
                                     title=f"Contrastive Embeddings ({dataset_name}, dim={embedding_dim})",
                                     save_path=emb_path)
    
    dist_path = os.path.join(save_dir, 'contrastive_distances.png')
    visualize_embedding_distances(model, test_dataset_base, device=device, 
                                  num_samples=200, save_path=dist_path)
    
    model_path = os.path.join(save_dir, 'contrastive_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    return model, embeddings, labels


if __name__ == "__main__":
    model, embeddings, labels = run_contrastive_experiment(
        dataset_name='CIFAR10',
        embedding_dim=32,
        num_epochs=10,
        batch_size=256,
        margin=1.0,
        save_dir='results_contrastive'
    )
