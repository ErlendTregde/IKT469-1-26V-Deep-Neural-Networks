"""
Assignment 4A - Part A1: Autoencoder-Based Image Embeddings

Implements an image autoencoder and uses its latent space as embeddings.

This is Part A of Assignment 4. Parts B and C are implemented separately.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


class ImageAutoencoder(nn.Module):
    """
    Autoencoder for image embeddings.
    Architecture: Encoder -> Latent Embedding -> Decoder
    """
    def __init__(self, input_dim=784, latent_dim=32):
        super(ImageAutoencoder, self).__init__()
        
        # Encoder: compresses input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        # Decoder: reconstructs input from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    
    def encode(self, x):
        """Extract embeddings only"""
        return self.encoder(x)


def train_autoencoder(model, train_loader, num_epochs=10, lr=0.001, device='cpu'):
    """
    Train the autoencoder model.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print("Training Autoencoder...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            # Flatten images
            data = data.view(data.size(0), -1).to(device)
            
            # Forward pass
            reconstruction, embedding = model(data)
            loss = criterion(reconstruction, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return losses


def extract_embeddings(model, data_loader, device='cpu'):
    """
    Extract embeddings and labels from the dataset.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(data.size(0), -1).to(device)
            embedding = model.encode(data)
            embeddings.append(embedding.cpu().numpy())
            labels.append(target.numpy())
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    return embeddings, labels


def visualize_embeddings(embeddings, labels, title="Autoencoder Embeddings", save_path=None):
    """
    Visualize embeddings using t-SNE.
    """
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


def visualize_reconstructions(model, test_loader, device='cpu', num_images=10, save_path=None, img_shape=(1, 28, 28)):
    """
    Visualize original vs reconstructed images.
    """
    model.eval()
    
    # Get a batch of test images
    data, _ = next(iter(test_loader))
    data = data[:num_images]
    data_flat = data.view(data.size(0), -1).to(device)
    
    with torch.no_grad():
        reconstructions, _ = model(data_flat)
    
    # Reshape back to images
    reconstructions = reconstructions.view(-1, *img_shape).cpu()
    data = data.cpu()
    
    # Determine if RGB or grayscale
    is_rgb = img_shape[0] == 3
    
    # Plot
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    for i in range(num_images):
        # Original images
        if is_rgb:
            # Convert from CHW to HWC for display
            axes[0, i].imshow(data[i].permute(1, 2, 0))
        else:
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed images
        if is_rgb:
            # Convert from CHW to HWC for display
            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).clamp(0, 1))
        else:
            axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle('Autoencoder Reconstructions')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
    
    plt.show()


def run_autoencoder_experiment(dataset_name='CIFAR10', latent_dim=32, 
                               num_epochs=10, batch_size=256, save_dir='results'):
    """
    Run complete autoencoder experiment.
    """
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
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform)
        input_dim = 3072  # 32x32x3
        img_shape = (3, 32, 32)
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                              download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                             download=True, transform=transform)
        input_dim = 784  # 28x28x1
        img_shape = (1, 28, 28)
    elif dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, 
                                       download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, 
                                      download=True, transform=transform)
        input_dim = 784  # 28x28x1
        img_shape = (1, 28, 28)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input dimension: {input_dim}")
    print(f"Image shape: {img_shape}")
    
    # Create and train model
    model = ImageAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nLatent dimension: {latent_dim}")
    
    losses = train_autoencoder(model, train_loader, num_epochs=num_epochs, 
                              lr=0.001, device=device)
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss')
    plt.grid(True)
    loss_path = os.path.join(save_dir, 'autoencoder_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved loss plot to {loss_path}")
    plt.show()
    
    # Extract embeddings
    print("\nExtracting embeddings from test set...")
    embeddings, labels = extract_embeddings(model, test_loader, device=device)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Visualize embeddings
    emb_path = os.path.join(save_dir, 'autoencoder_embeddings.png')
    visualize_embeddings(embeddings, labels, 
                        title=f"Autoencoder Embeddings ({dataset_name}, dim={latent_dim})",
                        save_path=emb_path)
    
    # Visualize reconstructions
    recon_path = os.path.join(save_dir, 'autoencoder_reconstructions.png')
    visualize_reconstructions(model, test_loader, device=device, 
                             num_images=10, save_path=recon_path, img_shape=img_shape)
    
    # Save model
    model_path = os.path.join(save_dir, 'autoencoder_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    return model, embeddings, labels


if __name__ == "__main__":
    # Run experiment
    model, embeddings, labels = run_autoencoder_experiment(
        dataset_name='CIFAR10',
        latent_dim=32,
        num_epochs=10,
        batch_size=256,
        save_dir='results_autoencoder'
    )
