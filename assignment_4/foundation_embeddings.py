
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import os

def load_clip_model(device):
    try:
        import clip
    except ImportError:
        raise ImportError("Please install clip: uv add git+https://github.com/openai/CLIP.git")
    
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"CLIP model loaded on {device}")
    return model, preprocess

def load_cifar10_for_clip(preprocess, batch_size=256):
    print("Loading CIFAR-10 dataset with CLIP preprocessing...")
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=preprocess
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Loaded {len(testset)} test images")
    return testloader, testset

def extract_clip_embeddings(model, dataloader, device):
    print("Extracting CLIP embeddings...")
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            
            # Get image features from CLIP
            image_features = model.encode_image(images)
            
            # Normalize features (CLIP standard practice)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * len(images)} images")
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings, labels

def visualize_clip_embeddings(embeddings, labels, save_path='results/clip'):
    os.makedirs(save_path, exist_ok=True)
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.6, 
        s=10
    )
    plt.colorbar(scatter, label='Class')
    plt.title('CLIP Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'clip_embeddings.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_file}")
    plt.close()

def compute_distance_stats(embeddings, labels):
    print("Computing distance statistics...")
    
    within_distances = []
    between_distances = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        other_embeddings = embeddings[~class_mask]
        
        # Sample for efficiency
        if len(class_embeddings) > 100:
            idx = np.random.choice(len(class_embeddings), 100, replace=False)
            class_sample = class_embeddings[idx]
        else:
            class_sample = class_embeddings
        
        if len(other_embeddings) > 100:
            idx = np.random.choice(len(other_embeddings), 100, replace=False)
            other_sample = other_embeddings[idx]
        else:
            other_sample = other_embeddings
        
        # Within-class distances
        within_dist = cdist(class_sample, class_sample, metric='euclidean')
        within_distances.extend(within_dist[np.triu_indices_from(within_dist, k=1)])
        
        # Between-class distances
        between_dist = cdist(class_sample, other_sample, metric='euclidean')
        between_distances.extend(between_dist.flatten())
    
    return np.array(within_distances), np.array(between_distances)

def visualize_distances(within_dist, between_dist, save_path='results/clip'):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(within_dist, bins=50, alpha=0.6, label='Within-class', color='green')
    plt.hist(between_dist, bins=50, alpha=0.6, label='Between-class', color='red')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('CLIP Embedding Distance Distribution')
    plt.legend()
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'clip_distances.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved distance distribution to {save_file}")
    plt.close()
    
    # Print statistics
    print(f"\nDistance Statistics:")
    print(f"Within-class  - Mean: {within_dist.mean():.4f}, Std: {within_dist.std():.4f}")
    print(f"Between-class - Mean: {between_dist.mean():.4f}, Std: {between_dist.std():.4f}")
    print(f"Separation ratio: {between_dist.mean() / within_dist.mean():.4f}")

def run_clip_experiment(batch_size=256):
    print("="*60)
    print("Starting CLIP Embedding Experiment (Part B)")
    print("="*60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = load_clip_model(device)
    
    # Load data
    dataloader, dataset = load_cifar10_for_clip(preprocess, batch_size)
    
    # Extract embeddings
    embeddings, labels = extract_clip_embeddings(model, dataloader, device)
    
    # Visualize
    visualize_clip_embeddings(embeddings, labels)
    
    # Compute distances
    within_dist, between_dist = compute_distance_stats(embeddings, labels)
    visualize_distances(within_dist, between_dist)
    
    print("\n" + "="*60)
    print("CLIP experiment completed!")
    print("="*60)
    
    return embeddings, labels

if __name__ == "__main__":
    embeddings, labels = run_clip_experiment()
