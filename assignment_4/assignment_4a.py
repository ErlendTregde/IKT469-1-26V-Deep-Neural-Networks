import torch
import matplotlib.pyplot as plt
import numpy as np
from autoencoder_embeddings import run_autoencoder_experiment
from contrastive_embeddings import run_contrastive_experiment
import os


def compare_embeddings(autoencoder_embeddings, autoencoder_labels,
                      contrastive_embeddings, contrastive_labels, save_dir='results'):
    from sklearn.manifold import TSNE
    
    
    min_samples = min(len(autoencoder_embeddings), len(contrastive_embeddings))
    autoencoder_embeddings = autoencoder_embeddings[:min_samples]
    autoencoder_labels = autoencoder_labels[:min_samples]
    contrastive_embeddings = contrastive_embeddings[:min_samples]
    contrastive_labels = contrastive_labels[:min_samples]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    ae_2d = tsne.fit_transform(autoencoder_embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    cont_2d = tsne.fit_transform(contrastive_embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter1 = axes[0].scatter(ae_2d[:, 0], ae_2d[:, 1], 
                              c=autoencoder_labels, cmap='tab10', alpha=0.6, s=5)
    axes[0].set_title('Autoencoder Embeddings', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE dimension 1')
    axes[0].set_ylabel('t-SNE dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Class')
    
    scatter2 = axes[1].scatter(cont_2d[:, 0], cont_2d[:, 1], 
                              c=contrastive_labels, cmap='tab10', alpha=0.6, s=5)
    axes[1].set_title('Contrastive Embeddings', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE dimension 1')
    axes[1].set_ylabel('t-SNE dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Class')
    
    plt.suptitle('Comparison: Autoencoder vs Contrastive Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = os.path.join(save_dir, 'embedding_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison visualization to {comparison_path}")
    plt.show()
    
    
    print("\nAutoencoder Embeddings:")
    print(f"  Shape: {autoencoder_embeddings.shape}")
    print(f"  Mean: {np.mean(autoencoder_embeddings):.6f}")
    print(f"  Std: {np.std(autoencoder_embeddings):.6f}")
    print(f"  Min: {np.min(autoencoder_embeddings):.6f}")
    print(f"  Max: {np.max(autoencoder_embeddings):.6f}")
    
    print("\nContrastive Embeddings:")
    print(f"  Shape: {contrastive_embeddings.shape}")
    print(f"  Mean: {np.mean(contrastive_embeddings):.6f}")
    print(f"  Std: {np.std(contrastive_embeddings):.6f}")
    print(f"  Min: {np.min(contrastive_embeddings):.6f}")
    print(f"  Max: {np.max(contrastive_embeddings):.6f}")
    
    from scipy.spatial.distance import cdist
    
    def calculate_class_distances(embeddings, labels, num_samples=500):
        indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
        sample_embeddings = embeddings[indices]
        sample_labels = labels[indices]
        
        within_class_dists = []
        between_class_dists = []
        
        for i in range(len(sample_embeddings)):
            for j in range(i+1, len(sample_embeddings)):
                dist = np.linalg.norm(sample_embeddings[i] - sample_embeddings[j])
                if sample_labels[i] == sample_labels[j]:
                    within_class_dists.append(dist)
                else:
                    between_class_dists.append(dist)
        
        return within_class_dists, between_class_dists
    
    ae_within, ae_between = calculate_class_distances(autoencoder_embeddings, autoencoder_labels)
    cont_within, cont_between = calculate_class_distances(contrastive_embeddings, contrastive_labels)
    
    print("\nClass Separation (smaller within-class, larger between-class is better):")
    print(f"\nAutoencoder:")
    print(f"  Within-class distance: {np.mean(ae_within):.4f} ± {np.std(ae_within):.4f}")
    print(f"  Between-class distance: {np.mean(ae_between):.4f} ± {np.std(ae_between):.4f}")
    print(f"  Separation ratio: {np.mean(ae_between) / np.mean(ae_within):.4f}")
    
    print(f"\nContrastive Learning:")
    print(f"  Within-class distance: {np.mean(cont_within):.4f} ± {np.std(cont_within):.4f}")
    print(f"  Between-class distance: {np.mean(cont_between):.4f} ± {np.std(cont_between):.4f}")
    print(f"  Separation ratio: {np.mean(cont_between) / np.mean(cont_within):.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(ae_within, bins=30, alpha=0.5, label='Within-class', color='green')
    axes[0].hist(ae_between, bins=30, alpha=0.5, label='Between-class', color='red')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Autoencoder: Distance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(cont_within, bins=30, alpha=0.5, label='Within-class', color='green')
    axes[1].hist(cont_between, bins=30, alpha=0.5, label='Between-class', color='red')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Contrastive: Distance Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_comparison_path = os.path.join(save_dir, 'distance_comparison.png')
    plt.savefig(dist_comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distance comparison to {dist_comparison_path}")
    plt.show()


def main():

    print("="*80)
    print("ASSIGNMENT 4A: LEARNING IMAGE EMBEDDINGS FROM SCRATCH")
    print("Parts 4B and 4C are implemented separately")
    print("="*80)
    
    DATASET = 'CIFAR10'
    LATENT_DIM = 32
    EMBEDDING_DIM = 32
    NUM_EPOCHS = 10
    BATCH_SIZE = 256
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Embedding dimension: {LATENT_DIM}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*80)
    print("PART A1: AUTOENCODER-BASED IMAGE EMBEDDINGS")
    print("="*80)
    
    ae_model, ae_embeddings, ae_labels = run_autoencoder_experiment(
        dataset_name=DATASET,
        latent_dim=LATENT_DIM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir='results/autoencoder'
    )
    
    print("\n" + "="*80)
    print("PART A2: CONTRASTIVE IMAGE EMBEDDINGS")
    print("="*80)
    
    cont_model, cont_embeddings, cont_labels = run_contrastive_experiment(
        dataset_name=DATASET,
        embedding_dim=EMBEDDING_DIM,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        margin=1.0,
        save_dir='results/contrastive'
    )
    
    compare_embeddings(ae_embeddings, ae_labels, 
                      cont_embeddings, cont_labels,
                      save_dir='results')
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved in:")
    print(f"  - results/autoencoder/")
    print(f"  - results/contrastive/")
    print(f"  - results/ (comparison plots)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nPart A1 - Autoencoder:")
    print("  ✓ Trained encoder-decoder architecture")
    print("  ✓ Extracted latent embeddings")
    print("  ✓ Visualized reconstruction quality")
    print("  ✓ Visualized embedding space with t-SNE")
    
    print("\nPart A2 - Contrastive Learning:")
    print("  ✓ Implemented positive/negative pair sampling")
    print("  ✓ Trained with contrastive loss")
    print("  ✓ Visualized embedding space with t-SNE")
    print("  ✓ Analyzed distance distributions")
    
    print("\nComparison:")
    print("  ✓ Side-by-side embedding visualizations")
    print("  ✓ Class separation analysis")
    print("  ✓ Distance distribution comparison")


def run_part_a():
    main()


def run_part_b():
    print("\n" + "="*80)
    print("PART B: Foundation Model Image Embeddings")
    print("="*80)
    
    from foundation_embeddings import run_clip_experiment
    embeddings, labels = run_clip_experiment()
    
    return embeddings, labels


def run_full_assignment():
    print("\n" + "="*80)
    print("RUNNING FULL ASSIGNMENT 4: Parts A + B + Comparison")
    print("="*80)
    
    from compare_all_embeddings import run_full_comparison
    run_full_comparison()
    
    print("\n" + "="*80)
    print("Full assignment completed!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "parta":
            run_part_a()
        elif sys.argv[1] == "partb":
            run_part_b()
        elif sys.argv[1] == "full":
            run_full_assignment()
        else:
            print("Usage: python assignment_4a.py [parta|partb|full]")
            print("  parta - Run Part A only (Autoencoder + Contrastive)")
            print("  partb - Run Part B only (CLIP)")
            print("  full  - Run everything with full comparison")
    else:
        run_part_a()
