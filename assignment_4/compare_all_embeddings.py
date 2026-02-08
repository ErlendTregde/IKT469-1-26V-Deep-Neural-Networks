

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import os
import torch

def load_part_a_embeddings():
    print("Loading Part A embeddings...")
    
    from autoencoder_embeddings import run_autoencoder_experiment
    print("\n--- Running Autoencoder (Part A1) ---")
    ae_model, ae_embeddings, ae_labels = run_autoencoder_experiment(num_epochs=10)
    
    from contrastive_embeddings import run_contrastive_experiment
    print("\n--- Running Contrastive (Part A2) ---")
    cont_model, cont_embeddings, cont_labels = run_contrastive_experiment(num_epochs=10)
    
    return {
        'autoencoder': (ae_embeddings, ae_labels),
        'contrastive': (cont_embeddings, cont_labels)
    }

def load_part_b_embeddings():
    print("\n--- Running CLIP (Part B) ---")
    from foundation_embeddings import run_clip_experiment
    clip_embeddings, clip_labels = run_clip_experiment()
    
    return clip_embeddings, clip_labels

def compute_all_distance_stats(embeddings_dict):
    stats = {}
    
    for name, (embeddings, labels) in embeddings_dict.items():
        print(f"\nComputing stats for {name}...")
        
        within_distances = []
        between_distances = []
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_mask = labels == label
            class_embeddings = embeddings[class_mask]
            other_embeddings = embeddings[~class_mask]
            
            sample_size = min(100, len(class_embeddings))
            if len(class_embeddings) > sample_size:
                idx = np.random.choice(len(class_embeddings), sample_size, replace=False)
                class_sample = class_embeddings[idx]
            else:
                class_sample = class_embeddings
            
            if len(other_embeddings) > 100:
                idx = np.random.choice(len(other_embeddings), 100, replace=False)
                other_sample = other_embeddings[idx]
            else:
                other_sample = other_embeddings
            
            within_dist = cdist(class_sample, class_sample, metric='euclidean')
            within_distances.extend(within_dist[np.triu_indices_from(within_dist, k=1)])
            
            between_dist = cdist(class_sample, other_sample, metric='euclidean')
            between_distances.extend(between_dist.flatten())
        
        within_distances = np.array(within_distances)
        between_distances = np.array(between_distances)
        
        stats[name] = {
            'within': within_distances,
            'between': between_distances,
            'within_mean': within_distances.mean(),
            'within_std': within_distances.std(),
            'between_mean': between_distances.mean(),
            'between_std': between_distances.std(),
            'separation_ratio': between_distances.mean() / within_distances.mean()
        }
    
    return stats

def visualize_all_embeddings(embeddings_dict, save_path='results/comparison'):
    os.makedirs(save_path, exist_ok=True)
    
    n_methods = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 6))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (name, (embeddings, labels)) in enumerate(embeddings_dict.items()):
        print(f"Computing t-SNE for {name}...")
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        scatter = axes[idx].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=10
        )
        
        axes[idx].set_title(f'{name.capitalize()}\n({embeddings.shape[1]}D)', fontsize=14)
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        
        if idx == n_methods - 1:
            plt.colorbar(scatter, ax=axes[idx], label='Class')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'all_embeddings_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison visualization to {save_file}")
    plt.close()

def visualize_distance_comparison(stats, save_path='results/comparison'):
    os.makedirs(save_path, exist_ok=True)
    
    n_methods = len(stats)
    fig, axes = plt.subplots(1, n_methods, figsize=(7*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (name, stat) in enumerate(stats.items()):
        axes[idx].hist(stat['within'], bins=50, alpha=0.6, 
                      label='Within-class', color='green')
        axes[idx].hist(stat['between'], bins=50, alpha=0.6, 
                      label='Between-class', color='red')
        axes[idx].set_xlabel('Euclidean Distance')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{name.capitalize()}\nRatio: {stat["separation_ratio"]:.3f}')
        axes[idx].legend()
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'distance_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved distance comparison to {save_file}")
    plt.close()

def visualize_metrics_comparison(stats, save_path='results/comparison'):
    os.makedirs(save_path, exist_ok=True)
    
    methods = list(stats.keys())
    within_means = [stats[m]['within_mean'] for m in methods]
    between_means = [stats[m]['between_mean'] for m in methods]
    separation_ratios = [stats[m]['separation_ratio'] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(methods, within_means, color='green', alpha=0.7)
    axes[0].set_ylabel('Mean Distance')
    axes[0].set_title('Within-Class Distance (lower is better)')
    axes[0].tick_params(axis='x', rotation=15)
    
    axes[1].bar(methods, between_means, color='red', alpha=0.7)
    axes[1].set_ylabel('Mean Distance')
    axes[1].set_title('Between-Class Distance (higher is better)')
    axes[1].tick_params(axis='x', rotation=15)
    
    axes[2].bar(methods, separation_ratios, color='blue', alpha=0.7)
    axes[2].set_ylabel('Ratio')
    axes[2].set_title('Separation Ratio (higher is better)')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'metrics_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {save_file}")
    plt.close()

def print_comparison_summary(stats):
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Dims':<8} {'Within':<12} {'Between':<12} {'Ratio':<10}")
    print("-"*80)
    
    for name, stat in stats.items():
        dims = "32" if name in ['autoencoder', 'contrastive'] else "512"
        print(f"{name.capitalize():<20} {dims:<8} "
              f"{stat['within_mean']:<12.4f} {stat['between_mean']:<12.4f} "
              f"{stat['separation_ratio']:<10.4f}")
    
    print("="*80)
    
    best_method = max(stats.items(), key=lambda x: x[1]['separation_ratio'])
    print(f"\nBest separation: {best_method[0].upper()} "
          f"(ratio: {best_method[1]['separation_ratio']:.4f})")
    print("\nNote: Higher separation ratio indicates better class discrimination")

def run_full_comparison():
    print("="*80)
    print("FULL COMPARISON: Part A vs Part B")
    print("="*80)
    
    part_a = load_part_a_embeddings()
    clip_embeddings, clip_labels = load_part_b_embeddings()
    
    all_embeddings = {
        'autoencoder': part_a['autoencoder'],
        'contrastive': part_a['contrastive'],
        'clip': (clip_embeddings, clip_labels)
    }
    
    print("\n" + "="*80)
    print("Computing comparison statistics...")
    print("="*80)
    stats = compute_all_distance_stats(all_embeddings)
    
    print("\n" + "="*80)
    print("Creating comparison visualizations...")
    print("="*80)
    visualize_all_embeddings(all_embeddings)
    visualize_distance_comparison(stats)
    visualize_metrics_comparison(stats)
    
    print_comparison_summary(stats)
    
    print("\n" + "="*80)
    print("Full comparison completed!")
    print("Results saved to results/comparison/")
    print("="*80)

if __name__ == "__main__":
    run_full_comparison()
