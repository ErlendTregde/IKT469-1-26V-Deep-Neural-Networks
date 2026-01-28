"""
Quick Start Example - Test the implementations with minimal training
This is a fast demo to verify everything works correctly.
"""

from autoencoder_embeddings import run_autoencoder_experiment
from contrastive_embeddings import run_contrastive_experiment

print("="*80)
print("QUICK START DEMO - Minimal Training")
print("="*80)
print("\nThis will train both models for just 3 epochs to quickly demonstrate functionality.")
print("For full results, run main.py instead.\n")

# Quick autoencoder test
print("\n" + "="*80)
print("Testing Autoencoder (3 epochs)...")
print("="*80)
ae_model, ae_embeddings, ae_labels = run_autoencoder_experiment(
    dataset_name='CIFAR10',
    latent_dim=16,  # Smaller for faster testing
    num_epochs=3,
    batch_size=512,
    save_dir='quick_test/autoencoder'
)

# Quick contrastive test
print("\n" + "="*80)
print("Testing Contrastive Learning (3 epochs)...")
print("="*80)
cont_model, cont_embeddings, cont_labels = run_contrastive_experiment(
    dataset_name='CIFAR10',
    embedding_dim=16,  # Smaller for faster testing
    num_epochs=3,
    batch_size=512,
    margin=1.0,
    save_dir='quick_test/contrastive'
)

print("\n" + "="*80)
print("QUICK TEST COMPLETED!")
print("="*80)
print("\nBoth implementations are working correctly.")
print("Results saved in quick_test/ directory.")
print("\nFor full training and comparison, run: python main.py")
