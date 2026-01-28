# Assignment 4A: Learning Image Embeddings from Scratch

**This is Part A of Assignment 4. Parts 4B and 4C should be implemented separately.**

This project implements two approaches for learning image embeddings from scratch using the CIFAR-10 dataset.

## Overview

### Part A1: Autoencoder-Based Image Embeddings
- Implements an encoder-decoder architecture
- Learns compressed latent representations (embeddings)
- Reconstructs images from embeddings
- Visualizes the learned embedding space

### Part A2: Contrastive Image Embeddings
- Uses positive pairs (same class) and negative pairs (different class)
- Implements contrastive loss to learn discriminative embeddings
- Pulls similar images together and pushes dissimilar images apart
- Visualizes embedding space and distance distributions

## Files

- `autoencoder_embeddings.py` - Part A1: Autoencoder implementation
- `contrastive_embeddings.py` - Part A2: Contrastive learning implementation
- `assignment_4a.py` - Main script to run both experiments and generate comparisons (formerly main.py)
- `quick_start.py` - Quick demo script (3 epochs only)
- `requirements.txt` - Required Python packages

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Run All Experiments (Assignment 4A)

To run both Part A1 and A2 with comparison:

```bash
python assignment_4a.py
```

This will:
- Train an autoencoder model
- Train a contrastive learning model
- Generate all visualizations
- Create comparison plots
- Save results to `results/` directory

### Run Individual Parts

**Part A1 only (Autoencoder):**
```bash
python autoencoder_embeddings.py
```

**Part A2 only (Contrastive Learning):**
```bash
python contrastive_embeddings.py
```

## Configuration

You can modify the configuration in `assignment_4a.py`:

```python
DATASET = 'FashionMNIST'  # or 'MNIST'
LATENT_DIM = 32           # Embedding dimension
NUM_EPOCHS = 10           # Training epochs
BATCH_SIZE = 256          # Batch size
```

## Output (Assignment 4A)

The scripts will create the following directories for Part A results:

- `results/autoencoder/` - Autoencoder results
  - `autoencoder_loss.png` - Training loss curve
  - `autoencoder_embeddings.png` - t-SNE visualization of embeddings
  - `autoencoder_reconstructions.png` - Original vs reconstructed images
  - `autoencoder_model.pth` - Saved model weights

- `results/contrastive/` - Contrastive learning results
  - `contrastive_loss.png` - Training loss curve
  - `contrastive_embeddings.png` - t-SNE visualization of embeddings
  - `contrastive_distances.png` - Distance distribution for positive/negative pairs
  - `contrastive_model.pth` - Saved model weights

- `results/` - Comparison plots
  - `embedding_comparison.png` - Side-by-side comparison of both approaches
  - `distance_comparison.png` - Within-class vs between-class distances

## Model Architectures

### Autoencoder
```
Input (784) → 256 → 128 → Latent (32) → 128 → 256 → Output (784)
```

### Contrastive Network
```
Input (784) → 256 → Dropout → 128 → Dropout → Embedding (32)
```

## Key Features

✓ Simple and efficient implementations suitable for demonstrating concepts  
✓ Automatic dataset downloading (Fashion-MNIST)  
✓ GPU support (automatically uses CUDA if available)  
✓ Comprehensive visualizations  
✓ Quantitative comparison metrics  
✓ Clean, well-documented code  

## Expected Results

- **Autoencoder**: Learns to reconstruct images, embeddings capture overall image structure
- **Contrastive**: Learns discriminative embeddings with better class separation
- **Comparison**: Contrastive learning typically shows clearer class clustering in embedding space

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 2-4 GB RAM
- (Optional) CUDA-capable GPU for faster training

## Training Time

On CPU:
- Autoencoder: ~5-10 minutes
- Contrastive: ~10-15 minutes

On GPU:
- Autoencoder: ~1-2 minutes
- Contrastive: ~2-3 minutes

## Notes

- **This is Assignment 4A only** - Parts 4B and 4C are separate implementations
- The networks are intentionally kept small for demonstration purposes
- For better results, increase `NUM_EPOCHS` or use larger architectures
- CIFAR-10 is used by default (can change to 'FashionMNIST' or 'MNIST' in assignment_4a.py)
- Results are saved in `results/` directory to keep separate from Parts B and C
