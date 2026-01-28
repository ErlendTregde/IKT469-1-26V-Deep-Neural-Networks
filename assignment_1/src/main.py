"""
Main entry point for Deep Neural Networks Assignment

Usage:
    python main.py --part A              # Run Part A experiments (Shallow vs Deep)
    python main.py --part B              # Run Part B experiments (Loss functions)
    python main.py --part C              # Run Part C experiments (CNNs)
    python main.py --download-wine       # Download Wine dataset
    python main.py --download-cifar10    # Download CIFAR-10 dataset
    python main.py --single              # Run single experiment (interactive)
"""

import argparse
import sys
from pathlib import Path

# Import modules
from data.preprocess import prepare_data, prepare_cifar10_data
from data.download_data import download_wine_dataset, download_cifar10, TARGET_DIR, CIFAR10_DIR
from models import create_model
from utils.train import train
from utils.lossfunctions import get_loss_function
import json
from datetime import datetime


def run_part_a():
    """Run Part A: Shallow vs Deep Networks experiments"""
    print("=" * 60)
    print("PART A: SHALLOW VS DEEP NETWORKS")
    print("Systematic Experiments")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = TARGET_DIR / "WineQT.csv"
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("Run: python main.py --download-wine")
        return
    
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    
    # Model configurations (matched parameter budget)
    MODELS = {
        'shallow': {
            'type': 'shallowNetwork.ShallowNetwork',
            'params': {'hidden_size': 256, 'activation': 'relu', 'dropout': 0.2}
        },
        'deep': {
            'type': 'deepNetwork.DeepNetwork',
            'params': {'hidden_sizes': [64, 64, 64, 64, 64, 64, 64, 64], 'activation': 'relu', 'dropout': 0.2}
        }
    }
    
    # Loss functions to test
    LOSS_FUNCTIONS = [
        ('cross_entropy', 'Cross-Entropy'),
        ('mse', 'MSE'),
        ('mae', 'MAE'),
        ('huber', 'Huber Loss'),
    ]
    
    # Prepare data
    print("\nLoading data...")
    data = prepare_data(DATA_PATH, batch_size=BATCH_SIZE)
    input_dim = data['input_dim']
    num_classes = data['num_classes']
    
    print(f"Input features: {input_dim}")
    print(f"Output classes: {num_classes}")
    
    # Run experiments
    results = {}
    experiment_num = 0
    total_experiments = len(MODELS) * len(LOSS_FUNCTIONS)
    
    for model_name, model_config in MODELS.items():
        for loss_key, loss_name in LOSS_FUNCTIONS:
            experiment_num += 1
            
            print(f"\n{'=' * 60}")
            print(f"EXPERIMENT {experiment_num}/{total_experiments}")
            print(f"Model: {model_name.upper()} | Loss: {loss_name}")
            print("=" * 60)
            
            # Create model
            model_params = model_config['params'].copy()
            model_params['input_size'] = input_dim
            model_params['output_size'] = num_classes
            
            model = create_model(model_config['type'], **model_params)
            print(f"Parameters: {model.count_parameters():,}")
            
            # Attach data loaders
            model.train_loader = data['train_loader']
            model.val_loader = data['val_loader']
            model.test_loader = data['test_loader']
            
            # Get loss function
            criterion = get_loss_function(loss_key, num_classes=num_classes)
            
            # Train model
            save_dir = f'trained/part_a/{model_name}_{loss_key}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            run_name = f'{model_name}_{loss_key}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            train(model, epochs=EPOCHS, lr=LEARNING_RATE, criterion=criterion,
                  delete_loaders=False, save_dir=save_dir, run_name=run_name)
            
            results[f"{model_name}_{loss_key}"] = {
                'model': model_name,
                'loss': loss_name,
                'parameters': model.count_parameters(),
                'save_dir': save_dir
            }
    
    # Save summary
    summary_file = f'trained/part_a/summary_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {summary_file}")
    print(f"View TensorBoard: tensorboard --logdir=runs")
    print("=" * 60)


def run_part_b():
    """Run Part B: Loss Function Experiments"""
    print("=" * 60)
    print("PART B: LOSS FUNCTION EXPERIMENTS")
    print("Analyze robustness, convergence speed, and stability")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = TARGET_DIR / "WineQT.csv"
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("Run: python main.py --download-wine")
        return
    
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    
    # Use a consistent model for fair comparison (Deep Network)
    MODEL_CONFIG = {
        'type': 'deepNetwork.DeepNetwork',
        'params': {'hidden_sizes': [64, 64, 64, 64, 64, 64, 64, 64], 'activation': 'relu', 'dropout': 0.2}
    }
    
    # All loss functions to test (suitable for classification)
    LOSS_FUNCTIONS = [
        ('cross_entropy', 'Cross-Entropy', {}),
        ('mse', 'MSE (Mean Squared Error)', {}),
        ('mae', 'MAE (Mean Absolute Error)', {}),
        ('huber', 'Huber Loss', {}),
    ]
    
    # Prepare data
    print("\nLoading data...")
    data = prepare_data(DATA_PATH, batch_size=BATCH_SIZE)
    input_dim = data['input_dim']
    num_classes = data['num_classes']
    
    print(f"Input features: {input_dim}")
    print(f"Output classes: {num_classes}")
    print(f"\nTesting {len(LOSS_FUNCTIONS)} loss functions on Deep Network")
    
    # Run experiments
    results = {}
    experiment_num = 0
    total_experiments = len(LOSS_FUNCTIONS)
    
    for loss_key, loss_name, loss_kwargs in LOSS_FUNCTIONS:
        experiment_num += 1
        
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {experiment_num}/{total_experiments}")
        print(f"Loss Function: {loss_name}")
        print("=" * 60)
        
        # Create model
        model_params = MODEL_CONFIG['params'].copy()
        model_params['input_size'] = input_dim
        model_params['output_size'] = num_classes
        
        model = create_model(MODEL_CONFIG['type'], **model_params)
        print(f"Model: {model.get_name()}")
        print(f"Parameters: {model.count_parameters():,}")
        
        # Attach data loaders
        model.train_loader = data['train_loader']
        model.val_loader = data['val_loader']
        model.test_loader = data['test_loader']
        
        # Get loss function
        try:
            if loss_key in ['mse', 'mae', 'huber']:
                criterion = get_loss_function(loss_key, num_classes=num_classes, **loss_kwargs)
            else:
                criterion = get_loss_function(loss_key, **loss_kwargs)
            print(f"Loss function initialized: {criterion.__class__.__name__}")
        except Exception as e:
            print(f"Error initializing loss function: {e}")
            continue
        
        # Train model
        save_dir = f'trained/part_b/{loss_key}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        run_name = f'part_b_{loss_key}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        train(model, epochs=EPOCHS, lr=LEARNING_RATE, criterion=criterion,
              delete_loaders=False, save_dir=save_dir, run_name=run_name)
        
        results[loss_key] = {
            'loss': loss_name,
            'parameters': model.count_parameters(),
            'save_dir': save_dir
        }
    
    # Save summary
    summary_file = f'trained/part_b/summary_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("PART B EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {summary_file}")
    print(f"\nLoss functions tested:")
    for loss_key, loss_data in results.items():
        print(f"  - {loss_data['loss']}")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS GUIDELINES")
    print("=" * 60)
    print("""
View TensorBoard to compare:
  tensorboard --logdir=runs

Key metrics to analyze:

1. CONVERGENCE SPEED:
   - Which loss reaches high accuracy fastest?
   - Compare training curves (first 10-20 epochs)
   
2. STABILITY:
   - Which loss has smoothest training curve?
   - Look for oscillations or instability
   
3. ROBUSTNESS TO OUTLIERS:
   - MSE: Sensitive to outliers (squared error)
   - MAE: Robust to outliers (linear penalty)
   - Huber: Balanced (quadratic → linear)
   - Cross-Entropy: Standard for classification
   - Quantile: Asymmetric loss, useful for certain distributions
   
4. FINAL PERFORMANCE:
   - Compare final test accuracy
   - Check confusion matrices in trained/part_b/
   
5. GENERALIZATION:
   - Compare train vs validation gap
   - Which loss overfits least?

Expected findings:
- Cross-Entropy: Best for classification (standard baseline)
- MSE: May struggle with outliers, slower convergence
- MAE: More stable, robust to outliers
- Huber: Good balance between MSE and MAE
- Quantile: Specialized use cases
""")
    print("=" * 60)


def run_part_c():
    """Run Part C: CNN Experiments on CIFAR-10"""
    print("=" * 60)
    print("PART C: CONVOLUTIONAL NETWORKS")
    print("Systematic CNN Experiments on CIFAR-10")
    print("=" * 60)
    
    # Check if CIFAR-10 is available
    if not CIFAR10_DIR.exists():
        print(f"CIFAR-10 dataset not found at {CIFAR10_DIR}")
        print("Run: python main.py --download-cifar10")
        return
    
    # Configuration
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    
    # CNN Models to test
    MODELS = {
        'shallow': {
            'type': 'shallowCNN.ShallowCNN',
            'params': {'num_classes': 10, 'dropout': 0.3}
        },
        'deep': {
            'type': 'deepCNN.DeepCNN',
            'params': {'num_classes': 10, 'dropout': 0.3}
        },
        'residual': {
            'type': 'residualCNN.ResidualCNN',
            'params': {'num_classes': 10, 'dropout': 0.3}
        }
    }
    
    # Loss functions to test
    LOSS_FUNCTIONS = [
        ('cross_entropy', 'Cross-Entropy Loss', {}),
        ('mse', 'MSE Loss', {}),
    ]
    
    # Prepare CIFAR-10 data
    print("\nLoading CIFAR-10 dataset...")
    data = prepare_cifar10_data(data_dir=CIFAR10_DIR, batch_size=BATCH_SIZE)
    print(f"Dataset prepared: {data['num_classes']} classes")
    print(f"Train samples: {len(data['train_loader'].dataset)}")
    print(f"Val samples: {len(data['val_loader'].dataset)}")
    print(f"Test samples: {len(data['test_loader'].dataset)}")
    
    # Track results
    results = {}
    total_experiments = len(MODELS) * len(LOSS_FUNCTIONS)
    experiment_num = 0
    
    # Run experiments
    for loss_key, loss_name, loss_kwargs in LOSS_FUNCTIONS:
        for model_key, model_config in MODELS.items():
            experiment_num += 1
            
            print(f"\n{'=' * 60}")
            print(f"EXPERIMENT {experiment_num}/{total_experiments}")
            print(f"Model: {model_key.upper()} CNN")
            print(f"Loss Function: {loss_name}")
            print("=" * 60)
            
            # Create model
            model_params = model_config['params'].copy()
            model = create_model(model_config['type'], **model_params)
            print(f"Model: {model.get_name()}")
            print(f"Parameters: {model.count_parameters():,}")
            
            # Attach data loaders
            model.train_loader = data['train_loader']
            model.val_loader = data['val_loader']
            model.test_loader = data['test_loader']
            
            # Get loss function
            try:
                if loss_key in ['mse', 'mae', 'huber']:
                    criterion = get_loss_function(loss_key, num_classes=10, **loss_kwargs)
                else:
                    criterion = get_loss_function(loss_key, **loss_kwargs)
                print(f"Loss function initialized: {criterion.__class__.__name__}")
            except Exception as e:
                print(f"Error initializing loss function: {e}")
                continue
            
            # Train model
            save_dir = f'trained/part_c/{loss_key}/{model_key}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            run_name = f'part_c_{loss_key}_{model_key}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            train(model, epochs=EPOCHS, lr=LEARNING_RATE, criterion=criterion,
                  delete_loaders=False, save_dir=save_dir, run_name=run_name)
            
            # Store results
            exp_key = f"{loss_key}_{model_key}"
            results[exp_key] = {
                'model': model_key,
                'loss': loss_name,
                'parameters': model.count_parameters(),
                'save_dir': save_dir
            }
    
    # Save summary
    summary_file = f'trained/part_c/summary_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("PART C EXPERIMENTS COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {summary_file}")
    print(f"\nModels tested:")
    for exp_key, exp_data in results.items():
        print(f"  - {exp_data['model']} CNN with {exp_data['loss']}: {exp_data['parameters']:,} params")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS GUIDELINES")
    print("=" * 60)
    print("""
View TensorBoard to compare:
  tensorboard --logdir=runs

Key comparisons to analyze:

1. DEPTH IMPACT (Same Loss, Different Depths):
   - Shallow CNN (3 layers) vs Deep CNN (10 layers) vs Residual CNN
   - Does deeper always mean better?
   - Compare convergence speed and final accuracy
   
2. RESIDUAL CONNECTIONS:
   - Deep CNN vs Residual CNN (same depth)
   - Do skip connections help training?
   - Compare training stability and gradient flow
   
3. PARAMETER EFFICIENCY:
   - Performance vs number of parameters
   - Which architecture is most efficient?
   
4. LOSS FUNCTION IMPACT ON CNNs:
   - Cross-Entropy vs MSE for image classification
   - How do loss functions affect CNN training?
   
5. OVERFITTING:
   - Compare train vs validation gap
   - Which architecture generalizes best?
   - Effect of dropout and batch normalization

Expected findings:
- Shallow CNN: Faster training, limited capacity
- Deep CNN: Higher capacity, potential gradient issues
- Residual CNN: Best performance, stable training
- Cross-Entropy: Standard baseline for classification
- MSE: May struggle compared to Cross-Entropy

Architecture differences:
- Shallow: 3 conv layers, ~400K parameters
- Deep: 10 conv layers, ~7M parameters  
- Residual: 10 conv layers + skip connections, ~11M parameters
""")
    print("=" * 60)


def run_single_experiment():
    """Run a single experiment interactively"""
    print("=" * 60)
    print("SINGLE EXPERIMENT MODE")
    print("=" * 60)
    
    # Data selection
    print("\nSelect dataset:")
    print("1. Wine Quality (tabular)")
    print("2. CIFAR-10 (images)")
    choice = input("Choice [1]: ").strip() or "1"
    
    if choice == "1":
        DATA_PATH = TARGET_DIR / "WineQT.csv"
        if not DATA_PATH.exists():
            print("Wine dataset not found. Run: python main.py --download-wine")
            return
        
        data = prepare_data(DATA_PATH, batch_size=32)
        input_dim = data['input_dim']
        num_classes = data['num_classes']
        
        # Model selection
        print("\nSelect model:")
        print("1. Shallow Network (1 hidden layer)")
        print("2. Deep Network (8 hidden layers)")
        model_choice = input("Choice [1]: ").strip() or "1"
        
        if model_choice == "1":
            model = create_model("shallowNetwork.ShallowNetwork",
                               input_size=input_dim, hidden_size=256,
                               output_size=num_classes, activation='relu', dropout=0.2)
        else:
            model = create_model("deepNetwork.DeepNetwork",
                               input_size=input_dim, 
                               hidden_sizes=[64, 64, 64, 64, 64, 64, 64, 64],
                               output_size=num_classes, activation='relu', dropout=0.2)
        
        # Loss function selection
        print("\nSelect loss function:")
        print("1. Cross-Entropy")
        print("2. MSE")
        print("3. MAE")
        print("4. Huber")
        loss_choice = input("Choice [1]: ").strip() or "1"
        
        loss_map = {'1': 'cross_entropy', '2': 'mse', '3': 'mae', '4': 'huber'}
        criterion = get_loss_function(loss_map.get(loss_choice, 'cross_entropy'), 
                                     num_classes=num_classes)
        
        # Attach loaders
        model.train_loader = data['train_loader']
        model.val_loader = data['val_loader']
        model.test_loader = data['test_loader']
        
        # Train
        print(f"\nTraining {model.get_name()} with {criterion.__class__.__name__}")
        train(model, epochs=50, lr=0.001, criterion=criterion, 
              delete_loaders=True, save_dir='trained')
    
    else:
        print("CIFAR-10 support coming soon...")


def main():
    parser = argparse.ArgumentParser(description='Deep Neural Networks Assignment')
    parser.add_argument('--part', type=str, choices=['A', 'B', 'C', 'a', 'b', 'c'],
                       help='Run experiments for specific part (A, B, or C)')
    parser.add_argument('--download-wine', action='store_true',
                       help='Download Wine Quality dataset')
    parser.add_argument('--download-cifar10', action='store_true',
                       help='Download CIFAR-10 dataset')
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment (interactive)')
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.download_wine:
        print("Downloading Wine Quality dataset...")
        download_wine_dataset()
    
    elif args.download_cifar10:
        print("Downloading CIFAR-10 dataset...")
        download_cifar10()
    
    elif args.part:
        part = args.part.upper()
        if part == 'A':
            run_part_a()
        elif part == 'B':
            run_part_b()
        elif part == 'C':
            run_part_c()
    
    elif args.single:
        run_single_experiment()
    
    else:
        # No arguments - show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START EXAMPLES")
        print("=" * 60)
        print("Download datasets:")
        print("  python main.py --download-wine")
        print("  python main.py --download-cifar10")
        print("\nRun experiments:")
        print("  python main.py --part A        # All Part A experiments")
        print("  python main.py --part B        # All Part B experiments")
        print("  python main.py --part C        # All Part C experiments")
        print("\nRun single experiment:")
        print("  python main.py --single")
        print("=" * 60)


if __name__ == "__main__":
    main()

