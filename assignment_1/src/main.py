########################################
#########        Imports       #########
########################################
from data.preprocess import prepare_data
from data.download_data import TARGET_DIR
from models import create_model
from utils.train import train
from utils.lossfunctions import get_loss_function

import os
from pathlib import Path


########################################
#########  Configuration      #########
########################################

DATA_PATH = TARGET_DIR / "WineQT.csv"

if not DATA_PATH.exists():
    print(f"Dataset not found at {DATA_PATH}")
    print("Please run: python assignment_1/src/data/download_data.py")
    exit(1)


########################################
#########  Prepare the Data   #########
########################################

print("=" * 60)
print("PREPARING DATA")
print("=" * 60)

data = prepare_data(DATA_PATH, batch_size=32)

input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"\nInput features: {input_dim}")
print(f"Output classes: {num_classes}")
print("=" * 60)


######################################
#########  Create the Model  #########
######################################

print("\n" + "=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = create_model(
     "deepNetwork.DeepNetwork",
     input_size=input_dim,
     hidden_sizes=[128, 128, 128, 128, 128],
     output_size=num_classes,
     activation='relu',
     dropout=0.2
)

print(f"Model: {model.get_name()}")
print(f"Parameters: {model.count_parameters():,}")

model.train_loader = data['train_loader']
model.val_loader = data['val_loader']
model.test_loader = data['test_loader']

print("=" * 60)


######################################
#########  Choose Loss Fn    #########
######################################

print("\n" + "=" * 60)
print("SELECTING LOSS FUNCTION")
print("=" * 60)


criterion = get_loss_function('cross_entropy', num_classes=num_classes)

# criterion = get_loss_function('mse', num_classes=num_classes)
# criterion = get_loss_function('mae', num_classes=num_classes)
# criterion = get_loss_function('huber', num_classes=num_classes)
# criterion = get_loss_function('label_smoothing', num_classes=num_classes, smoothing=0.1)

print(f"Loss function: {criterion.__class__.__name__}")
print("=" * 60)


######################################
#########  Train the Model   #########
######################################

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trained_model = train(
    model,
    epochs=50,
    lr=0.001,
    criterion=criterion,
    delete_loaders=True,
    save_dir='trained'
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("Check the 'trained' folder for saved models and confusion matrix")
print("Check 'runs' folder for TensorBoard logs")
print("\nTo view TensorBoard:")
print("  tensorboard --logdir=runs")
print("=" * 60)

