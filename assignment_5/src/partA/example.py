import torch
import os

from ..models.resnet import ResNet
from ..data import get_cifar10_data
from .occlusion_sensitivity import (
    predict_single,
    sliding_window_occlusion,
    plot_occlusion_heatmap
)


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def simple_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading ResNet model...")
    model = ResNet(input_channels=3)
    checkpoint_path = "weights/best_resnet_cifar10_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using:")
        print("  uv run -m src.main --model resnet --dataset cifar10 --epochs 15")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    print("Loading CIFAR-10 dataset...")
    _, test_loader = get_cifar10_data()
    test_dataset = test_loader.dataset
    
    image, true_label = test_dataset[0]
    print(f"\nAnalyzing first test image: {CIFAR10_CLASSES[true_label]}")
    
    pred_class, confidence, _ = predict_single(model, image, device)
    print(f"Prediction: {CIFAR10_CLASSES[pred_class]} (confidence: {confidence:.4f})")
    print(f"Ground truth: {CIFAR10_CLASSES[true_label]}")
    
    print("\nComputing occlusion sensitivity map...")
    patch_size = 8
    stride = 4
    
    sensitivity_map = sliding_window_occlusion(
        model,
        image,
        pred_class,
        patch_size,
        stride,
        device
    )
    
    print(f"Sensitivity map shape: {sensitivity_map.shape}")
    print(f"Max sensitivity: {sensitivity_map.max():.4f}")
    print(f"Mean sensitivity: {sensitivity_map.mean():.4f}")
    
    os.makedirs("src/partA/example_output", exist_ok=True)
    output_path = "src/partA/example_output/example_heatmap.png"
    
    print(f"\nGenerating visualization...")
    plot_occlusion_heatmap(
        image,
        sensitivity_map,
        pred_class,
        true_label,
        output_path,
        class_names=CIFAR10_CLASSES
    )
    
    print(f"Saved visualization to {output_path}")
    print("\nDone! Check the output image to see which regions are important for the prediction.")


if __name__ == "__main__":
    simple_example()
