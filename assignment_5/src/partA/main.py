import torch
import argparse
import os

from ..models.resnet import ResNet
from ..models.cnn import CNNModel
from ..models.inception import InceptionNet
from ..models.squeezenet import SqueezeNet
from ..models.custom import custom_model
from ..data import get_cifar10_data
from .occlusion_sensitivity import (
    predict_single,
    load_sample_images,
    sliding_window_occlusion,
    plot_occlusion_heatmap
)


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_model_from_checkpoint(model_type, checkpoint_path, device, input_channels=3):
    if model_type == "resnet":
        model = ResNet(input_channels=input_channels)
    elif model_type == "cnn":
        model = CNNModel(input_channels=input_channels)
    elif model_type == "inception":
        model = InceptionNet(input_channels=input_channels)
    elif model_type == "squeezenet":
        model = SqueezeNet(input_channels=input_channels)
    elif model_type == "custom":
        model = custom_model(input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Occlusion Sensitivity Analysis")
    parser.add_argument("--model", type=str, choices=["resnet", "cnn", "inception", "squeezenet", "custom"], 
                        default="resnet", help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per category")
    parser.add_argument("--patch_size", type=int, default=8, help="Size of occlusion patch")
    parser.add_argument("--stride", type=int, default=4, help="Stride for sliding window")
    parser.add_argument("--output_dir", type=str, default="src/partA/results", help="Output directory for heatmaps")
    parser.add_argument("--data_dir", type=str, default="data/cifar10", help="CIFAR-10 data directory")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.model, args.checkpoint, device)
    print("Model loaded successfully")
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_data(data_dir=args.data_dir)
    test_dataset = test_loader.dataset
    print(f"Dataset loaded: {len(test_dataset)} test images")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/correct", exist_ok=True)
    os.makedirs(f"{args.output_dir}/misclassified", exist_ok=True)
    
    print(f"\nFinding {args.num_samples} correctly classified images...")
    correct_samples = load_sample_images(test_dataset, args.num_samples, True, model, device)
    print(f"Found {len(correct_samples)} correctly classified images")
    
    print(f"\nFinding {args.num_samples} misclassified images...")
    misclassified_samples = load_sample_images(test_dataset, args.num_samples, False, model, device)
    print(f"Found {len(misclassified_samples)} misclassified images")
    
    print("\nProcessing correctly classified images...")
    for idx, sample in enumerate(correct_samples):
        print(f"  [{idx+1}/{len(correct_samples)}] Processing image {sample['index']}: "
              f"Class={CIFAR10_CLASSES[sample['true_label']]}, Confidence={sample['confidence']:.4f}")
        
        sensitivity_map = sliding_window_occlusion(
            model, 
            sample['image'], 
            sample['pred_class'], 
            args.patch_size, 
            args.stride, 
            device
        )
        
        save_path = f"{args.output_dir}/correct/image_{sample['index']}_class_{sample['true_label']}.png"
        plot_occlusion_heatmap(
            sample['image'],
            sensitivity_map,
            sample['pred_class'],
            sample['true_label'],
            save_path,
            class_names=CIFAR10_CLASSES
        )
        print(f"    Saved heatmap to {save_path}")
    
    print("\nProcessing misclassified images...")
    for idx, sample in enumerate(misclassified_samples):
        print(f"  [{idx+1}/{len(misclassified_samples)}] Processing image {sample['index']}: "
              f"True={CIFAR10_CLASSES[sample['true_label']]}, "
              f"Pred={CIFAR10_CLASSES[sample['pred_class']]}, "
              f"Confidence={sample['confidence']:.4f}")
        
        sensitivity_map = sliding_window_occlusion(
            model, 
            sample['image'], 
            sample['pred_class'], 
            args.patch_size, 
            args.stride, 
            device
        )
        
        save_path = f"{args.output_dir}/misclassified/image_{sample['index']}_true_{sample['true_label']}_pred_{sample['pred_class']}.png"
        plot_occlusion_heatmap(
            sample['image'],
            sensitivity_map,
            sample['pred_class'],
            sample['true_label'],
            save_path,
            class_names=CIFAR10_CLASSES
        )
        print(f"    Saved heatmap to {save_path}")
    
    print(f"\nOcclusion sensitivity analysis complete!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
