import torch
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..models.resnet import ResNet
from ..data import get_cifar10_data
from .occlusion_sensitivity import predict_single, apply_occlusion


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def measure_confidence_changes(model, image, image_index, patch_size, stride, device):
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    pred_class, original_confidence, _ = predict_single(model, image, device)
    
    print(f"\n{'='*80}")
    print(f"MEASURING CONFIDENCE CHANGES - Image {image_index}")
    print(f"{'='*80}")
    print(f"Original Prediction: {CIFAR10_CLASSES[pred_class]}")
    print(f"Original Confidence: {original_confidence*100:.2f}%")
    print(f"Patch Size: {patch_size}x{patch_size}, Stride: {stride}")
    print(f"{'='*80}\n")
    
    _, c, h, w = image.shape
    
    num_rows = (h - patch_size) // stride + 1
    num_cols = (w - patch_size) // stride + 1
    
    results = []
    
    print("Testing occlusion at different positions...\n")
    
    for i in range(num_rows):
        for j in range(num_cols):
            y = i * stride
            x = j * stride
            
            occluded_image = apply_occlusion(image, x, y, patch_size)
            
            pred_class_occ, confidence_occ, _ = predict_single(model, occluded_image, device)
            
            confidence_drop = original_confidence - confidence_occ
            
            results.append({
                'Position': f"({y},{x})",
                'Original_Conf_%': f"{original_confidence*100:.2f}%",
                'Occluded_Conf_%': f"{confidence_occ*100:.2f}%",
                'Drop_%': f"{confidence_drop*100:.2f}%",
                'Drop_Value': confidence_drop
            })
    
    df = pd.DataFrame(results)
    
    print("CONFIDENCE CHANGES TABLE:")
    print("-" * 80)
    print(df[['Position', 'Original_Conf_%', 'Occluded_Conf_%', 'Drop_%']].to_string(index=False))
    print("-" * 80)
    
    top_3 = df.nlargest(3, 'Drop_Value')
    print(f"\nMOST IMPORTANT POSITIONS (Highest confidence drop):")
    for idx, row in top_3.iterrows():
        print(f"  {row['Position']}: {row['Drop_%']} drop")
    
    bottom_3 = df.nsmallest(3, 'Drop_Value')
    print(f"\nLEAST IMPORTANT POSITIONS (Lowest confidence drop):")
    for idx, row in bottom_3.iterrows():
        print(f"  {row['Position']}: {row['Drop_%']} drop")
    
    print(f"\n{'='*80}\n")
    
    return df


def save_occluded_image_visualization(image, position, patch_size, confidence_before, confidence_after, class_name, output_path):
    y, x = position
    
    occluded_image = apply_occlusion(image.unsqueeze(0), x, y, patch_size).squeeze(0)
    
    if image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        occluded_display = occluded_image.permute(1, 2, 0).cpu().numpy()
        occluded_display = (occluded_display - occluded_display.min()) / (occluded_display.max() - occluded_display.min())
    else:
        img_display = image.squeeze().cpu().numpy()
        occluded_display = occluded_image.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_display, cmap='gray' if image.shape[0] == 1 else None)
    axes[0].set_title(f'Original Image\nConfidence: {confidence_before:.2f}%')
    axes[0].axis('off')
    
    axes[1].imshow(occluded_display, cmap='gray' if image.shape[0] == 1 else None)
    rect = plt.Rectangle((x, y), patch_size, patch_size, linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title(f'Occluded at Position ({y},{x})\nConfidence: {confidence_after:.2f}%')
    axes[1].axis('off')
    
    confidence_drop = confidence_before - confidence_after
    fig.suptitle(f'Most Important Region for {class_name}\nConfidence Drop: {confidence_drop:.2f}%', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved to {output_path}")
    print(f"  Shows 8x8 patch at position ({y},{x}) with red border")


def main():
    parser = argparse.ArgumentParser(description="Measure confidence changes for image 3")
    parser.add_argument("--checkpoint", type=str, default="weights/best_resnet_cifar10_model.pth", help="Path to model checkpoint")
    parser.add_argument("--patch_size", type=int, default=8, help="Size of occlusion patch")
    parser.add_argument("--stride", type=int, default=2, help="Stride for sliding window")
    parser.add_argument("--output_csv", type=str, default="src/partA/image3_confidence_table.csv", help="Output CSV file")
    parser.add_argument("--output_image", type=str, default="src/partA/most_important_occlusion.png", help="Output visualization image")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading ResNet model...")
    model = ResNet(input_channels=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    print("Loading image 3 from CIFAR-10 test set...")
    _, test_loader = get_cifar10_data(data_dir="data/cifar10")
    test_dataset = test_loader.dataset
    
    image, true_label = test_dataset[3]
    print(f"True label: {CIFAR10_CLASSES[true_label]}\n")
    
    pred_class, original_confidence, _ = predict_single(model, image, device)
    
    df = measure_confidence_changes(model, image, 3, args.patch_size, args.stride, device)
    
    df_save = df[['Position', 'Original_Conf_%', 'Occluded_Conf_%', 'Drop_%']].copy()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_save.to_csv(args.output_csv, index=False)
    print(f"✓ Table saved to {args.output_csv}")
    print(f"  You can include this CSV in your report!")
    
    max_drop_idx = df['Drop_Value'].idxmax()
    max_drop_row = df.iloc[max_drop_idx]
    position_str = max_drop_row['Position']
    y, x = map(int, position_str.strip('()').split(','))
    
    occluded_conf_str = max_drop_row['Occluded_Conf_%'].rstrip('%')
    occluded_conf = float(occluded_conf_str)
    
    print(f"\nCreating visualization of most important region...")
    save_occluded_image_visualization(
        image, 
        (y, x), 
        args.patch_size,
        original_confidence * 100,
        occluded_conf,
        CIFAR10_CLASSES[pred_class],
        args.output_image
    )
    print(f"  Use this image in your report to show the most important region!")


if __name__ == "__main__":
    main()
