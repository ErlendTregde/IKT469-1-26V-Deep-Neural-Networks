import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def predict_single(model, image, device):
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return pred_class, confidence, probs[0]


def load_sample_images(dataset, num_samples, correctly_classified, model, device):
    samples = []
    
    for i in range(len(dataset)):
        if len(samples) >= num_samples:
            break
        
        image, true_label = dataset[i]
        pred_class, confidence, _ = predict_single(model, image, device)
        
        is_correct = (pred_class == true_label)
        
        if is_correct == correctly_classified:
            samples.append({
                'image': image,
                'true_label': true_label,
                'pred_class': pred_class,
                'confidence': confidence,
                'index': i
            })
    
    return samples


def apply_occlusion(image, x, y, patch_size, occlusion_value=0.5):
    occluded_image = image.clone()
    h, w = image.shape[-2:]
    
    x_end = min(x + patch_size, w)
    y_end = min(y + patch_size, h)
    
    if len(image.shape) == 3:
        occluded_image[:, y:y_end, x:x_end] = occlusion_value
    elif len(image.shape) == 4:
        occluded_image[:, :, y:y_end, x:x_end] = occlusion_value
    
    return occluded_image


def sliding_window_occlusion(model, image, original_pred_class, patch_size, stride, device):
    model.eval()
    
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    _, c, h, w = image.shape
    
    with torch.no_grad():
        original_output = model(image)
        original_probs = F.softmax(original_output, dim=1)
        original_confidence = original_probs[0, original_pred_class].item()
    
    num_rows = (h - patch_size) // stride + 1
    num_cols = (w - patch_size) // stride + 1
    
    sensitivity_map = np.zeros((num_rows, num_cols))
    
    with torch.no_grad():
        for i in range(num_rows):
            for j in range(num_cols):
                y = i * stride
                x = j * stride
                
                occluded_image = apply_occlusion(image, x, y, patch_size)
                
                output = model(occluded_image)
                probs = F.softmax(output, dim=1)
                confidence = probs[0, original_pred_class].item()
                
                confidence_drop = original_confidence - confidence
                sensitivity_map[i, j] = confidence_drop
    
    return sensitivity_map


def plot_occlusion_heatmap(image, sensitivity_map, pred_class, true_class, save_path, class_names=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    else:
        img_display = image.squeeze().cpu().numpy()
    
    axes[0].imshow(img_display, cmap='gray' if image.shape[0] == 1 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    h, w = image.shape[-2:]
    heatmap_pil = Image.fromarray(sensitivity_map)
    heatmap_resized = np.array(heatmap_pil.resize((w, h), Image.NEAREST))
    
    axes[1].imshow(heatmap_resized, cmap='jet_r', interpolation='nearest')
    axes[1].set_title('Occlusion Sensitivity Map')
    axes[1].axis('off')
    cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Confidence Drop')
    
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    if image.shape[0] == 3:
        overlay = img_display.copy()
        heatmap_colored = plt.cm.jet_r(heatmap_normalized)[:, :, :3]
        overlay = 0.6 * overlay + 0.4 * heatmap_colored
    else:
        overlay = np.stack([img_display] * 3, axis=-1)
        heatmap_colored = plt.cm.jet_r(heatmap_normalized)[:, :, :3]
        overlay = 0.6 * overlay + 0.4 * heatmap_colored
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    if class_names:
        pred_name = class_names[pred_class]
        true_name = class_names[true_class]
        status = "Correct" if pred_class == true_class else "Misclassified"
        fig.suptitle(f'{status}: Pred={pred_name}, True={true_name}', fontsize=14, fontweight='bold')
    else:
        status = "Correct" if pred_class == true_class else "Misclassified"
        fig.suptitle(f'{status}: Pred={pred_class}, True={true_class}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
