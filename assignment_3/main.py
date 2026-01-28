"""
YOLO vs RT-DETR Object Detection Comparison on Pascal VOC
Minimum implementation for the assignment
"""

import os
import torch
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# Install required packages (run once):
# pip install ultralytics torch torchvision matplotlib pillow tqdm

from ultralytics import YOLO

# ============================================================================
# STEP 1: Find Pascal VOC 2007 Dataset
# ============================================================================

# VOC classes in order
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
               'train', 'tvmonitor']

# COCO class names (what YOLO/RT-DETR use)
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
                'toothbrush']

# Map COCO class indices to VOC class indices
COCO_TO_VOC = {}
for voc_idx, voc_name in enumerate(VOC_CLASSES):
    # Find matching COCO class
    for coco_idx, coco_name in enumerate(COCO_CLASSES):
        # Handle naming differences
        if voc_name == coco_name:
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'aeroplane' and coco_name == 'airplane':
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'diningtable' and coco_name == 'dining table':
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'motorbike' and coco_name == 'motorcycle':
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'pottedplant' and coco_name == 'potted plant':
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'sofa' and coco_name == 'couch':
            COCO_TO_VOC[coco_idx] = voc_idx
        elif voc_name == 'tvmonitor' and coco_name == 'tv':
            COCO_TO_VOC[coco_idx] = voc_idx

print(f"COCO to VOC mapping: {COCO_TO_VOC}")
print(f"Mapped {len(COCO_TO_VOC)} classes from COCO to VOC")

def find_pascal_voc():
    """Find already downloaded Pascal VOC 2007 dataset"""
    print("Looking for Pascal VOC 2007 dataset...")
    
    # Start from current directory and search
    base_path = Path("./data/pascal-voc-2007")
    
    if not base_path.exists():
        raise FileNotFoundError(
            "Dataset directory not found at ./data/pascal-voc-2007/\n"
            "Please ensure the dataset is downloaded to this location."
        )
    
    print(f"Searching in: {base_path.absolute()}")
    
    # Search recursively for JPEGImages and Annotations folders
    for root, dirs, files in os.walk(base_path):
        if "JPEGImages" in dirs and "Annotations" in dirs:
            voc_path = Path(root)
            # Verify it has actual data
            jpeg_dir = voc_path / "JPEGImages"
            anno_dir = voc_path / "Annotations"
            
            jpg_files = list(jpeg_dir.glob("*.jpg"))
            xml_files = list(anno_dir.glob("*.xml"))
            
            if jpg_files and xml_files:
                print(f"✓ Found VOC2007 data at: {voc_path}")
                print(f"  - {len(jpg_files)} images")
                print(f"  - {len(xml_files)} annotations")
                return str(voc_path)
    
    raise FileNotFoundError(
        f"Could not find JPEGImages/ and Annotations/ folders in {base_path}\n"
        f"Please check the dataset structure."
    )

# ============================================================================
# STEP 2: Load Dataset Annotations
# ============================================================================

def parse_voc_annotation(xml_file):
    """Parse Pascal VOC XML annotation file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    labels = []
    difficult = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in VOC_CLASSES:
            continue
            
        diff = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_CLASSES.index(label))
        difficult.append(diff)
    
    return {
        'boxes': np.array(boxes),
        'labels': np.array(labels),
        'difficult': np.array(difficult)
    }

def load_dataset(voc_path, num_images=100):
    """Load a subset of Pascal VOC dataset"""
    image_dir = Path(voc_path) / "JPEGImages"
    anno_dir = Path(voc_path) / "Annotations"
    
    # Get list of images
    image_files = sorted(list(image_dir.glob("*.jpg")))[:num_images]
    
    dataset = []
    for img_file in image_files:
        anno_file = anno_dir / (img_file.stem + '.xml')
        if anno_file.exists():
            annotations = parse_voc_annotation(str(anno_file))
            dataset.append({
                'image_path': str(img_file),
                'image_id': img_file.stem,
                'annotations': annotations
            })
    
    print(f"Loaded {len(dataset)} images")
    return dataset

# ============================================================================
# STEP 3: Load Pre-trained Models
# ============================================================================

def load_models():
    """Load YOLO and RT-DETR models"""
    print("\nLoading models...")
    
    # YOLOv8 model
    yolo_model = YOLO('yolov8n.pt')
    print("✓ YOLOv8 loaded")
    
    # RT-DETR model
    rtdetr_model = YOLO('rtdetr-l.pt')
    print("✓ RT-DETR loaded")
    
    return yolo_model, rtdetr_model

# ============================================================================
# STEP 4: Run Inference
# ============================================================================

def run_inference(model, dataset, model_name):
    """Run inference on dataset"""
    print(f"\nRunning {model_name} inference...")
    
    predictions = []
    for item in tqdm(dataset):
        results = model(item['image_path'], verbose=False)
        
        # Extract predictions
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.cpu().numpy()
        coco_classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Convert COCO classes to VOC classes
        voc_boxes = []
        voc_scores = []
        voc_labels = []
        
        for i, coco_class in enumerate(coco_classes):
            if coco_class in COCO_TO_VOC:
                voc_boxes.append(boxes[i])
                voc_scores.append(scores[i])
                voc_labels.append(COCO_TO_VOC[coco_class])
        
        predictions.append({
            'image_id': item['image_id'],
            'boxes': np.array(voc_boxes),
            'scores': np.array(voc_scores),
            'labels': np.array(voc_labels)
        })
    
    return predictions

# ============================================================================
# STEP 5: Calculate IoU and mAP
# ============================================================================

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_ap(predictions, ground_truths, class_id, iou_threshold=0.5):
    """Calculate Average Precision for a single class"""
    # Collect all predictions and ground truths for this class
    pred_list = []
    for pred in predictions:
        mask = pred['labels'] == class_id
        for i in range(len(pred['boxes'])):
            if mask[i]:
                pred_list.append({
                    'image_id': pred['image_id'],
                    'box': pred['boxes'][i],
                    'score': pred['scores'][i]
                })
    
    # Sort by confidence
    pred_list.sort(key=lambda x: x['score'], reverse=True)
    
    # Count ground truth objects
    n_gt = sum([np.sum(gt['annotations']['labels'] == class_id) 
                for gt in ground_truths])
    
    if n_gt == 0 or len(pred_list) == 0:
        return 0.0
    
    # Match predictions to ground truths
    tp = np.zeros(len(pred_list))
    fp = np.zeros(len(pred_list))
    
    gt_matched = {gt['image_id']: [] for gt in ground_truths}
    
    for i, pred in enumerate(pred_list):
        # Find ground truth for this image
        gt = next((g for g in ground_truths if g['image_id'] == pred['image_id']), None)
        if gt is None:
            fp[i] = 1
            continue
        
        # Find matching ground truth boxes
        gt_boxes = gt['annotations']['boxes'][gt['annotations']['labels'] == class_id]
        
        max_iou = 0
        max_idx = -1
        for j, gt_box in enumerate(gt_boxes):
            if j in gt_matched[pred['image_id']]:
                continue
            iou = calculate_iou(pred['box'], gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        if max_iou >= iou_threshold:
            tp[i] = 1
            gt_matched[pred['image_id']].append(max_idx)
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calculate AP (11-point interpolation)
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """Calculate mean Average Precision"""
    aps = []
    for class_id in range(len(VOC_CLASSES)):
        ap = calculate_ap(predictions, ground_truths, class_id, iou_threshold)
        aps.append(ap)
        if ap > 0:
            print(f"  {VOC_CLASSES[class_id]:15s}: {ap:.3f}")
    
    mean_ap = np.mean(aps)
    return mean_ap, aps

# ============================================================================
# STEP 6: Find Failure Cases
# ============================================================================

def calculate_image_score(pred, gt):
    """Calculate F1 score for a single image using 1-to-1 matching"""
    gt_boxes = gt['annotations']['boxes']
    gt_labels = gt['annotations']['labels']
    
    if len(gt_boxes) == 0:
        # No ground truth: score is 1.0 if no predictions, 0.0 if any predictions
        return 1.0 if len(pred['boxes']) == 0 else 0.0
    
    if len(pred['boxes']) == 0:
        # No predictions but ground truth exists
        return 0.0
    
    # Track which GT boxes have been matched (1-to-1 matching)
    gt_matched = set()
    tp = 0  # True positives
    
    # For each prediction, find best matching GT box (greedy approach)
    for pred_box, pred_label in zip(pred['boxes'], pred['labels']):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            # Skip if GT already matched or class doesn't match
            if gt_idx in gt_matched or pred_label != gt_label:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # If IoU >= 0.5, count as true positive
        if best_iou >= 0.5:
            tp += 1
            gt_matched.add(best_gt_idx)
    
    # Calculate metrics
    fp = len(pred['boxes']) - tp  # False positives
    fn = len(gt_boxes) - tp  # False negatives (unmatched GT)
    
    # Calculate F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def find_failure_cases(yolo_preds, rtdetr_preds, ground_truths):
    """Find images where one model fails and the other succeeds"""
    scores = []
    
    for yolo_pred, rtdetr_pred, gt in zip(yolo_preds, rtdetr_preds, ground_truths):
        yolo_score = calculate_image_score(yolo_pred, gt)
        rtdetr_score = calculate_image_score(rtdetr_pred, gt)
        
        scores.append({
            'image_id': gt['image_id'],
            'image_path': gt['image_path'],
            'yolo_score': yolo_score,
            'rtdetr_score': rtdetr_score,
            'diff': rtdetr_score - yolo_score,
            'gt': gt,
            'yolo_pred': yolo_pred,
            'rtdetr_pred': rtdetr_pred
        })
    
    # Filter out ties (where both models perform equally)
    # Only keep cases where there's a meaningful difference
    meaningful_diff = [s for s in scores if abs(s['diff']) > 0.01]
    
    # Sort by difference
    meaningful_diff.sort(key=lambda x: x['diff'])
    
    # YOLO fails, RT-DETR succeeds (positive diff)
    yolo_fails = meaningful_diff[-10:] if len(meaningful_diff) >= 10 else meaningful_diff
    
    # RT-DETR fails, YOLO succeeds (negative diff)
    rtdetr_fails = meaningful_diff[:10] if len(meaningful_diff) >= 10 else []
    
    return yolo_fails, rtdetr_fails

# ============================================================================
# STEP 7: Analyze Trends
# ============================================================================

def analyze_failure_trends(failure_cases, title):
    """Analyze why models fail on specific images"""
    print(f"\n{title}")
    print("=" * 60)
    
    for i, case in enumerate(failure_cases):
        gt = case['gt']['annotations']
        
        # Calculate object sizes
        if len(gt['boxes']) > 0:
            areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in gt['boxes']]
            avg_area = np.mean(areas)
            min_area = np.min(areas)
            
            # Categorize
            has_small = any(area < 32*32 for area in areas)
            has_large = any(area > 96*96 for area in areas)
            num_objects = len(gt['boxes'])
            
            print(f"\nImage {i+1}: {case['image_id']}")
            print(f"  YOLO score: {case['yolo_score']:.2f}")
            print(f"  RT-DETR score: {case['rtdetr_score']:.2f}")
            print(f"  Objects: {num_objects}")
            print(f"  Avg area: {avg_area:.0f} px²")
            print(f"  Small objects: {has_small}")
            print(f"  Large objects: {has_large}")
            print(f"  Crowded: {num_objects > 5}")

# ============================================================================
# STEP 8: Visualize Results
# ============================================================================

def visualize_comparison(case, save_path):
    """Visualize ground truth vs predictions"""
    img = Image.open(case['image_path'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground truth
    axes[0].imshow(img)
    axes[0].set_title('Ground Truth', fontsize=20, fontweight='bold', pad=15)
    for box, label in zip(case['gt']['annotations']['boxes'], 
                         case['gt']['annotations']['labels']):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                linewidth=3, edgecolor='g', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(box[0], box[1]-5, VOC_CLASSES[label], 
                    color='green', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[0].axis('off')
    
    # YOLO prediction
    axes[1].imshow(img)
    axes[1].set_title('YOLO', fontsize=20, fontweight='bold', pad=15)
    for box, label, score in zip(case['yolo_pred']['boxes'],
                                 case['yolo_pred']['labels'],
                                 case['yolo_pred']['scores']):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                linewidth=3, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(box[0], box[1]-5, f"{VOC_CLASSES[label]} {score:.2f}",
                    color='red', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[1].axis('off')
    
    # RT-DETR prediction
    axes[2].imshow(img)
    axes[2].set_title('RT-DETR', fontsize=20, fontweight='bold', pad=15)
    for box, label, score in zip(case['rtdetr_pred']['boxes'],
                                 case['rtdetr_pred']['labels'],
                                 case['rtdetr_pred']['scores']):
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                linewidth=3, edgecolor='b', facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(box[0], box[1]-5, f"{VOC_CLASSES[label]} {score:.2f}",
                    color='blue', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Find and load dataset
    voc_path = find_pascal_voc()
    dataset = load_dataset(voc_path, num_images=100)  # Use 100 images
    
    # Load models
    yolo_model, rtdetr_model = load_models()
    
    # Run inference
    yolo_predictions = run_inference(yolo_model, dataset, "YOLO")
    rtdetr_predictions = run_inference(rtdetr_model, dataset, "RT-DETR")
    
    # Calculate mAP
    print("\n" + "="*60)
    print("YOLO mAP@0.5:")
    yolo_map, yolo_aps = calculate_map(yolo_predictions, dataset, iou_threshold=0.5)
    print(f"\nYOLO Mean AP: {yolo_map:.3f}")
    
    print("\n" + "="*60)
    print("RT-DETR mAP@0.5:")
    rtdetr_map, rtdetr_aps = calculate_map(rtdetr_predictions, dataset, iou_threshold=0.5)
    print(f"\nRT-DETR Mean AP: {rtdetr_map:.3f}")
    
    # Find failure cases
    print("\n" + "="*60)
    print("Finding failure cases...")
    yolo_fails, rtdetr_fails = find_failure_cases(yolo_predictions, 
                                                   rtdetr_predictions, 
                                                   dataset)
    
    # Analyze trends
    analyze_failure_trends(yolo_fails, "Cases where YOLO FAILS, RT-DETR SUCCEEDS")
    analyze_failure_trends(rtdetr_fails, "Cases where RT-DETR FAILS, YOLO SUCCEEDS")
    
    # Visualize all failure cases
    print("\n" + "="*60)
    print("Generating visualizations...")
    os.makedirs("images/YOLO", exist_ok=True)
    os.makedirs("images/RT-DETR", exist_ok=True)
    
    print("\nSaving YOLO failure cases (where YOLO fails, RT-DETR succeeds)...")
    for i, case in enumerate(yolo_fails):
        visualize_comparison(case, f"images/YOLO/fail_{i+1:02d}_{case['image_id']}.png")
    
    print("\nSaving RT-DETR failure cases (where RT-DETR fails, YOLO succeeds)...")
    for i, case in enumerate(rtdetr_fails):
        visualize_comparison(case, f"images/RT-DETR/fail_{i+1:02d}_{case['image_id']}.png")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"YOLO mAP@0.5:    {yolo_map:.3f}")
    print(f"RT-DETR mAP@0.5: {rtdetr_map:.3f}")
    print(f"\nFailure case visualizations saved:")
    print(f"  - YOLO failures:   images/YOLO/    (10 images)")
    print(f"  - RT-DETR failures: images/RT-DETR/ (10 images)")
    print("Analysis complete!")

if __name__ == "__main__":
    main()