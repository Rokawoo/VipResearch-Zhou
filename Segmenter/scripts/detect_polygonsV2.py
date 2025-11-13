"""
Polygon Detection Script for Garlic Segmentation V2
Location: scripts/detect_polygonsV2.py

Detects and visualizes all classes:
- Gripper (Green)
- Garlic Bulb (Blue)  
- Garlic Root (Red)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Class definitions
CLASSES = {
    0: {"name": "Garlic Bulb", "color": (255, 0, 0)},      # Blue (BGR)
    1: {"name": "Garlic Root", "color": (0, 0, 255)},      # Red (BGR)
    2: {"name": "Gripper", "color": (0, 255, 0)}           # Green (BGR)
}


def detect_and_visualize(
    image_path,
    model_path=None,
    conf_threshold=0.25,
    show_labels=True,
    show_confidence=True,
    fill_alpha=0.3,
    output_dir=None
):
    """
    Detect and visualize all garlic segmentation classes
    
    Args:
        image_path: Path to input image (relative or absolute)
        model_path: Path to model weights (default: scripts/runs/garlic_seg/final/weights/best.pt)
        conf_threshold: Confidence threshold (0-1)
        show_labels: Show class names
        show_confidence: Show confidence scores
        fill_alpha: Transparency for filled polygons (0-1)
        output_dir: Where to save results (default: Segmenter/results/)
    """
    
    # Setup paths
    if model_path is None:
        # Model is in scripts/runs/garlic_seg/final/weights/best.pt
        model_path = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'
    else:
        model_path = Path(model_path)
    
    if output_dir is None:
        # Output goes to Segmenter/results/
        output_dir = PROJECT_ROOT / 'results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    image_path = Path(image_path)
    if not image_path.exists():
        # Try relative to script
        image_path = SCRIPT_DIR / image_path
        if not image_path.exists():
            # Try relative to project root
            image_path = PROJECT_ROOT / image_path
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print(f"   Looking in: {model_path.parent}")
        print(f"   Script dir: {SCRIPT_DIR}")
        return None
    
    print(f"✅ Image: {image_path}")
    print(f"✅ Model: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Could not read image: {image_path}")
        return None
    
    original = img.copy()
    overlay = img.copy()
    
    # Run inference
    print(f"\n🔍 Running detection (conf={conf_threshold})...")
    results = model.predict(img, retina_masks=True, conf=conf_threshold, verbose=False)
    
    # Count detections per class
    class_counts = {0: 0, 1: 0, 2: 0}
    
    # Process each detection
    for result in results:
        if result.masks is None:
            print("No objects detected")
            continue
        
        masks = result.masks.data
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        for i, (mask, cls, conf) in enumerate(zip(masks, classes, confidences)):
            class_counts[cls] += 1
            
            # Get class info
            class_info = CLASSES.get(cls, {"name": f"Class {cls}", "color": (128, 128, 128)})
            color = class_info["color"]
            name = class_info["name"]
            
            # Convert mask to numpy
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            
            # Find contours
            contours, _ = cv2.findContours(
                mask_resized, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            # Get main contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [main_contour], color)
            
            # Draw contour outline (thicker)
            cv2.drawContours(img, [main_contour], -1, color, 3)
            
            # Calculate centroid for label
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw center point
                cv2.circle(img, (cx, cy), 5, color, -1)
                cv2.circle(img, (cx, cy), 6, (255, 255, 255), 2)
                
                # Build label text
                if show_labels or show_confidence:
                    label_parts = []
                    if show_labels:
                        label_parts.append(f"{name} #{class_counts[cls]}")
                    if show_confidence:
                        label_parts.append(f"{conf:.2f}")
                    
                    label = " | ".join(label_parts)
                    
                    # Draw label background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Position label above center
                    label_x = cx - text_w // 2
                    label_y = cy - 15
                    
                    # Draw background rectangle
                    padding = 2
                    cv2.rectangle(
                        img,
                        (label_x - padding, label_y - text_h - padding),
                        (label_x + text_w + padding, label_y + baseline + padding),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        img, label, (label_x, label_y),
                        font, font_scale, (255, 255, 255), thickness
                    )
    
    # Blend overlay with original
    img = cv2.addWeighted(img, 1.0, overlay, fill_alpha, 0)
    
    # Print detection summary
    print("\n📊 Detection Summary:")
    print("-" * 40)
    total = 0
    for cls_id, count in class_counts.items():
        if count > 0:
            name = CLASSES[cls_id]["name"]
            print(f"   {name}: {count}")
            total += count
    print("-" * 40)
    print(f"   Total: {total} objects\n")
    
    # Save result
    output_path = output_dir / f"detected_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"💾 Saved: {output_path}")
    
    # Create side-by-side comparison
    # Resize if images are too large
    max_height = 800
    if original.shape[0] > max_height:
        scale = max_height / original.shape[0]
        new_width = int(original.shape[1] * scale)
        original_resized = cv2.resize(original, (new_width, max_height))
        img_resized = cv2.resize(img, (new_width, max_height))
    else:
        original_resized = original
        img_resized = img
    
    # Add labels to images
    label_height = 30
    original_labeled = cv2.copyMakeBorder(
        original_resized, label_height, 0, 0, 0, 
        cv2.BORDER_CONSTANT, value=(50, 50, 50)
    )
    img_labeled = cv2.copyMakeBorder(
        img_resized, label_height, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(50, 50, 50)
    )
    
    cv2.putText(
        original_labeled, "Original", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        img_labeled, f"Detected ({total} objects)", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    combined = np.hstack((original_labeled, img_labeled))
    
    # Save comparison
    comparison_path = output_dir / f"comparison_{image_path.stem}.jpg"
    cv2.imwrite(str(comparison_path), combined)
    print(f"💾 Comparison: {comparison_path}")
    
    # Display
    print(f"\n👁️  Displaying result (press any key to close)...")
    cv2.imshow('Detection Results', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img


def batch_detect(image_dir, model_path=None, **kwargs):
    """
    Run detection on all images in a directory
    
    Args:
        image_dir: Directory containing images
        model_path: Path to model weights
        **kwargs: Additional arguments for detect_and_visualize
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        # Try relative to script
        test_path = SCRIPT_DIR / image_dir
        if test_path.exists():
            image_dir = test_path
        else:
            # Try relative to project root
            test_path = PROJECT_ROOT / image_dir
            if test_path.exists():
                image_dir = test_path
            else:
                print(f"❌ Directory not found: {image_dir}")
                return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
    
    # Remove duplicates (case-insensitive on Windows)
    unique_images = {}
    for img in images:
        if img.stem.lower() not in unique_images:
            unique_images[img.stem.lower()] = img
    
    images = list(unique_images.values())
    
    if not images:
        print(f"❌ No images found in: {image_dir}")
        return
    
    print(f"\n📂 Processing {len(images)} images from: {image_dir}\n")
    
    for i, img_path in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Image {i}/{len(images)}: {img_path.name}")
        print('='*60)
        
        try:
            detect_and_visualize(img_path, model_path=model_path, **kwargs)
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n✅ Batch processing complete!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GARLIC SEGMENTATION DETECTOR V2")
    print("="*60)
    print(f"Script: {SCRIPT_DIR}")
    print(f"Project: {PROJECT_ROOT}")
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        # Check if it's a directory (batch mode)
        path = Path(input_path)
        if not path.is_absolute():
            # Try relative to script
            path = SCRIPT_DIR / input_path
            if not path.exists():
                path = PROJECT_ROOT / input_path
        
        if path.exists() and path.is_dir():
            print(f"\n📁 Batch mode: {path}")
            batch_detect(path)
        else:
            print(f"\n🖼️  Single image: {input_path}")
            detect_and_visualize(input_path)
    else:
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print("\nSingle image:")
        print("  python detect_polygonsV2.py <image_path>")
        print("\nExamples:")
        print("  python detect_polygonsV2.py ../dataset/images/train/image.jpg")
        print("  python detect_polygonsV2.py dataset/images/train/image.jpg")
        print("  python detect_polygonsV2.py C:/full/path/to/image.jpg")
        print("\nBatch mode (all images in directory):")
        print("  python detect_polygonsV2.py ../dataset/images/train/")
        print("  python detect_polygonsV2.py dataset/images/val/")
        print("\n" + "="*60)
        print("PATHS:")
        print("="*60)
        print(f"  Model: scripts/runs/garlic_seg/final/weights/best.pt")
        print(f"  Output: {PROJECT_ROOT / 'results'}")
        print("\n" + "="*60)
        print("COLOR CODING:")
        print("="*60)
        print("  🔵 Garlic Bulb - Blue")
        print("  🔴 Garlic Root - Red")
        print("  🟢 Gripper - Green")
        print("="*60)