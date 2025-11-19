"""
Video Segmentation Script for Garlic Detection
Location: scripts/detect_video.py

Processes videos with special logic:
- Garlic Bulb: Blue (if no root detected inside)
- Garlic Bulb: Red (if root centroid is inside the bulb)
- Garlic Root: Not drawn (only used to detect overlap)
- Gripper: Green
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from tqdm import tqdm

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Class IDs
CLASS_GARLIC_BULB = 0
CLASS_GARLIC_ROOT = 1
CLASS_GRIPPER = 2

# Colors (BGR)
COLOR_BULB_NORMAL = (255, 0, 0)   # Blue
COLOR_BULB_WITH_ROOT = (0, 0, 255)  # Red
COLOR_GRIPPER = (0, 255, 0)        # Green


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using OpenCV"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def process_frame(frame, model, conf_threshold=0.25, fill_alpha=0.3):
    """
    Process a single frame with segmentation logic
    
    Args:
        frame: Input frame (BGR)
        model: YOLO model
        conf_threshold: Confidence threshold
        fill_alpha: Transparency for filled polygons
        
    Returns:
        Processed frame with segmentations
    """
    img = frame.copy()
    overlay = frame.copy()
    
    # Run inference
    results = model.predict(frame, retina_masks=True, conf=conf_threshold, verbose=False)
    
    if not results or results[0].masks is None:
        return img
    
    result = results[0]
    masks = result.masks.data
    boxes = result.boxes
    classes = boxes.cls.cpu().numpy().astype(int)
    
    # Collect all detections by class
    bulb_data = []
    root_centroids = []
    gripper_data = []
    
    for mask, cls in zip(masks, classes):
        # Convert mask to numpy
        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_resized, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None
        
        # Store by class
        if cls == CLASS_GARLIC_BULB:
            bulb_data.append({
                'contour': main_contour,
                'centroid': centroid
            })
        elif cls == CLASS_GARLIC_ROOT:
            if centroid:
                root_centroids.append(centroid)
        elif cls == CLASS_GRIPPER:
            gripper_data.append({
                'contour': main_contour,
                'centroid': centroid
            })
    
    # Draw garlic bulbs (check if they contain roots)
    for bulb in bulb_data:
        contour = bulb['contour']
        
        # Check if any root centroid is inside this bulb
        has_root = False
        for root_centroid in root_centroids:
            if point_in_polygon(root_centroid, contour):
                has_root = True
                break
        
        # Choose color based on root presence
        color = COLOR_BULB_WITH_ROOT if has_root else COLOR_BULB_NORMAL
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [contour], color)
        
        # Draw contour outline
        cv2.drawContours(img, [contour], -1, color, 2)
    
    # Draw grippers
    for gripper in gripper_data:
        contour = gripper['contour']
        
        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [contour], COLOR_GRIPPER)
        
        # Draw contour outline
        cv2.drawContours(img, [contour], -1, COLOR_GRIPPER, 2)
    
    # Blend overlay with original
    img = cv2.addWeighted(img, 1.0, overlay, fill_alpha, 0)
    
    return img


def process_video(
    video_path,
    model_path=None,
    output_path=None,
    conf_threshold=0.5,
    fill_alpha=0.3,
    show_progress=True
):
    """
    Process video with garlic segmentation
    
    Args:
        video_path: Path to input video
        model_path: Path to model weights (default: scripts/runs/garlic_seg/final/weights/best.pt)
        output_path: Path to output video (default: results/output_video.mp4)
        conf_threshold: Confidence threshold (0-1)
        fill_alpha: Transparency for filled polygons (0-1)
        show_progress: Show progress bar
    """
    
    # Setup paths
    if model_path is None:
        model_path = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'
    else:
        model_path = Path(model_path)
    
    video_path = Path(video_path)
    
    if output_path is None:
        output_dir = PROJECT_ROOT / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"segmented_{video_path.stem}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    if not video_path.exists():
        # Try relative to script
        test_path = SCRIPT_DIR / video_path
        if test_path.exists():
            video_path = test_path
        else:
            # Try relative to project root
            test_path = PROJECT_ROOT / video_path
            if test_path.exists():
                video_path = test_path
            else:
                print(f"❌ Error: Video not found: {video_path}")
                return None
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        return None
    
    print(f"✅ Video: {video_path}")
    print(f"✅ Model: {model_path}")
    print(f"✅ Output: {output_path}")
    
    # Load model
    print(f"\n🤖 Loading model...")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video")
        cap.release()
        return None
    
    # Process frames
    print(f"\n🎬 Processing video...")
    
    frame_count = 0
    
    if show_progress:
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process frame
        processed_frame = process_frame(frame, model, conf_threshold, fill_alpha)
        
        # Write frame
        out.write(processed_frame)
        
        frame_count += 1
        
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {frame_count} frames")
    print(f"   Output: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_path


def process_video_with_preview(
    video_path,
    model_path=None,
    conf_threshold=0.25,
    fill_alpha=0.3
):
    """
    Process video with live preview (no output file)
    Press 'q' to quit, 'space' to pause
    
    Args:
        video_path: Path to input video
        model_path: Path to model weights
        conf_threshold: Confidence threshold
        fill_alpha: Transparency
    """
    
    # Setup paths
    if model_path is None:
        model_path = SCRIPT_DIR / 'runs' / 'garlic_seg' / 'final' / 'weights' / 'best.pt'
    else:
        model_path = Path(model_path)
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        # Try relative paths
        test_path = SCRIPT_DIR / video_path
        if test_path.exists():
            video_path = test_path
        else:
            test_path = PROJECT_ROOT / video_path
            if test_path.exists():
                video_path = test_path
    
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        return
    
    print(f"✅ Video: {video_path}")
    print(f"✅ Model: {model_path}")
    
    # Load model
    print(f"\n🤖 Loading model...")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"\n📹 Playing at {fps} FPS")
    print("   Press 'q' to quit")
    print("   Press 'space' to pause/resume")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video finished")
                break
            
            # Process frame
            processed_frame = process_frame(frame, model, conf_threshold, fill_alpha)
            
            # Display
            cv2.imshow('Garlic Segmentation (q=quit, space=pause)', processed_frame)
        
        # Handle keyboard
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Stopped by user")
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  Paused" if paused else "▶️  Resumed")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GARLIC VIDEO SEGMENTATION")
    print("="*60)
    print(f"Script: {SCRIPT_DIR}")
    print(f"Project: {PROJECT_ROOT}")
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        # Check for preview mode
        if len(sys.argv) > 2 and sys.argv[2] == '--preview':
            print("\n🎥 Preview mode (no output file)")
            process_video_with_preview(video_path)
        else:
            print("\n🎥 Processing mode (will save output)")
            
            # Optional output path
            output_path = sys.argv[2] if len(sys.argv) > 2 else None
            
            process_video(video_path, output_path=output_path)
    else:
        print("\n" + "="*60)
        print("USAGE:")
        print("="*60)
        print("\nProcess and save video:")
        print("  python detect_video.py <video_path>")
        print("  python detect_video.py <video_path> <output_path>")
        print("\nPreview mode (no output):")
        print("  python detect_video.py <video_path> --preview")
        print("\nExamples:")
        print("  python detect_video.py ../videos/garlic.mp4")
        print("  python detect_video.py videos/garlic.mp4 results/output.mp4")
        print("  python detect_video.py videos/garlic.mp4 --preview")
        print("\n" + "="*60)
        print("SEGMENTATION LOGIC:")
        print("="*60)
        print("  🔵 Garlic Bulb - Blue (normal)")
        print("  🔴 Garlic Bulb - Red (contains root)")
        print("  🟢 Gripper - Green")
        print("  ⚫ Garlic Root - Not drawn (only used for detection)")
        print("\nNOTE: If a root's center is inside a bulb, the bulb turns red")
        print("="*60)