import torch
from ultralytics import YOLO
import os

def train():
    """
    Optimal training configuration for:
    - 80 images, ~640 instances (6-7 labels/image)
    - Classes: Garlic Bulb, Garlic Root, Gripper
    - RTX 5090 (24GB VRAM)
    - Polygon segmentation
    """
    
    # ═══════════════════════════════════════════════════════════════
    # GPU CHECK
    # ═══════════════════════════════════════════════════════════════
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # DATASET VALIDATION
    # ═══════════════════════════════════════════════════════════════
    data_yaml = '../configs/garlic_dataset.yaml'
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    print(f"Dataset: {data_yaml}")
    print("Expected classes: Garlic Bulb, Garlic Root, Gripper\n")
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    # Medium model perfect for 640 instances
    model = YOLO('yolo11m-seg.pt')
    
    # ═══════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════
    results = model.train(
        # Dataset
        data=data_yaml,
        
        # Training duration
        epochs=1000,
        patience=75,
        
        # Image settings
        imgsz=640,
        batch=12,
        
        # Data augmentation (optimized for 640 instances)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=3,
        perspective=0.0003,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        close_mosaic=10,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Segmentation
        overlap_mask=True,
        mask_ratio=4,
        
        # Hardware
        device=0,
        workers=8,
        amp=True,
        
        # Saving
        project='runs/garlic',
        name='final',
        exist_ok=True,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
        
        # Validation
        val=True,
    )
    
    # ═══════════════════════════════════════════════════════════════
    # POST-TRAINING VALIDATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    best_model_path = 'runs/garlic/final/weights/best.pt'
    print(f"\nBest model: {best_model_path}")
    
    # Load best model and validate
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    
    print("\n" + "="*60)
    print("FINAL VALIDATION METRICS")
    print("="*60)
    print(f"mAP50-95 (Box):  {metrics.box.map:.3f}")
    print(f"mAP50 (Box):     {metrics.box.map50:.3f}")
    print(f"mAP50-95 (Mask): {metrics.seg.map:.3f}")
    print(f"mAP50 (Mask):    {metrics.seg.map50:.3f}")
    
    # Check for overfitting
    train_metrics = best_model.val(split='train')
    val_metrics = best_model.val(split='val')
    gap = train_metrics.seg.map50 - val_metrics.seg.map50
    
    print("\n" + "="*60)
    print("OVERFITTING CHECK")
    print("="*60)
    print(f"Train mAP50: {train_metrics.seg.map50:.3f}")
    print(f"Val mAP50:   {val_metrics.seg.map50:.3f}")
    print(f"Gap:         {gap:.3f}")
    
    if gap < 0.10:
        print("✅ Excellent generalization!")
    elif gap < 0.15:
        print("✅ Good generalization")
    elif gap < 0.25:
        print("⚠️  Slight overfitting (acceptable)")
    else:
        print("❌ Overfitting detected")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review training plots: runs/garlic/final/")
    print("2. Test on new images:")
    print("   from ultralytics import YOLO")
    print(f"   model = YOLO('{best_model_path}')")
    print("   results = model.predict('test_image.jpg', conf=0.25)")
    print("="*60)


if __name__ == '__main__':
    train()