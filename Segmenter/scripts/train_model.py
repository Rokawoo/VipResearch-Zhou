import torch
from ultralytics import YOLO

def train():
    # Check GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # IMPORTANT: Choose the right model based on your labels
    # Use 'yolo11m-seg.pt' for segmentation (polygon labels)
    # Use 'yolo11m.pt' for detection (box labels)

    model = YOLO('yolo11m-seg.pt')  # For segmentation/ellipse fitting

    # Train
    results = model.train(
        data='../configs/garlic_dataset.yaml',  # Path to your config
        epochs=100,                          # Start with 100
        imgsz=640,                           # Image size
        batch=-1,                            # Auto batch for RTX 5090
        device=0,                            # GPU 0
        amp=True,                            # Mixed precision
        project='runs',                      # Where to save
        name='garlic_exp1',                  # Experiment name
        patience=20,                         # Early stopping
        save=True,
        save_period=10,                      # Save every 10 epochs
        plots=True,                          # Generate plots
        verbose=True,                        # Show progress
        workers=0  # IMPORTANT: Set to 0 for Windows to avoid issues
    )

    print("Training complete!")
    print(f"Best model saved at: runs/garlic_exp1/weights/best.pt")

if __name__ == '__main__':
    train()