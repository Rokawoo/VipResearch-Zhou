# scripts/train_model.py
import torch
from ultralytics import YOLO

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load YOLO11 segmentation model
model = YOLO('yolo11m-seg.pt')  # Will auto-download

# Train
results = model.train(
    data='../configs/garlic_dataset.yaml',
    epochs=100,  # Start with 100 for testing
    imgsz=1280,  # High res for small objects
    batch=-1,    # Auto batch size for your RTX 5090
    device=0,
    amp=True,    # Mixed precision for speed
    project='../runs',
    name='garlic_seg'
)

print(f"Training complete! Model saved to: runs/garlic_seg/weights/best.pt")