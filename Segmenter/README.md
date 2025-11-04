# Create conda environment
conda create -n garlic-seg python=3.10 -y
conda activate garlic-seg

# Install PyTorch with CUDA (Fill in XXX)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
- See: https://pytorch.org/get-started/locally/

# Install YOLO (includes opencv, numpy, pillow, matplotlib, etc.)
pip install ultralytics

## Verify Installation
```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvidia-smi | findstr "CUDA Version"

# Check PyTorch GPU detection
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'cuDNN Version: {torch.backends.cudnn.version()}')"

# Check VRAM
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

## Quick Test
```bash
# Test YOLO
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('YOLO ready')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

# Label Images
pip install label-studio
label-studio start

## Requirements
- Windows 10/11
- NVIDIA GPU with CUDA 12.4+ support
- NVIDIA Driver 545.84+
- 16GB+ RAM recommended