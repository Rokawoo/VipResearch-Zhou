# Create conda environment
conda create -n garlic_detector python=3.10 -y
conda activate garlic_detector

# Install PyTorch with CUDA (Fill in XXX)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
- See: https://pytorch.org/get-started/locally/

# Install YOLO (includes opencv, numpy, pillow, matplotlib, etc.)
pip install ultralytics