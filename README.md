<div align="center">
  <h1>Vision-Language-Action Model (VLA) Research</h1>
  <p>By Rokawoo</p>
</div>

https://github.com/user-attachments/assets/ebc63621-427d-4463-8598-0c6160ca47d5

Research combining self-supervised vision (DINOv2), vision-language alignment (CLIP), and action generation (π0) for robotics applications.

## Foundation Models

### DINOv2 (Meta AI)
Self-supervised vision transformer
* [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
* [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
* [Project Page](https://dinov2.metademolab.com/)
* [Meta AI Blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)

### CLIP (OpenAI)
Contrastive Language-Image Pre-training
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

### π0 Models
Vision-Language-Action models
* [π0: A Vision-Language-Action Flow Model](https://www.physicalintelligence.company/blog/pi0)
* [π0.5: Enhanced Vision-Language-Action Model](https://www.physicalintelligence.company/blog/pi0-5)

## VLA Caption Generator

Generate hierarchical, spatially-aware captions for training Vision-Language-Action models.

### Environment Setup

```bash
# Create conda environment with Python 3.10
conda create -n vla_caption python=3.10 -y
conda activate vla_caption

# Install PyTorch (choose one)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install dependencies
pip install transformers==4.36.2 pillow==10.2.0 numpy==1.24.3 accelerate==0.25.0 sentencepiece==0.1.99 transformers accelerate hf_xet timm
```

### Usage

```python
from vla_caption import VLACaptionGenerator

# Initialize
generator = VLACaptionGenerator()

# Generate caption
caption = generator.generate_caption("image.jpg", "kitchen cleaning")

# Get training format
data = caption.to_training_format()
```

### Output Format

```json
{
  "high_level": "clean the kitchen",
  "semantic": "pick up the plate", 
  "low_level": "move gripper to plate at position (0.35, 0.62)",
  "grounding": {
    "plate": [0.25, 0.55, 0.45, 0.70],
    "cup": [0.60, 0.50, 0.75, 0.65]
  }
}
```

### Requirements

* Python 3.10
* 8GB+ GPU VRAM (recommended) or 16GB RAM (CPU)
* 6GB disk space for model cache
