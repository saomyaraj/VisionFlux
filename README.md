# VisionFlux: Diffusion-Based Data Augmentation for Few-Shot Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

VisionFlux is a diffusion-based data augmentation pipeline designed to improve few-shot learning performance. By leveraging the generative capabilities of diffusion models, we create synthetic training examples to augment limited datasets, enhancing the performance of downstream classifiers on few-shot learning tasks.

This project builds upon previous GAN-based augmentation approaches, addressing their limitations with the improved sample quality and diversity offered by diffusion models.

## Features

- **Diffusion-based data generation**: Generate high-quality, diverse synthetic images to augment few-shot learning datasets.
- **Configurable pipeline**: Easily adjust diffusion parameters, model architecture, and training settings.
- **Support for mini-ImageNet**: Ready-to-use implementation for the popular few-shot learning benchmark.
- **Vision Transformer integration**: Train and evaluate ViT classifiers on augmented datasets.
- **Comprehensive evaluation**: Measure performance across various few-shot configurations (N-way, K-shot).
- **GPU acceleration**: Optimized for CUDA with automatic fallback to CPU.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/saomyaraj/VisionFlux.git
cd VisionFlux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
# Import the main pipeline
from diffusion_augmentation import run_diffusion_augmentation_pipeline

# Run the complete pipeline
run_diffusion_augmentation_pipeline(
    n_way=5,
    k_shot=5,
    diffusion_steps=100,
    batch_size=16,
    vit_epochs=50
)
```

### Creating a Custom Few-Shot Task

```python
from diffusion_augmentation import create_few_shot_loaders

# Create few-shot data loaders
train_loader, val_loader, test_loader = create_few_shot_loaders(
    dataset_name='mini-imagenet',
    n_way=5,
    k_shot=1,
    batch_size=16,
    seed=42
)
```

### Training a Diffusion Model

```python
from diffusion_augmentation import train_diffusion_model

# Train a diffusion model on the few-shot dataset
diffusion_model = train_diffusion_model(
    train_loader=train_loader,
    epochs=100,
    learning_rate=1e-4,
    hidden_dims=64,
    diffusion_steps=100
)
```

### Generating Synthetic Images

```python
from diffusion_augmentation import generate_images

# Generate synthetic images for a specific class
synthetic_images = generate_images(
    diffusion_model=diffusion_model,
    class_idx=0,
    num_images=100,
    diffusion_steps=100
)
```

### Training a Classifier on Augmented Data

```python
from diffusion_augmentation import train_vit

# Train a Vision Transformer on the augmented dataset
vit_model, metrics = train_vit(
    train_loader=augmented_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=1e-3
)
```

## Pipeline Overview

1. **Data Loading**: Load and preprocess the few-shot dataset
2. **Diffusion Model Training**: Train a diffusion model on the few-shot dataset
3. **Synthetic Data Generation**: Generate synthetic images for each class
4. **Data Augmentation**: Combine real and synthetic data to create an augmented dataset
5. **Classifier Training**: Train a Vision Transformer on the augmented dataset
6. **Evaluation**: Evaluate the classifier on the test set

## Configuration

The pipeline can be configured through the `Config` class:

```python
from diffusion_augmentation import Config

# Configure the pipeline
config = Config(
    n_way=5,                  # Number of classes for few-shot task
    k_shot=5,                 # Number of examples per class
    diffusion_steps=100,      # Number of diffusion steps
    batch_size=16,            # Batch size for training
    vit_img_size=224,         # Input image size for ViT
    vit_patch_size=16,        # Patch size for ViT
    vit_embed_dim=384,        # Embedding dimension for ViT
    vit_depth=6,              # Number of transformer layers
    num_synthetic_per_class=100,  # Number of synthetic images per class
    force_cpu=False,          # Force CPU usage even if GPU is available
)
```

## Acknowledgements

This work draws inspiration from:

- [Synthetic Data from Diffusion Models Improves ImageNet Classification](https://openreview.net/forum?id=DlRsoxjyPm)
- [Overcoming challenges in leveraging GANs for few-shot data augmentation](https://github.com/christopher-beckham/challenges-few-shot-gans)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<!-- ## Citation

If you find this project useful in your research, please consider citing:

```cite
@software{visionflux2023,
  author = {Your Name},
  title = {VisionFlux: Diffusion-Based Data Augmentation for Few-Shot Learning},
  year = {2023},
  url = {https://github.com/your-username/VisionFlux}
}
``` -->
