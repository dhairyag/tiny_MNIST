# Training Raw Attempts

This folder contains raw training attempts, experiments and logs from different model architectures, optimizers and training configurations used while developing the MNIST classifier.

## Files Overview

### augmntc-summary.ipynb
Contains experiments with different data augmentation strategies:

1. **Base Augmentation Pipeline**
- ShiftScaleRotate (shift=0.0625, scale=0.1, rotate=15°)
- GridDistortion, GaussNoise, Perspective transforms
- ElasticTransform and CoarseDropout
- Achieved 98.22% accuracy

2. **Aggressive Augmentations** 
- Increased rotation (30°), scale (0.15) and shift (0.1)
- Stronger distortions and noise
- Best accuracy: 96.39%

3. **Light Augmentations**
- Focus on noise and dropout
- Reduced geometric transforms
- Best accuracy: 97.86%

4. **Geometric Focus**
- Heavy use of affine and elastic transforms
- Best accuracy: 97.71%

5. **Random Application**
- OneOf blocks for transforms
- Probability-based application
- Best accuracy: 97.67%

### models-summary.ipynb
Collection of different model architectures tested:

1. **Base CNN Model**
- 3 conv blocks with BatchNorm and MaxPool
- 2,350 parameters
- Best accuracy: 98.09%

2. **Deeper Network**
- 4 conv blocks with extra 1x1 convs
- 2,950 parameters
- Best accuracy: 97.65%

3. **Dilated Convolutions**
- Used dilated convs in middle layers
- 2,686 parameters
- Best accuracy: 97.44%

4. **Bottleneck Architecture**
- Used 1x1 bottleneck layers
- 1,464 parameters
- Best accuracy: 97.08%

5. **Mixed Innovations**
- Combined attention, residuals and depthwise convs
- 3,451 parameters
- Best accuracy: 99.12%

### optimization-methods-summary.ipynb
Experiments with different optimization strategies:

1. **Learning Rate Range Test**
- Found optimal LR range: 0.01 to 0.4
- Used for cyclical learning rates

2. **OneCycleLR Results**
- max_lr=0.4, div_factor=25
- warmup_pct=0.5
- Best accuracy: 99.35%

3. **Lookahead Optimizer**
- Base Adam with lr=0.01
- k=5, alpha=0.5
- Best accuracy: 97.90%

4. **SWA Results**
- Started at epoch 6
- swa_lr=0.001
- Best accuracy: 97.80%

### optm-summary.ipynb
Additional optimization experiments:

1. **Gradient Clipping**
- max_norm=1.0
- With Adam optimizer
- Best accuracy: 97.61%

2. **Custom Schedulers**
- StepLR with gamma=0.1
- CosineAnnealingLR
- Best accuracy: 97.67%

## Key Findings

1. Light augmentations with focused geometric transforms worked better than aggressive augmentations

2. Mixed architecture innovations (attention + residuals) achieved best accuracy while maintaining parameter efficiency

3. OneCycleLR with proper LR range and warmup gave most consistent results

4. Gradient clipping helped stabilize training with higher learning rates

## Best Configuration

The best performing configuration combined:
- Mixed architecture (attention + residuals)
- Light augmentations
- OneCycleLR scheduler
- Achieved 99.45% accuracy with 3,130 parameters

This configuration was used as the basis for the final model in main_mnist.py. 