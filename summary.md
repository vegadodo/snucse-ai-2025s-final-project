# Learning with Noisy Labels: An Expanded Study on CIFAR-10 Classification

## Backgrounds

- Deep learning models achieve impressive performance on clean datasets, but often face challenges with noisy labels
- Real-world datasets frequently contain annotation errors, especially in large-scale data collections
- Understanding model behavior under label corruption is essential for practical applications

## Research Goals

- Quantify the relationship between label noise levels and model performance degradation
- Examine class-dependent effects of label noise across different object categories
- Evaluate the efficacy of noise-robust training methods (Mixup)

## Methodology

- Dataset
  - CIFAR-10
  - 50,000 images for training
  - 10,000 images for testing
  - 10 classes
- Model
  - Basic CNN with 3 convolutional layers and 2 fully-connected layers
- Data Manipulations
  - Uniform label noise at varied percentages (10%, 20%, 30%, 50%)
  - Class-dependent noise
    - 30% for animal classes
    - 10% for vehicle classes
  - Mixup data augmentation for noise robustness
- Training
  - Adam optimizer with learning rate scheduling
  - 20 epochs per experiment with batch size of 128
  - Best model selection based on validation performance
- Full codes and results
  - https://github.com/vegadodo/snucse-ai-2025s-final-project

## Experiments

- Performance vs Noise Level
  - Clear correlation between noise level and accuracy degradation
  - Performance degradation is not strictly linear
- Class-dependent Noise Analysis
  - Vehicle classes maintain significantly higher accuracy, while animal classes show much lower performance
  - Class characteristics strongly influence noise sensitivity
- Mixup Augmentation Evaluation
  - Mixup training showed minimal impact on 20% noise
  - Per-class analysis shows Mixup improved performance on some classes (plane: 77.0 % -> 87.1), while reducing performance on others (ship: 89.7% -> 87.1%)

## Concluding Remarks

- Neural networks demonstrate significant robustness to moderate label noise
- Performance degradation follows a non-linear pattern, with sharper drops at higher noise levels
- Class characteristics strongly influence noise sensitivity, with visually distinctive classes showing greater robustness
- Focusing cleaning efforts on naturally difficult classes may be more efficient than uniform data cleaning
