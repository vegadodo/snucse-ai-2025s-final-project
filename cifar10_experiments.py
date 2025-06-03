import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
from data_utils import get_data_loaders, get_cifar10_classes
from model import BasicCNN, ResNet18, train_epoch, train_epoch_mixup, evaluate_model
from plot_utils import (
    plot_training_curves, plot_class_accuracy,
    plot_comparison, plot_final_comparison
)

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate(exp_name, trainloader, testloader, device, model_type='cnn', use_mixup=False, num_epochs=20, lr=0.001):
    """Run a complete training and evaluation process."""
    if model_type == 'cnn':
        model = BasicCNN().to(device)
    elif model_type == 'resnet18':
        model = ResNet18().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Get class names
    classes = get_cifar10_classes()

    # Track results
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0.0

    # Main training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {exp_name}")

        # Train for one epoch (with or without mixup)
        if use_mixup:
            train_loss, train_acc = train_epoch_mixup(model, trainloader, device, criterion, optimizer, alpha=1.0)
        else:
            train_loss, train_acc = train_epoch(model, trainloader, device, criterion, optimizer)
            
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate the model
        test_loss, test_acc, class_correct, class_total = evaluate_model(
            model, testloader, device, criterion, classes)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update learning rate
        scheduler.step(test_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{exp_name}_best.pth')
            print("Best model saved!")

    # Final evaluation with best model
    model.load_state_dict(torch.load(f'models/{exp_name}_best.pth'))
    final_loss, final_acc, class_correct, class_total = evaluate_model(
        model, testloader, device, criterion, classes)

    # Prepare results dictionary
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'class_correct': class_correct,
        'class_total': class_total,
        'best_acc': best_acc,
        'model_type': model_type,
        'use_mixup': use_mixup
    }

    # Save results to JSON
    os.makedirs('results', exist_ok=True)
    with open(f'results/{exp_name}_results.json', 'w') as f:
        # Convert NumPy arrays to lists for JSON serialization
        serializable_results = {
            k: v if not isinstance(v, (np.ndarray, list)) or k in ['class_correct', 'class_total']
            else [float(item) for item in v]
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=4)

    return results

def main():
    # Configuration
    seed = 42
    num_epochs = 20  # Reduced for demonstration
    batch_size = 128
    lr = 0.001

    # Set random seed
    set_seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define the expanded experiments
    expanded_experiments = [
        # Experiment type, Model type, Use mixup
        ('baseline', 'cnn', False), # Baseline: Original CIFAR-10
        ('label_noise_10', 'cnn', False),  # 10% noise
        ('label_noise_20', 'cnn', False),  # 20% noise (same as mini project)
        ('label_noise_30', 'cnn', False),  # 30% noise
        ('label_noise_50', 'cnn', False),  # 50% noise
        ('class_dependent_noise', 'cnn', False), # Class-dependent noise
        ('label_noise_20', 'cnn', True)  # Test mixup for noise robustness

        # Excluded due to time constraints:
        # ('baseline', 'resnet18', False),  # Test ResNet on baseline
        # ('label_noise_20', 'resnet18', False),  # Test ResNet on noisy labels
    ]
    
    # Dictionary to store all results
    all_results = {}
    
    # Run each experiment
    for exp_type, model_type, use_mixup in expanded_experiments:
        exp_name = f"{exp_type}_{model_type}"
        if use_mixup:
            exp_name += "_mixup"
        
        print(f"\n{'='*20} Running {exp_name} experiment {'='*20}")
        
        # Get data loaders
        trainloader, testloader = get_data_loaders(exp_type, batch_size)
            
        # Run experiment
        results = train_and_evaluate(
            exp_name, trainloader, testloader, device, 
            model_type=model_type, use_mixup=use_mixup, 
            num_epochs=num_epochs, lr=lr
        )
        
        # Store results
        all_results[exp_name] = results
    
    # Generate additional comparison plots
    # 1. Noise level comparison
    noise_results = {k: v for k, v in all_results.items() if 'label_noise' in k and 'cnn' in k and 'mixup' not in k}
    plot_comparison(noise_results, 'test_acc', 'Test Accuracy vs. Noise Level')
    
    # 2. Model comparison
    model_results = {k: v for k, v in all_results.items() if ('baseline' in k or 'label_noise_20' in k) and 'mixup' not in k}
    plot_comparison(model_results, 'test_acc', 'CNN vs ResNet Performance')
    
    # 3. Mixup comparison (Excluded due to time constraints)
    # mixup_results = {k: v for k, v in all_results.items() if 'label_noise_20' in k and 'cnn' in k}
    # plot_comparison(mixup_results, 'test_acc', 'Standard vs Mixup Training with Noise')
    
    print("\nAll expanded experiments completed! Results saved to 'results' directory.")

    # 4. Run additional visualizations
    print("Generating additional visualizations...")
    from visualization import main as viz_main
    viz_main()

if __name__ == '__main__':
    main()
