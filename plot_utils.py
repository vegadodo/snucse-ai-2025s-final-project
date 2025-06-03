import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

def plot_training_curves(exp_name, train_losses, train_accs, test_losses, test_accs):
    """Plot training and testing curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{exp_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{exp_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}_training_curves.png', dpi=300)
    plt.close()

def plot_class_accuracy(exp_name, classes, class_correct, class_total):
    """Plot per-class accuracy."""
    accuracies = [100 * correct / total for correct, total in zip(class_correct, class_total)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{exp_name} - Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/{exp_name}_class_accuracy.png', dpi=300)
    plt.close()

def plot_comparison(results_dict, metric, title):
    """Plot comparison of a metric across multiple experiments."""
    plt.figure(figsize=(12, 6))
    
    for exp_name, result in results_dict.items():
        # Display a cleaner name for the legend
        display_name = exp_name.replace('_', ' ')
        plt.plot(result[metric], label=display_name)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{title.replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_final_comparison(results_dict, metric, title):
    """Plot bar chart comparing final values of a metric."""
    plt.figure(figsize=(12, 6))
    
    exp_names = []
    values = []
    
    for exp_name, result in results_dict.items():
        display_name = exp_name.replace('_', ' ')
        exp_names.append(display_name)
        values.append(result[f'final_{metric}'])
    
    plt.bar(exp_names, values)
    plt.xlabel('Experiment')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/{title.replace(" ", "_")}.png', dpi=300)
    plt.close()