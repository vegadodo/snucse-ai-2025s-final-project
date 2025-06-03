import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Load all result files from the results directory
def load_all_results():
    results = {}
    for filename in os.listdir('results'):
        if filename.endswith('_results.json'):
            exp_name = filename.replace('_results.json', '')
            with open(f'results/{filename}', 'r') as f:
                results[exp_name] = json.load(f)
    return results

# Plot noise level comparison
def plot_noise_comparison(results):
    plt.figure(figsize=(10, 6))
    
    # Extract noise levels and corresponding final accuracies
    noise_levels = []
    accuracies = []
    cnn_exps = []
    
    for exp_name, data in results.items():
        if 'label_noise_' in exp_name and 'cnn' in exp_name and 'mixup' not in exp_name:
            noise_level = int(exp_name.split('_')[2])
            noise_levels.append(noise_level)
            accuracies.append(data['final_acc'])
            cnn_exps.append(exp_name)
    
    # Sort by noise level
    sorted_indices = np.argsort(noise_levels)
    noise_levels = [noise_levels[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.plot(noise_levels, accuracies, 'o-', linewidth=2)
    plt.xlabel('Noise Level (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Performance Degradation with Increasing Label Noise')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/noise_level_comparison.png', dpi=300)
    plt.close()

# Plot model architecture comparison
def plot_model_comparison(results):
    plt.figure(figsize=(12, 6))
    
    baselines = {}
    noisy = {}
    
    for exp_name, data in results.items():
        if 'mixup' not in exp_name:
            if 'baseline' in exp_name:
                model = exp_name.split('_')[1]
                baselines[model] = data['final_acc']
            elif 'label_noise_20' in exp_name:
                model = exp_name.split('_')[2]
                noisy[model] = data['final_acc']
    
    models = sorted(set(list(baselines.keys()) + list(noisy.keys())))
    x = np.arange(len(models))
    width = 0.35
    
    baseline_vals = [baselines.get(model, 0) for model in models]
    noisy_vals = [noisy.get(model, 0) for model in models]
    
    plt.bar(x - width/2, baseline_vals, width, label='Clean Labels')
    plt.bar(x + width/2, noisy_vals, width, label='20% Label Noise')
    
    plt.xlabel('Model Architecture')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Architecture Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.close()

# Plot mixup vs standard training
def plot_mixup_comparison(results):
    plt.figure(figsize=(10, 6))
    
    exp_names = []
    accuracies = []
    
    for exp_name, data in results.items():
        if 'label_noise_20_cnn' in exp_name:
            if 'mixup' in exp_name:
                name = 'With Mixup'
            else:
                name = 'Standard Training'
            exp_names.append(name)
            accuracies.append(data['final_acc'])
    
    plt.bar(exp_names, accuracies)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Mixup vs Standard Training with 20% Label Noise')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('results/mixup_comparison.png', dpi=300)
    plt.close()

# Plot class-dependent noise results
def plot_class_dependent_results(results):
    plt.figure(figsize=(12, 6))
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
    vehicle_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
    
    # Find relevant experiments
    baseline = None
    uniform_noise = None
    class_dependent = None
    
    for exp_name, data in results.items():
        if 'baseline_cnn' == exp_name:
            baseline = data
        elif 'label_noise_20_cnn' == exp_name:
            uniform_noise = data
        elif 'class_dependent_noise_cnn' == exp_name:
            class_dependent = data
    
    if baseline and uniform_noise and class_dependent:
        # Calculate per-class accuracy
        baseline_acc = [100 * baseline['class_correct'][i] / baseline['class_total'][i] for i in range(10)]
        uniform_acc = [100 * uniform_noise['class_correct'][i] / uniform_noise['class_total'][i] for i in range(10)]
        classdep_acc = [100 * class_dependent['class_correct'][i] / class_dependent['class_total'][i] for i in range(10)]
        
        # Set up the bar plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 7))
        rects1 = ax.bar(x - width, baseline_acc, width, label='Baseline')
        rects2 = ax.bar(x, uniform_acc, width, label='Uniform 20% Noise')
        rects3 = ax.bar(x + width, classdep_acc, width, label='Class-Dependent Noise')
        
        for i in range(10):
            if i in [0, 1, 8, 9]:  # vehicle classes
                plt.axvspan(i-0.4, i+0.4, alpha=0.1, color='blue')
            else:  # animal classes
                plt.axvspan(i-0.4, i+0.4, alpha=0.1, color='green')
        
        # Add annotation for vehicle and animal sections
        vehicle_patch = mpatches.Patch(color='blue', alpha=0.3, label='Vehicles (10% Noise)')
        animal_patch = mpatches.Patch(color='green', alpha=0.3, label='Animals (30% Noise)')
        plt.legend(handles=[vehicle_patch, animal_patch], loc='upper center')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Class-dependent Noise Effects')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        plt.savefig('results/class_dependent_noise.png', dpi=300)
        plt.close()

def main():
    # Load all results
    all_results = load_all_results()
    
    # Generate comparison visualizations
    plot_noise_comparison(all_results)
    plot_model_comparison(all_results)
    plot_mixup_comparison(all_results)
    plot_class_dependent_results(all_results)
    
    print("Generated expanded visualizations!")

if __name__ == "__main__":
    main()
