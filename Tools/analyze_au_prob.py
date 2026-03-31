#!/usr/bin/env python3
"""
Analyze AU Probability (Continuous Values) Distribution
Purpose: Analyze continuous AU probability values for robot continuous motion
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json

# Setup matplotlib for server environment
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_au_prob(dataset, split_name='train'):
    """
    Analyze AU probability (continuous values) distribution
    """
    print(f"\n{'='*70}")
    print(f"Analyzing AU Probability Distribution for {split_name} set")
    print(f"{'='*70}")
    
    au_prob_stats = {}
    au_prob_values = defaultdict(list)
    
    # Collect all probability values for each AU
    for sample in tqdm(dataset, desc=f"Processing {split_name}"):
        au_prob = sample['listener_au_prob']  # Dict[str, List[float]]
        
        for au_name, probs in au_prob.items():
            probs_array = np.array(probs)
            au_prob_values[au_name].extend(probs_array.tolist())
    
    # Calculate statistics for each AU
    print("\nAU Probability Statistics (Continuous Values):")
    print("-" * 100)
    print(f"{'AU':<8} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10} {'N Frames':<10}")
    print("-" * 100)
    
    for au_name in sorted(au_prob_values.keys()):
        probs = np.array(au_prob_values[au_name])
        
        stats = {
            'mean': float(np.mean(probs)),
            'median': float(np.median(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
            'p25': float(np.percentile(probs, 25)),
            'p75': float(np.percentile(probs, 75)),
            'n_frames': len(probs),
            'active_frames': int(np.sum(probs > 0.5))  # Frames with prob > 0.5
        }
        
        au_prob_stats[au_name] = stats
        
        print(f"{au_name:<8} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
              f"{stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f} {stats['n_frames']:<10}")
    
    print("\n" + "-" * 100)
    print(f"Total AUs: {len(au_prob_stats)}")
    
    return au_prob_stats, au_prob_values

def visualize_au_prob(train_au_prob_stats, train_au_prob_values, val_au_prob_stats, output_dir):
    """
    Visualize AU probability distributions
    """
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean probability comparison
    ax1 = axes[0, 0]
    au_names = sorted(train_au_prob_stats.keys())
    train_means = [train_au_prob_stats[au]['mean'] for au in au_names]
    val_means = [val_au_prob_stats[au]['mean'] for au in au_names]
    
    x = np.arange(len(au_names))
    width = 0.35
    ax1.bar(x - width/2, train_means, width, label='Train', color='steelblue')
    ax1.bar(x + width/2, val_means, width, label='Val', color='coral')
    ax1.set_xlabel('AU')
    ax1.set_ylabel('Mean Probability')
    ax1.set_title('Mean AU Probability by Split')
    ax1.set_xticks(x)
    ax1.set_xticklabels(au_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Probability distribution (histogram for all AUs combined)
    ax2 = axes[0, 1]
    all_probs = []
    for probs_list in train_au_prob_values.values():
        all_probs.extend(probs_list)
    
    ax2.hist(all_probs, bins=50, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(all_probs), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_probs):.3f}')
    ax2.axvline(np.median(all_probs), color='orange', linestyle='--',
               label=f'Median: {np.median(all_probs):.3f}')
    ax2.axvline(0.5, color='purple', linestyle='--', alpha=0.5, label='Threshold: 0.5')
    ax2.set_xlabel('Probability Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of All AU Probabilities (Train Set)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Box plot of mean probabilities
    ax3 = axes[1, 0]
    bp_data = [train_au_prob_values[au] for au in au_names]
    bp = ax3.boxplot(bp_data, labels=au_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_xlabel('AU')
    ax3.set_ylabel('Probability Value')
    ax3.set_title('AU Probability Distribution (Box Plot)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold: 0.5')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
📊 AU Probability Analysis Summary

Total AUs Analyzed: {len(train_au_prob_stats)}

Global Statistics (Train Set):
  • Mean probability: {np.mean(all_probs):.4f}
  • Median probability: {np.median(all_probs):.4f}
  • Std Dev: {np.std(all_probs):.4f}
  • Min: {np.min(all_probs):.4f}
  • Max: {np.max(all_probs):.4f}

Frames Analysis:
  • Total frames: {len(all_probs):,}
  • Frames with prob > 0.5: {int(np.sum(np.array(all_probs) > 0.5)):,}
  • Frames with prob > 0.8: {int(np.sum(np.array(all_probs) > 0.8)):,}
  • Frames with prob = 0: {int(np.sum(np.array(all_probs) == 0)):,}

🎯 Recommendations for Robot Motion:

1. Probability Thresholds:
   • High confidence (prob > 0.8): Strong motion
   • Medium confidence (0.3-0.8): Moderate motion
   • Low confidence (0-0.3): Subtle motion
   
2. Continuous Motion Control:
   • Use probability directly as motion scale
   • Smooth transitions between frames
   • Apply Gaussian filtering for stability

3. Loss Function Design:
   • Use MSE loss with continuous values
   • Weight samples by probability variation
   • Penalize large prediction jumps
"""
    
    ax4.text(0.05, 0.95, summary_text,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'au_probability_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {output_path}")
    plt.close()

def main():
    # Setup paths
    data_root = '/net/scratch/k09562zs/LLM_reaction_Robot/Reaction_DataSet/processed'
    output_dir = Path('/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/LLM_reaction_Robot/analysis_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Loading datasets...")
    train_dataset = load_from_disk(f'{data_root}/train')
    val_dataset = load_from_disk(f'{data_root}/val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Analyze AU probability
    train_au_prob_stats, train_au_prob_values = analyze_au_prob(train_dataset, 'train')
    val_au_prob_stats, val_au_prob_values = analyze_au_prob(val_dataset, 'val')
    
    # Visualize
    visualize_au_prob(train_au_prob_stats, train_au_prob_values, val_au_prob_stats, output_dir)
    
    # Save configuration for training
    prob_config = convert_to_native_types({
        'train_statistics': train_au_prob_stats,
        'val_statistics': val_au_prob_stats,
        'recommendations': {
            'loss_function': 'MSE with continuous values',
            'threshold_high_confidence': 0.8,
            'threshold_medium_confidence': 0.5,
            'threshold_low_confidence': 0.3,
            'smoothing_kernel': 'gaussian',
            'smoothing_sigma': 2.0
        }
    })
    
    with open(output_dir / 'au_prob_config.json', 'w') as f:
        json.dump(prob_config, f, indent=2)
    print(f"✓ Saved config: {output_dir / 'au_prob_config.json'}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nOutput files in: {output_dir}/")
    print("  • au_probability_analysis.png")
    print("  • au_prob_config.json")
    print()

if __name__ == '__main__':
    import matplotlib
    main()
