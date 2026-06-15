#!/usr/bin/env python3
"""
Generate ROC Comparison Plots - Load Pre-trained Models
========================================================

Load all 5 saved models and generate ROC comparison curves.
Models: LOD (Clinical+OTU+DEG), LO (Clinical+OTU), L (Clinical), O (OTU), D (DEG)

Output: 4-5cm figures with large fonts, 3 size versions
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

# Nature-style colors
COLORS = {
    'LOD': '#E64B35',   # Red - Clinical+OTU+DEG
    'LO': '#4DBBD5',    # Blue - Clinical+OTU
    'L': '#00A087',     # Teal - Clinical
    'O': '#3C5488',     # Dark blue - OTU
    'D': '#F39B7F',     # Orange - DEG
}

# Figure size (8cm square, good for publication)
FIG_SIZE = 8 / 2.54  # 8cm in inches

# Font sizes (smaller relative to figure)
FONTS = {'label': 8, 'tick': 7, 'legend': 6}


def plot_roc_comparison(results, y_test, output_path):
    """Plot ROC curves comparison for all 5 models"""

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = FONTS['tick']
    plt.rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))

    # Sort by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'] if x[1]['auc'] else 0, reverse=True)

    # Plot ROC curves
    for model_name, data in sorted_results:
        if data['prob'] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, data['prob'])
        ax.plot(fpr, tpr, color=COLORS[model_name], lw=1.5,
                label=f"{data['label']} ({data['auc']:.3f})")

    # Random line
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=FONTS['label'])
    ax.set_ylabel('True Positive Rate', fontsize=FONTS['label'])
    ax.tick_params(axis='both', labelsize=FONTS['tick'])

    # Top-left labels: Clinical+OTU+DEG, Clinical+OTU
    top_left_labels = []
    bottom_right_labels = []

    for model_name, data in sorted_results:
        if data['prob'] is None:
            continue
        if model_name in ['LOD', 'LO']:
            top_left_labels.append((model_name, f"{data['label']} ({data['auc']:.3f})"))
        else:
            bottom_right_labels.append((model_name, f"{data['label']} ({data['auc']:.3f})"))

    # Add text annotations without frame
    y_pos = 0.95
    for i, (model_name, label) in enumerate(top_left_labels):
        ax.text(0.03, y_pos - i * 0.06, label, transform=ax.transAxes,
                fontsize=FONTS['legend'], color=COLORS[model_name], fontweight='bold',
                verticalalignment='top')

    y_pos = 0.30
    for i, (model_name, label) in enumerate(bottom_right_labels):
        ax.text(0.55, y_pos - i * 0.06, label, transform=ax.transAxes,
                fontsize=FONTS['legend'], color=COLORS[model_name], fontweight='bold',
                verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


def load_models(model_type):
    """Load saved predictions and y_test directly (no re-prediction)"""

    if model_type == 'CST':
        models_path = os.path.join(DATA_DIR, "models", "cst_all_models.pkl")
    else:  # TNF
        models_path = os.path.join(DATA_DIR, "models", "tnf_all_models.pkl")

    # Load all models (includes saved prob and y_test)
    with open(models_path, 'rb') as f:
        all_models = pickle.load(f)
    
    model_names = [k for k in all_models.keys() if k != 'y_test']
    print(f"  Loaded: {os.path.basename(models_path)}")
    print(f"  Models: {model_names}")

    # Get saved y_test
    y_test = all_models['y_test']

    # Use saved predictions directly
    results = {}
    for model_name in model_names:
        model_data = all_models[model_name]
        prob = model_data['prob']
        auc = model_data['auc']
        label = model_data['label']

        results[model_name] = {'prob': prob, 'auc': auc, 'label': label}
        print(f"    {model_name}: AUC = {auc:.4f}")

    return results, y_test


def main():
    print("=" * 60)
    print("Generate ROC Plots - Load Pre-trained Models (5 models)")
    print("=" * 60)

    # 临时输出目录，测试用
    import tempfile
    output_dir = tempfile.mkdtemp(prefix="roc_test_")
    print(f"\nOutput to temp dir: {output_dir}")
    
    # 正式输出目录（测试通过后取消注释）
    # output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "MainFigures")
    # os.makedirs(output_dir, exist_ok=True)

    for model_type in ['CST', 'TNF']:
        print(f"\n{model_type} Model:")
        results, y_test = load_models(model_type)

        output_path = os.path.join(output_dir, f"Fig5_{model_type}_roc.pdf")
        plot_roc_comparison(results, y_test, output_path)

    print("\n" + "=" * 60)
    print(f"Done! Generated ROC plots in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
