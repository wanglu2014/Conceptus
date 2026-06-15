# -*- coding: utf-8 -*-
"""
Supplementary Figures Generation Script
For: Microbiome Intervention Prediction Paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Paths - use relative paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUT_DIR = SCRIPT_DIR.parent / 'SupplementaryFigures'

# =============================================================================
# Load Data
# =============================================================================
xlsx = pd.ExcelFile(DATA_DIR / 'Supplementary_Table_Model_Performance_Comparison.xlsx')
df_auc = pd.read_excel(xlsx, 'AUC_by_FeatureSet')
df_net = pd.read_excel(xlsx, 'Sample_Network_Attributes')

# Note: Antifragility Index data computed from network attributes
df_index = pd.DataFrame()  # Not used in current figures

COHORT_LABELS = {
    'AELC16s_AED.M0': 'Metabolic disease_Ex+Fib (16S)',
    'AELC16s_AEx.M0': 'Metabolic disease_Ex (16S)',
    'AELC16s_Diet.M0': 'Metabolic disease_Fib (16S)',
    'AELCmeta_AED.M0': 'Metabolic disease_Ex+Fib (MGS)',
    'AELCmeta_AEx.M0': 'Metabolic disease_Ex (MGS)',
    'AELCmeta_Diet': 'Metabolic disease_Fib (MGS)',
    'ERP013257_time.preABX': 'IBD_FMT-Base (ERP01)',
    'ERP116682_time.Pre': 'IBD_FMT-Base (ERP11)',
    'exerT2D_Onlyone': 'Metabolic disease_Exercise',
    'LSSdel3_CD.1': 'IBD_CD-Ctrl',
    'LSSdel3_UC.1': 'IBD_UC-Ctrl',
    'MCTEOOV_tEVOO.1': 'Metabolic disease_EVOO',
    'MCTEOOV_tMCT.1': 'Metabolic disease_MCT',
    'pleasespecif_antiTNF.1': 'IBD_Anti-TNF',
    'pleasespecif_EEN.1': 'IBD_EEN',
    'pleasespecif_PEN.1': 'IBD_PEN',
    'SRP135559_time.Baseline': 'IBD_FMT-Base (SRP13)',
    'SRP135559_time.Donor': 'IBD_FMT-Donor',
    'protect_5ASA.0.biopsy': 'IBD_5-ASA',
}

# =============================================================================
# SF5: Responder vs NonResponder Network Attributes Comparison (Network topology)
# =============================================================================
def generate_sf3():
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes = axes.flatten()

    metrics_net = ['mean_degree', 'diameter', 'edgenumber', 'nodenumber', 'mean_closeness']
    labels_net = ['Mean Degree', 'Diameter', 'Edge Number', 'Node Number', 'Mean Closeness']

    stats_results = []

    for i, (metric, label) in enumerate(zip(metrics_net, labels_net)):
        ax = axes[i]
        resp_data = df_net[df_net['response_label']=='Responder'][metric].dropna()
        nonresp_data = df_net[df_net['response_label']=='NonResponder'][metric].dropna()

        bp = ax.boxplot([resp_data, nonresp_data], positions=[1, 2], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('steelblue')

        stat, pval = stats.mannwhitneyu(resp_data, nonresp_data, alternative='two-sided')
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Responder\n(n={len(resp_data)})', f'NonResponder\n(n={len(nonresp_data)})'], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n(p={pval:.2e}, {sig})')
        
        stats_results.append({'metric': label, 'resp_median': resp_data.median(), 
                             'nonresp_median': nonresp_data.median(), 'pval': pval, 'sig': sig})
    
    # Antifragility Index (from new_index)
    ax = axes[5]
    if len(df_index) > 0:
        resp_idx = df_index[df_index['response_label']=='Responder']['new_index'].dropna()
        nonresp_idx = df_index[df_index['response_label']=='NonResponder']['new_index'].dropna()
        
        bp = ax.boxplot([resp_idx, nonresp_idx], positions=[1, 2], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('steelblue')
        
        stat, pval = stats.mannwhitneyu(resp_idx, nonresp_idx, alternative='two-sided')
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Responder\n(n={len(resp_idx)})', f'NonResponder\n(n={len(nonresp_idx)})'], fontsize=9)
        ax.set_ylabel('Antifragility Index')
        ax.set_title(f'Antifragility Index\n(p={pval:.2e}, {sig})')
        
        stats_results.append({'metric': 'Antifragility Index', 'resp_median': resp_idx.median(),
                             'nonresp_median': nonresp_idx.median(), 'pval': pval, 'sig': sig})
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Supplementary_Figure_5_Network_Comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'Supplementary_Figure_5_Network_Comparison.pdf', bbox_inches='tight')
    plt.close()
    
    return stats_results

# =============================================================================
# SF4: Network Metrics Correlation Heatmap
# =============================================================================
def generate_sf4():
    metrics = ['mean_degree', 'diameter', 'mean_closeness', 'edgenumber', 'nodenumber']
    corr_matrix = df_net[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation')
    
    labels = ['Mean Degree', 'Diameter', 'Mean Closeness', 'Edge Number', 'Node Number']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                   ha='center', va='center', color='black', fontsize=10)
    
    ax.set_title('Network Metrics Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Supplementary_Figure_4_Network_Correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'Supplementary_Figure_4_Network_Correlation.pdf', bbox_inches='tight')
    plt.close()
    
    return corr_matrix

# =============================================================================
# SF2: Cohort Sample Distribution
# =============================================================================
def generate_sf6():
    labeled_df = df_net[df_net['response_label'].notna()].copy()
    cohort_stats = labeled_df.groupby('cohort').agg({
        'samples': 'count',
        'response_label': lambda x: (x == 'Responder').sum()
    }).rename(columns={'samples': 'total', 'response_label': 'responders'})

    cohort_stats['nonresponders'] = cohort_stats['total'] - cohort_stats['responders']
    cohort_stats['response_rate'] = cohort_stats['responders'] / cohort_stats['total'] * 100
    cohort_stats = cohort_stats.sort_values('total', ascending=True)
    cohort_labels = [COHORT_LABELS.get(cohort, cohort) for cohort in cohort_stats.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    y = range(len(cohort_stats))
    ax1.barh(y, cohort_stats['responders'], label='Responder', color='coral')
    ax1.barh(y, cohort_stats['nonresponders'], left=cohort_stats['responders'], 
             label='NonResponder', color='steelblue')
    ax1.set_yticks(y)
    ax1.set_yticklabels(cohort_labels)
    ax1.set_xlabel('Sample Count')
    ax1.set_title('Sample Distribution by Cohort')
    ax1.legend()
    
    for i, (total, resp) in enumerate(zip(cohort_stats['total'], cohort_stats['responders'])):
        ax1.text(total + 5, i, f'n={total}', va='center', fontsize=8)
    
    ax2.barh(y, cohort_stats['response_rate'], color='gray')
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(cohort_labels)
    ax2.set_xlabel('Response Rate (%)')
    ax2.set_title('Response Rate by Cohort')
    ax2.set_xlim(0, 100)
    
    for i, rate in enumerate(cohort_stats['response_rate']):
        ax2.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'Supplementary_Figure_2_Sample_Distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT_DIR / 'Supplementary_Figure_2_Sample_Distribution.pdf', bbox_inches='tight')
    plt.close()
    
    return cohort_stats

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('Generating SF5 (Network topology comparison)...')
    sf3_stats = generate_sf3()
    print('SF5 done!')
    
    print('Generating SF4...')
    sf4_corr = generate_sf4()
    print('SF4 done!')
    
    
    print('Generating SF2 (Cohort sample distribution)...')
    sf6_stats = generate_sf6()
    print('SF2 done!')
    
    print('\nAll supplementary figures generated successfully!')
