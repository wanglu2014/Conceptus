# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Paths - use relative paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUT_DIR = SCRIPT_DIR.parent / 'SupplementaryFigures'

xlsx = pd.ExcelFile(DATA_DIR / 'Supplementary_Table_Model_Performance_Comparison.xlsx')
df_auc = pd.read_excel(xlsx, 'AUC_by_FeatureSet')

# Updated intervention mapping (removed chol492, separated Anti-TNF/EEN/PEN)
intervention_mapping = {
    'FMT': ['SRP135559_time.Donor', 'SRP135559_time.Baseline'],
    'Exercise': ['exerT2D_Onlyone', 'AELC16s_AEx.M0', 'AELCmeta_AEx.M0'],
    'Diet (EVOO/MCT)': ['MCTEOOV_tEVOO.1', 'MCTEOOV_tMCT.1'],
    'Diet (Other)': ['AELC16s_Diet.M0', 'AELCmeta_Diet'],
    'Anti-TNF': ['pleasespecif_antiTNF.1'],
    'EEN': ['pleasespecif_EEN.1'],
    'PEN': ['pleasespecif_PEN.1'],
    'AED': ['AELC16s_AED.M0', 'AELCmeta_AED.M0'],
    'Antibiotic': ['ERP013257_time.preABX'],
    'Other': ['diabetesMet_M.M0', 'ERP116682_time.Pre', 'LSSdel3_CD.1']
}

# Sample sizes from SRP057027 paper (PMC4633303)
# "90 children initiated therapy (52 anti-TNF; 22 EEN; 16 PEN)"
sample_sizes = {
    'FMT': 41,
    'Exercise': 542,
    'Diet (EVOO/MCT)': 1034,
    'Diet (Other)': 236,
    'Anti-TNF': 39,  # User confirmed (subset of SRP057027)
    'EEN': 22,       # SRP057027: 22 patients (Exclusive Enteral Nutrition)
    'PEN': 16,       # SRP057027: 16 patients (Partial Enteral Nutrition)
    'AED': 196,
    'Antibiotic': 7,
    'Other': 269
}

intervention_stats = []
for interv, cohorts in intervention_mapping.items():
    subset = df_auc[df_auc['cohort'].isin(cohorts)]
    if len(subset) > 0:
        intervention_stats.append({
            'intervention': interv,
            'cohorts': cohorts,
            'n_samples': sample_sizes.get(interv, 0),
            'OTU_mean': subset['OTU'].mean(),
            'Deg_mean': subset['Deg'].mean(),
            'Combo_mean': subset['OTU+Deg+index'].mean()
        })

df_stats = pd.DataFrame(intervention_stats).sort_values('Combo_mean', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
y = np.arange(len(df_stats))
width = 0.25

ax.barh(y - width, df_stats['OTU_mean'], width, label='OTU only', color='lightblue')
ax.barh(y, df_stats['Deg_mean'], width, label='Degree only', color='lightgreen')
ax.barh(y + width, df_stats['Combo_mean'], width, label='OTU+Deg+Index', color='coral')

ylabels = [f"{row['intervention']} (n={row['n_samples']})" for _, row in df_stats.iterrows()]

ax.set_xlabel('Mean AUC')
ax.set_ylabel('Intervention Type')
ax.set_yticks(y)
ax.set_yticklabels(ylabels)
ax.legend(loc='lower right')
ax.set_xlim(0, 1.15)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Model Performance by Intervention Type')

for i, combo in enumerate(df_stats['Combo_mean']):
    ax.text(combo + 0.02, i + width, f'{combo:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / 'Supplementary_Figure_6_AUC_by_Intervention.png', dpi=300, bbox_inches='tight')
print('SF6 PNG saved!')

try:
    plt.savefig(OUT_DIR / 'Supplementary_Figure_6_AUC_by_Intervention.pdf', bbox_inches='tight')
    print('SF6 PDF saved!')
except PermissionError:
    print('PDF permission denied - file may be open')

plt.close()

print('\nResults:')
print(df_stats[['intervention', 'n_samples', 'Combo_mean']].to_string())
