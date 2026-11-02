# ============================================================
# Rebuilds Figure 2 using the CORRECTED, mean-reversion-adjusted
# coefficients (matching Table 1 and the medical-placebo
# subsection), replacing the earlier version of this figure which
# plotted the pre-correction, unadjusted numbers (-3.41/-3.45-ish)
# -- a stale-figure inconsistency caught while proofreading the
# compiled PDF against Table 1's now-correct -2.320 headline
# number.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

main_ctrl = pd.read_csv(os.path.join(DATA, 'main_with_controls.csv'))
row = main_ctrl[main_ctrl['spec'] == '+ baseline overdose level'].set_index('x')
rec = {
    'switcher': dict(coef=row.loc['switcher', 'coef'], se=row.loc['switcher', 'se'], p=row.loc['switcher', 'p']),
    'dist_mile': dict(coef=row.loc['d_dist_mi', 'coef'], se=row.loc['d_dist_mi', 'se'], p=row.loc['d_dist_mi', 'p']),
}
with open(os.path.join(DATA, 'medical_adjusted_results.json')) as f:
    med_raw = json.load(f)
med = {
    'switcher': med_raw['switcher'],
    'dist_mile': med_raw['d_dist_mi'],
}

fig, ax = plt.subplots(figsize=(8.5, 4.6))
labels = ['Recreational: switcher\n(binary, <50mi)', 'Recreational: distance\n(per 100mi, for scale)',
          'Medical (placebo): switcher\n(binary, <50mi)', 'Medical (placebo): distance\n(per 100mi, for scale)']
specs = [(rec['switcher'], 1), (rec['dist_mile'], 100), (med['switcher'], 1), (med['dist_mile'], 100)]
coefs, los, his, cols = [], [], [], []
for spec, scale in specs:
    b, se, p = spec['coef'] * scale, spec['se'] * scale, spec['p']
    coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se)
    cols.append(ACCENT if p < 0.05 else GRAY)
y = np.arange(len(labels))
for i in range(len(labels)):
    ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]], fmt='o', color=cols[i], capsize=5, ms=10, elinewidth=2)
ax.axvline(0, color='#bbb', lw=1, linestyle='--')
ax.axhline(1.5, color='#ddd', lw=0.8, linestyle=':')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.8); ax.invert_yaxis()
healy(ax, title='Figure 2. Effect on Drug-Overdose Mortality, Preferred Specification (95% CI)',
      xlabel='Change in overdose deaths per 100,000')
plt.tight_layout(rect=[0, 0.18, 1, 0.94])
fig.text(0.02, 0.02, 'SE clustered by state; transition FE + baseline-overdose-rate-level control. Colored marker = significant at 5%.\n'
                      'The distance rows are signed as the effect of an INCREASE in distance, so a positive distance coefficient and a\n'
                      'negative switcher coefficient point the same direction. Medical row uses medical-access dates as a placebo check.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig2_overdose_coefplot_multiperiod.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig2_overdose_coefplot_multiperiod.png (corrected, mean-reversion-adjusted numbers)")
print(f"rec switcher: {rec['switcher']}")
print(f"rec dist: {rec['dist_mile']}")
print(f"med switcher: {med['switcher']}")
print(f"med dist: {med['dist_mile']}")
