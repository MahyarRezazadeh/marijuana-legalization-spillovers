# ============================================================
# Figure for the age-cohort heterogeneity analysis (Part C):
# primary continuous-distance coefficient, by age bin, for
# marijuana and alcohol, with 95% CIs.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

BLUE = '#1f77b4'; RED = '#d62728'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors='#444444')
    if title:  ax.set_title(title, fontsize=9, loc='left', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#555555')
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color='#555555')

res = pd.read_csv(os.path.join(DATA, '..', 'tables', 'tC1_age_regressions.csv'))
AGE_ORDER = ['<21', '21-30', '31-45', '46+']
res['age_bin'] = pd.Categorical(res['age_bin'], categories=AGE_ORDER, ordered=True)

sub = res[res['spec'] == 'dist_100mi'].sort_values('age_bin')

fig, ax = plt.subplots(figsize=(7, 4.5))
y_base = np.arange(len(AGE_ORDER))
offset = 0.12
for outcome, col, dy in [('Marijuana', BLUE, offset), ('Alcohol', RED, -offset)]:
    s = sub[sub['outcome'] == outcome].set_index('age_bin').reindex(AGE_ORDER)
    y = y_base + dy
    lo = s['coef'] - 1.96 * s['se']
    hi = s['coef'] + 1.96 * s['se']
    ax.errorbar(s['coef'], y, xerr=[s['coef'] - lo, hi - s['coef']],
                fmt='o', color=col, capsize=4, ms=7, elinewidth=1.6, label=outcome)

ax.axvline(0, color='#999', lw=1, linestyle='--')
ax.set_yticks(y_base)
ax.set_yticklabels(AGE_ORDER, fontsize=9)
ax.invert_yaxis()
ax.legend(fontsize=8, frameon=False, loc='best')
healy(ax, title='Figure C1. Distance-to-Border Coefficient by Age of Arrestee (95% CI)',
      xlabel='Change in arrest rate per 100,000 per 100-mile change in distance')
fig.text(0.02, -0.03,
         'NIBRS Arrestee-Level microdata (2020, 2023), same switcher/control county sample and continuous-distance '
         'treatment as Design A. SE clustered by state. See Section 3.4 for the NIBRS-vs-SRS scale caveat.',
         fontsize=6, color='#777', style='italic', wrap=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figC1_age_coefplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figC1_age_coefplot.png")
