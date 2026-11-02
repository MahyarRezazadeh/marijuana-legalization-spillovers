# ============================================================
# Centerpiece figure for the Applied Economics reframed draft:
# the overdose estimate across three levels of design rigor
# (binary switcher -> continuous distance -> spatial RDD /
# diff-in-disc), showing the significant effect does not survive
# as identification gets more demanding. Kieran Healy style.
# ============================================================
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'
MUTED = '#8a8a8a'
ACCENT = '#c1502e'
GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, colors=INK)
    if title:  ax.set_title(title, fontsize=10.5, loc='left', pad=8, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=MUTED)

# (label, coef, se, p, n)
specs = [
    ('Binary switcher\n(<50mi threshold)', -3.546, 1.632, 0.030, 928),
    ('Continuous distance\n(per 50mi)', -0.554, 0.340, 0.103, 928),
    ('Spatial RDD / diff-in-disc\n(border-line running variable,\nCCT-optimal bandwidth)', 3.82, 12.175, 0.754, None),
]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(specs))
for i, (label, coef, se, p, n) in enumerate(specs):
    lo, hi = coef - 1.96*se, coef + 1.96*se
    col = ACCENT if p < 0.05 else GRAY
    ax.errorbar(coef, y[i], xerr=[[coef-lo], [hi-coef]], fmt='o', color=col,
                capsize=5, ms=10, elinewidth=2)
    ax.annotate(f'p={p:.3f}', (hi, y[i]), xytext=(8, 0), textcoords='offset points',
                fontsize=8.5, color=col, va='center')

ax.axvline(0, color='#bbb', lw=1, linestyle='--')
ax.set_yticks(y)
ax.set_yticklabels([s[0] for s in specs], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlim(-24, 30)
healy(ax, title='Figure 1. The Overdose Estimate Across Three Levels of Design Rigor',
      xlabel='Effect on overdose deaths per 100,000')
fig.text(0.02, -0.06,
         'Each row uses progressively more of the border geometry and progressively less of the sample far from it.\n'
         'Colored marker = significant at 5%. Confidence intervals widen and the point estimate loses its sign\n'
         'consistency as the design becomes more locally identified.',
         fontsize=7.5, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig1_fragility_summary.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig1_fragility_summary.png")
