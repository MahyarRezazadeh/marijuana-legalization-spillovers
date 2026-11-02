# ============================================================
# Figure for the diff-in-discontinuities appendix check: the RD
# jump in overdose rate at the 50-mile threshold, estimated
# separately in each year 2020-2025, with the CCT-optimal
# bandwidth. Kieran Healy style, consistent with the rest of the
# project.
# ============================================================
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
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

df = pd.read_csv(os.path.join(TABLES, 'diffindisc_by_year.csv'))

fig, ax = plt.subplots(figsize=(7, 4.4))
cols = [ACCENT if p < 0.05 else GRAY for p in df['p']]
ax.errorbar(df['year'], df['coef'], yerr=1.96*df['se'], fmt='o-', color=INK, ecolor=GRAY,
            capsize=4, ms=0, elinewidth=1.4, lw=1, zorder=1)
ax.scatter(df['year'], df['coef'], c=cols, s=90, zorder=2)
ax.axhline(0, color='#bbb', lw=1, linestyle='--')
healy(ax, title='Figure A4. RD Jump at the 50-Mile Threshold, by Year (CCT-Optimal Bandwidth)',
      xlabel='Year', ylabel='RD estimate (overdose deaths per 100,000)')
ax.set_xticks(df['year'])
plt.tight_layout(rect=[0, 0.13, 1, 0.92])
fig.text(0.02, 0.02, 'Cross-sectional RD (levels, not first-differenced) at each year\'s snapshot; positive = far side (>=50mi) has\n'
                     'higher overdose rate than near side. Red = significant at 5%. The 2020 jump (before most legalization had\n'
                     'occurred) is itself significant, and the jump shrinks toward zero by 2025 rather than strengthening.',
          fontsize=7.2, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figA4_diffindisc_byyear.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figA4_diffindisc_byyear.png")
