# ============================================================
# Figure D: small-multiples of state-level overdose mortality
# trends (monthly, 12-month-ending, 2015-2026) for the 21
# recreational-legal states, with vertical lines marking each
# state's medical and recreational marijuana legalization dates.
# Descriptive/exploratory only -- not part of the regression
# design, not yet added to the main text.
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

INK = '#3f3f3f'
MUTED = '#8a8a8a'
ACCENT = '#c1502e'
MED_COLOR = '#4c7fa6'

def healy_mini(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#ececec', linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=6.5, colors=INK)

df = pd.read_csv(os.path.join(DATA, 'state_overdose_trend.csv'), parse_dates=['date'])
df = df.sort_values('date')

states = sorted(df['state'].unique())
n = len(states)
ncols = 5
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.6*nrows), sharex=True)
axes = axes.flatten()

for i, st in enumerate(states):
    ax = axes[i]
    sub = df[df['state'] == st]
    ax.plot(sub['date'], sub['rate_per_100k'], color=INK, lw=1.1)

    med_date = pd.to_datetime(sub['legal_med'].iloc[0])
    rec_date = pd.to_datetime(sub['legal_rec'].iloc[0])
    if sub['date'].min() <= med_date <= sub['date'].max():
        ax.axvline(med_date, color=MED_COLOR, lw=1.1, linestyle=':', alpha=0.85)
    ax.axvline(rec_date, color=ACCENT, lw=1.3, linestyle='--', alpha=0.9)

    ax.set_title(st, fontsize=9, loc='left', pad=3, color=INK, fontweight='bold')
    healy_mini(ax)
    ax.tick_params(axis='x', rotation=0)
    ax.set_xlim(df['date'].min(), df['date'].max())

for j in range(n, len(axes)):
    axes[j].axis('off')

# shared legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], color=INK, lw=1.4, label='Overdose death rate (12-mo. ending, per 100k)'),
    Line2D([0], [0], color=MED_COLOR, lw=1.4, linestyle=':', label='Medical legalization date'),
    Line2D([0], [0], color=ACCENT, lw=1.4, linestyle='--', label='Recreational legalization date'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Figure D. State Overdose Mortality Trends, 2015-2026, with Legalization Dates',
             fontsize=13, x=0.02, ha='left', y=0.995, color=INK)
caption = ('CDC VSRR state-level monthly (12-month-ending) provisional drug overdose death counts, divided by 2020\n'
           'Census population (fixed denominator across the whole series -- a simplification for this descriptive\n'
           'trend chart, not a rate-per-year calculation). The 21 panels are the states that have legalized\n'
           'recreational marijuana as of 2023-2024; medical-legalization dates are general public-record dates,\n'
           'not freshly verified this session.')
plt.tight_layout(rect=[0, 0.09, 1, 0.965])
fig.text(0.01, 0.005, caption, fontsize=7.5, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figD_state_overdose_trends.png'), dpi=160, facecolor='white')
plt.close()
print("Saved figD_state_overdose_trends.png")
