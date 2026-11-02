import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; MED_COLOR = '#4c7fa6'; GRAY = '#8a8a8a'

def healy_mini(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#ececec', linewidth=0.5); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=6.5, colors=INK)

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=10, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

df = pd.read_csv(os.path.join(DATA, 'state_overdose_trend_new.csv'), parse_dates=['date', 'effective_date', 'retail_date'])
df = df.sort_values('date')

# ============================================================
# Fig D: state trends, 24 states, effective + retail dates
# ============================================================
states = sorted(df['state'].unique())
n = len(states); ncols = 5; nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.6*nrows), sharex=True)
axes = axes.flatten()
for i, st in enumerate(states):
    ax = axes[i]
    sub = df[df['state'] == st]
    ax.plot(sub['date'], sub['rate_per_100k'], color=INK, lw=1.1)
    eff_date = sub['effective_date'].iloc[0]; ret_date = sub['retail_date'].iloc[0]
    if pd.notna(eff_date) and sub['date'].min() <= eff_date <= sub['date'].max():
        ax.axvline(eff_date, color=MED_COLOR, lw=1.1, linestyle=':', alpha=0.85)
    if pd.notna(ret_date) and sub['date'].min() <= ret_date <= sub['date'].max():
        ax.axvline(ret_date, color=ACCENT, lw=1.3, linestyle='--', alpha=0.9)
    ax.set_title(st, fontsize=9, loc='left', pad=3, color=INK, fontweight='bold')
    healy_mini(ax); ax.set_xlim(df['date'].min(), df['date'].max())
for j in range(n, len(axes)): axes[j].axis('off')
legend_elems = [
    Line2D([0], [0], color=INK, lw=1.4, label='Overdose death rate (12-mo. ending, per 100k)'),
    Line2D([0], [0], color=MED_COLOR, lw=1.4, linestyle=':', label='Recreational-use policy effective date'),
    Line2D([0], [0], color=ACCENT, lw=1.4, linestyle='--', label='First recreational retail sales date'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Figure D. State Overdose Mortality Trends, 2015-2026, with Legalization Dates', fontsize=13, x=0.02, ha='left', y=0.995, color=INK)
caption = ('CDC VSRR state-level monthly (12-month-ending) provisional overdose death counts / fixed 2020 Census '
           'population. All 24 states with a recreational-use policy as of the source table (effective date and\n'
           'first retail sales date both shown where they fall in-window); Virginia has no retail sales date '
           '(no general retail market established) so only the effective-date line appears.')
plt.tight_layout(rect=[0, 0.09, 1, 0.965])
fig.text(0.01, 0.005, caption, fontsize=7.5, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figD_state_overdose_trends_new.png'), dpi=160, facecolor='white')
plt.close()
print("Saved figD_state_overdose_trends_new.png")

# ============================================================
# Fig E: event study around retail-legalization cutoff
# ============================================================
df2 = df.dropna(subset=['retail_date']).copy()  # exclude VA (no retail date)
df2['event_month'] = ((df2['date'].dt.year - df2['retail_date'].dt.year) * 12 +
                       (df2['date'].dt.month - df2['retail_date'].dt.month))
WINDOW = 36
ev = df2[df2['event_month'].between(-WINDOW, WINDOW)].copy()
base = ev[ev['event_month'].between(-3, -1)].groupby('state')['rate_per_100k'].mean().rename('base')
ev = ev.merge(base, on='state')
ev['index'] = ev['rate_per_100k'] / ev['base'] * 100
agg = ev.groupby('event_month')['index'].agg(['mean', 'std', 'count']).reset_index()
agg['se'] = agg['std'] / np.sqrt(agg['count'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
raw_agg = ev.groupby('event_month')['rate_per_100k'].mean().reset_index()
ax.plot(raw_agg['event_month'], raw_agg['rate_per_100k'], color=GRAY, lw=1.8)
ax.axvline(0, color=ACCENT, lw=1.3, linestyle='--')
healy(ax, title='(a) Pooled level (23 states w/ retail date, unindexed)', xlabel='Months since first retail sales', ylabel='Mean overdose rate per 100,000')
ax = axes[1]
ax.plot(agg['event_month'], agg['mean'], color=INK, lw=1.8)
ax.fill_between(agg['event_month'], agg['mean']-1.96*agg['se'], agg['mean']+1.96*agg['se'], color=INK, alpha=0.12, linewidth=0)
ax.axvline(0, color=ACCENT, lw=1.3, linestyle='--'); ax.axhline(100, color='#ccc', lw=0.8)
healy(ax, title="(b) Indexed to each state's own pre-period (=100)", xlabel='Months since first retail sales', ylabel='Index (event_month -3 to -1 = 100)')
fig.suptitle('Figure E. Overdose Mortality Around the Retail-Legalization Cutoff, Pooled Across 23 States', fontsize=11.5, x=0.02, ha='left', y=1.0, color=INK)
caption = ('Panel (a) pools raw rates and mostly reflects the shared national fentanyl-driven trend. Panel (b) '
           'indexes each state to its own 3-month pre-retail-launch average before pooling, isolating any '
           'discontinuity right at the cutoff; shaded band is a naive +/-1.96 SE across states (descriptive '
           'only). No visible discontinuity at month 0. Virginia excluded (no retail sales date).')
plt.tight_layout(rect=[0, 0.11, 1, 0.92])
fig.text(0.02, 0.02, caption, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figE_eventstudy_overdose_new.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figE_eventstudy_overdose_new.png")
