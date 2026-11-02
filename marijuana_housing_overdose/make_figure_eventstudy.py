# ============================================================
# Figure E: pooled event-study / discontinuity-around-cutoff
# view of state overdose mortality, aligned by months since each
# state's OWN recreational legalization date (event time = 0).
# Shows whether there's a visible jump right at the cutoff,
# distinct from the secular trend already visible in Figure D.
# Descriptive/exploratory only.
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
GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=10, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

df = pd.read_csv(os.path.join(DATA, 'state_overdose_trend.csv'), parse_dates=['date', 'legal_rec'])
df['event_month'] = ((df['date'].dt.year - df['legal_rec'].dt.year) * 12 +
                      (df['date'].dt.month - df['legal_rec'].dt.month))

WINDOW = 36
ev = df[df['event_month'].between(-WINDOW, WINDOW)].copy()

# ── Normalize each state to its own event_month=-1 level (index = 100), so the pooled
#    average isolates the SHAPE around the cutoff, not just the highest-rate states dominating ──
base = ev[ev['event_month'].between(-3, -1)].groupby('state')['rate_per_100k'].mean().rename('base')
ev = ev.merge(base, on='state')
ev['index'] = ev['rate_per_100k'] / ev['base'] * 100

agg = ev.groupby('event_month')['index'].agg(['mean', 'std', 'count']).reset_index()
agg['se'] = agg['std'] / np.sqrt(agg['count'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) raw pooled level (deaths per 100k), unindexed -- shows secular trend dominates
ax = axes[0]
raw_agg = ev.groupby('event_month')['rate_per_100k'].mean().reset_index()
ax.plot(raw_agg['event_month'], raw_agg['rate_per_100k'], color=GRAY, lw=1.8)
ax.axvline(0, color=ACCENT, lw=1.3, linestyle='--')
healy(ax, title='(a) Pooled level (21 states, unindexed)',
      xlabel='Months since recreational legalization', ylabel='Mean overdose rate per 100,000')

# (b) indexed to each state's own pre-period, isolating the shape at the cutoff
ax = axes[1]
ax.plot(agg['event_month'], agg['mean'], color=INK, lw=1.8)
ax.fill_between(agg['event_month'], agg['mean']-1.96*agg['se'], agg['mean']+1.96*agg['se'],
                 color=INK, alpha=0.12, linewidth=0)
ax.axvline(0, color=ACCENT, lw=1.3, linestyle='--')
ax.axhline(100, color='#ccc', lw=0.8)
healy(ax, title='(b) Indexed to each state\'s own pre-period (=100)',
      xlabel='Months since recreational legalization', ylabel='Index (event_month -3 to -1 = 100)')

fig.suptitle('Figure E. Overdose Mortality Around the Recreational-Legalization Cutoff, Pooled Across 21 States',
             fontsize=11.5, x=0.02, ha='left', y=1.0, color=INK)
caption = ('Panel (a) pools raw rates and mostly reflects the shared national fentanyl-driven trend, not a '
           'legalization effect. Panel (b) indexes each state to its own 3-month pre-legalization average before '
           'pooling, isolating any discontinuity right at the cutoff from the secular trend; shaded band is\n'
           'a naive +/-1.96 SE across states (not clustered/state-weighted -- descriptive only). No visible '
           'discontinuity at month 0 in either panel.')
plt.tight_layout(rect=[0, 0.11, 1, 0.92])
fig.text(0.02, 0.02, caption, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figE_eventstudy_overdose.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figE_eventstudy_overdose.png")
