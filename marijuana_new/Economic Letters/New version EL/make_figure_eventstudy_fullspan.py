# ============================================================
# Figure for the full-time-span event study (analyze_eventstudy_
# fullspan.py) -- dynamically sized to whatever event-time range
# the data actually supports (turned out to be -3..+3, identical
# to the earlier fixed-window version -- see script docstring).
# ============================================================
import os, json
import numpy as np
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

with open(os.path.join(DATA, 'eventstudy_fullspan_results.json')) as f:
    res = json.load(f)

event_times = res['event_times']
def bin_label(et):
    if et < 0: return f'et_m{abs(et)}'
    return f'et_p{et}' if et > 0 else 'et_0'
BINS = [bin_label(e) for e in event_times]
xpos = np.array(event_times)
labels = [f'{e}\n(switch)' if e == 0 else str(e) for e in event_times]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for offset, (key, color, name) in enumerate([('raw', GRAY, 'Raw (no controls)'), ('adjusted', ACCENT, 'Mean-reversion-adjusted (preferred)')]):
    dx = -0.08 if offset == 0 else 0.08
    coefs = [res[key][b]['coef'] for b in BINS]
    ses = [res[key][b]['se'] for b in BINS]
    ax.errorbar(xpos + dx, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=color, capsize=4, ms=9,
                elinewidth=1.8, label=name)

ax.axhline(0, color='#bbb', lw=1, linestyle='--')
ax.axvline(0, color='#eee', lw=10, zorder=0)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=8.5, frameon=False, loc='upper right')
healy(ax, title=f'Event Study, Full Panel Span (t={event_times[0]} to t={event_times[-1]}): Treatment vs. Control Counties',
      xlabel='Years relative to switch (t=0: county crosses the 50-mile legal-market threshold)',
      ylabel='Coefficient (overdose deaths per 100,000, vs. never-switcher counties)')
plt.tight_layout(rect=[0, 0.17, 1, 0.94])
fig.text(0.02, 0.02,
          f'Window set to the full span the 2020-2025 panel actually supports (start = earliest transition, end = latest\n'
          f'transition), with NO endpoint pooling -- every event time gets its own dummy. The true observed span turned out to\n'
          f'be t={event_times[0]} to t={event_times[-1]}, identical to the earlier fixed +/-3 window: no county with non-suppressed overdose data on both\n'
          f'sides of a transition has a switch further than 3 years from either end of the panel, so the two designs coincide\n'
          f'exactly (coefficients are numerically identical). Reference/control group: never-switcher counties (n=1,779).\n'
          f'Standard errors clustered by county.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig8_eventstudy_fullspan.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig8_eventstudy_fullspan.png")
