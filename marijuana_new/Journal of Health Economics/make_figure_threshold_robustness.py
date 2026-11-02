# ============================================================
# Journal of Health Economics draft -- robustness figure: full
# event-study curves at 75-mile and 100-mile thresholds (50-mile is
# the primary design, already Figures 1-2), 2x2 grid: rows =
# threshold (75, 100), columns = design (main, dispensary).
# ============================================================
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=7.5, colors=INK)
    if title:  ax.set_title(title, fontsize=8.5, loc='left', pad=5, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5, color=MUTED)

BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
xpos = np.array([-3, -2, -1, 0, 1, 2, 3])
labels = ['-3', '-2', '-1', '0', '+1', '+2', '+3']

panels = [
    ('main', 75, 'Main design, 75-mile threshold'),
    ('dispensary', 75, 'Dispensary design, 75-mile threshold'),
    ('main', 100, 'Main design, 100-mile threshold'),
    ('dispensary', 100, 'Dispensary design, 100-mile threshold'),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, (design, thresh, title) in zip(axes.flat, panels):
    with open(os.path.join(DATA, f'eventstudy_{design}_t{thresh}_results.json')) as f:
        res = json.load(f)
    adj = res['adjusted']
    xs = [x for x, b in zip(xpos, BINS) if b in adj]
    coefs = [adj[b]['coef'] for b in BINS if b in adj]
    ses = [adj[b]['se'] for b in BINS if b in adj]
    ax.errorbar(xs, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=ACCENT, capsize=3, ms=7, elinewidth=1.5)
    ax.axhline(0, color='#bbb', lw=1, linestyle='--')
    ax.axvline(0, color='#eee', lw=8, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels([labels[list(xpos).index(x)] for x in xs], fontsize=8)
    healy(ax, title=f"{title} (n switchers={res['n_switchers']})",
          xlabel='Years relative to switch', ylabel='Coef. (per 100,000)')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.suptitle('Figure 3. Event-Study Robustness: 75-Mile and 100-Mile Thresholds', fontsize=11, x=0.02, ha='left', y=0.99)
fig.text(0.02, 0.01,
          'Same specification as Figures 1-2 (baseline-2020-rate-adjusted, county-clustered SEs, far-away never-switcher\n'
          'counties as control), rerun at wider switcher thresholds. The primary 50-mile results are in Figures 1-2 and Table 4.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig3_eventstudy_threshold_robustness.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig3_eventstudy_threshold_robustness.png")
