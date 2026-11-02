# ============================================================
# Journal of Health Economics draft -- event-study figures for
# both designs (main distance-to-legal-county, and the dispensary
# actual-opening-date alternative), single script since both share
# the same plotting logic. Kieran Healy style, consistent with the
# rest of this project.
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
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

BINS = ['et_m3', 'et_m2', 'et_m1', 'et_0', 'et_p1', 'et_p2', 'et_p3']
xpos = np.array([-3, -2, -1, 0, 1, 2, 3])
labels = ['−3', '−2', '−1', '0\n(switch)', '+1', '+2', '+3']

def make_fig(results_file, out_name, title, note, small_n_bins=None):
    with open(os.path.join(DATA, results_file)) as f:
        res = json.load(f)
    adj = res['adjusted']
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    coefs, ses, xs = [], [], []
    for x, b in zip(xpos, BINS):
        if b not in adj:
            continue
        xs.append(x); coefs.append(adj[b]['coef']); ses.append(adj[b]['se'])
    color_list = [ACCENT if (small_n_bins is None or BINS[list(xpos).index(x)] not in small_n_bins) else '#c9a227' for x in xs]
    for x, c, se, col in zip(xs, coefs, ses, color_list):
        ax.errorbar(x, c, yerr=1.96 * se, fmt='o', color=col, capsize=4, ms=10, elinewidth=1.9)
    ax.axhline(0, color='#bbb', lw=1, linestyle='--')
    ax.axvline(0, color='#eee', lw=10, zorder=0)
    used_labels = [labels[list(xpos).index(x)] for x in xs]
    ax.set_xticks(xs); ax.set_xticklabels(used_labels, fontsize=9)
    healy(ax, title=title,
          xlabel='Years relative to switch (t=0: county crosses the 50-mile threshold)',
          ylabel='Coefficient (overdose deaths per 100,000, vs. far-away control counties)')
    plt.tight_layout(rect=[0, 0.15, 1, 0.90])
    fig.text(0.02, 0.02, note, fontsize=7.3, color=MUTED, style='italic')
    fig.savefig(os.path.join(FIG, out_name), dpi=170, facecolor='white')
    plt.close()
    print(f"Saved {out_name}")

make_fig(
    'eventstudy_main_results.json', 'fig1_eventstudy_main.png',
    'Figure 1. Event Study: Distance to Nearest Legal-Market County',
    'Treatment: switcher counties (n=436, crossing the 50-mile threshold in a given transition). Control: counties\n'
    'remaining >=50 miles from any legal-market county throughout ("far-away" counties, n=1,779). Baseline-(2020)-rate-\n'
    'adjusted specification; standard errors clustered by county. Switch-year effect (t=0): -2.48 deaths per 100,000\n'
    '(p=0.001); the effect does not persist -- point estimates return to positive by t=+1 and stay there through t=+3.'
)

make_fig(
    'eventstudy_dispensary_results.json', 'fig2_eventstudy_dispensary.png',
    "Figure 2. Event Study: Distance to Nearest Operating Dispensary (Alternative Measure)",
    'Treatment: switcher counties by distance to an actually operating dispensary (real license/opening dates, 9 of 24\n'
    'legal states geocoded; n=240). Control: far-away counties (n=2,038). Baseline-(2020)-rate-adjusted specification,\n'
    'clustered by county. Gold points (t=+2, t=+3) rest on only 6 and 2 transition-rows respectively -- too thin to\n'
    'interpret and shown only for completeness, not as evidence. Switch-year effect: -3.17 (p=0.008), consistent in sign\n'
    'and comparable in magnitude to the main design\'s -2.48.',
    small_n_bins={'et_p2', 'et_p3'}
)
