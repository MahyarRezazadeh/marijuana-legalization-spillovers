# ============================================================
# Journal of Health Economics draft -- Callaway-Sant'Anna (2021)
# group-time ATT comparison, at the paper's own primary threshold
# (50 miles), using the county-clustered dynamic ATT already
# computed in cs_result['cs_nocov_county'] / ['cs_cov_county'] from
# twfe_cs_result_t50.json (built earlier in this project on an
# identical treatment/control/threshold definition, just under
# CS's own balanced-panel requirement -- see main.tex Section 3.2
# for the sample-construction caveat). Same Healy visual style as
# Figures 1-3.
# ============================================================
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'; MUTED = '#8a8a8a'; ACCENT = '#c1502e'; BLUE = '#3f6fa8'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white'); ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6); ax.xaxis.grid(False)
    ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8, colors=INK)
    if title:  ax.set_title(title, fontsize=9, loc='left', pad=5, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=MUTED)

with open(os.path.join(DATA, 'twfe_cs_result_t50.json')) as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

for ax, key, color, title in [
    (axes[0], 'cs_nocov_county', BLUE, "(a) No covariates"),
    (axes[1], 'cs_cov_county', ACCENT, "(b) + baseline-rate covariate"),
]:
    dyn = res[key]['dynamic']
    xs = [d['event_time'] for d in dyn]
    coefs = [d['att'] for d in dyn]
    ses = [d['se'] for d in dyn]
    ax.errorbar(xs, coefs, yerr=[1.96 * se for se in ses], fmt='o', color=color, capsize=4, ms=8, elinewidth=1.6)
    ax.axhline(0, color='#bbb', lw=1, linestyle='--')
    ax.axvline(0, color='#eee', lw=8, zorder=0)
    ax.set_xticks(xs)
    healy(ax, title=title, xlabel='Event time (years since crossing 50 miles)', ylabel='ATT (overdose deaths per 100,000)')

plt.tight_layout(rect=[0, 0.1, 1, 0.9])
fig.suptitle("Figure 4. Callaway–Sant'Anna (2021) Group-Time ATT, Event-Study Form", fontsize=11, x=0.02, ha='left', y=0.98)
fig.text(0.02, 0.02,
          'Group-time ATT estimator applied to the identical treatment/control/threshold definition as Table 1, under\n'
          "Callaway-Sant'Anna's own balanced-panel requirement (counties with complete, non-suppressed data in all six\n"
          'years; this shrinks the usable switching cohorts to three annual groups, 2022/2023/2024, coarser than the\n'
          'per-transition cohorts in the main design). Standard errors clustered by county. See Section 3.2 for discussion.',
          fontsize=7.3, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'fig4_cs_comparison.png'), dpi=170, facecolor='white')
plt.close()
print("Saved fig4_cs_comparison.png")
