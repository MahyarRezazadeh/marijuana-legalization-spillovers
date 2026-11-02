# ============================================================
# Figures for the overdose-only Economic Inquiry paper (main1).
# Kieran Healy-style: white ground, no chartjunk spines, light
# recessive gridlines, muted ink labels, restrained accent color
# used only where it carries meaning (the significant estimate).
# ============================================================
import os, json
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
ACCENT = '#c1502e'    # brick red -- the significant estimate
ACCENT2 = '#4c9a6a'   # sage green -- second series (alcohol) in supplementary
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
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

# ── Fig 2: overdose coefficient plot, both co-primary specs + robustness ──
with open(os.path.join(DATA, 'overdose_results.json')) as f:
    res = json.load(f)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
labels = ['Switcher\n(binary, <50mi)', 'Distance\n(per 50mi, continuous)', 'Log-distance\n(robustness)']
keys = ['switcher', 'dist_thresh', 'log_dist']
coefs, los, his, cols = [], [], [], []
for k in keys:
    b, se, p, n = res[k]
    coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se)
    cols.append(ACCENT if p < 0.05 else GRAY)
y = np.arange(len(labels))
for i in range(len(labels)):
    ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]],
                fmt='o', color=cols[i], capsize=4, ms=9, elinewidth=1.8)
ax.axvline(0, color='#bbb', lw=1, linestyle='--')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
healy(ax, title='Figure 2. Effect on Drug-Overdose Mortality, Both Treatment Specifications (95% CI)',
      xlabel='Change in overdose deaths per 100,000 (2020-2023)')
fig.text(0.02, -0.05, 'SE clustered by state. Colored marker = significant at 5%. N=928 non-suppressed counties (81 switchers, 847 controls).',
         fontsize=7, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig2_overdose_coefplot.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()

# ── Fig 3: overdose trend, switcher vs control ──
panel = pd.read_csv(os.path.join(DATA, 'overdose_county_panel.csv'))
avail = panel[panel['d_od'].notna()]
tr = avail.groupby('switcher')[['od_rate20', 'od_rate23']].mean()
fig, ax = plt.subplots(figsize=(5.5, 4.2))
for swv, col, ls, lab in [(0, GRAY, '-', 'Control'), (1, ACCENT, '-', 'Switcher')]:
    ax.plot([2020, 2023], [tr.loc[swv, 'od_rate20'], tr.loc[swv, 'od_rate23']],
            color=col, linestyle=ls, marker='o', ms=8, lw=2.2)
    ax.annotate(lab, (2023, tr.loc[swv, 'od_rate23']), xytext=(6, 0), textcoords='offset points',
                fontsize=9, color=col, va='center')
ax.set_xticks([2020, 2023])
ax.set_xlim(2019.7, 2023.6)
healy(ax, title='Figure 3. Drug-Overdose Death Rate: Switchers vs. Controls', ylabel='Deaths per 100,000 (12-month-ending)')
fig.text(0.02, -0.04, 'CDC VSRR provisional county-level data, non-suppressed counties only (N=928).', fontsize=7, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig3_overdose_trends.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()

# ── Fig 4: continuous-treatment scatter (the regression itself) ──
fig, ax = plt.subplots(figsize=(6, 4.5))
x, y = panel['d_dist_thresh'], panel['d_od']
ok = x.notna() & y.notna()
ax.scatter(x[ok], y[ok], s=12, alpha=0.4, color=GRAY, linewidths=0)
z = np.polyfit(x[ok], y[ok], 1)
xr = np.linspace(x[ok].quantile(0.02), x[ok].quantile(0.98), 50)
ax.plot(xr, np.polyval(z, xr), color=ACCENT, lw=2)
ax.axvline(0, color='#ddd', lw=0.8); ax.axhline(0, color='#ddd', lw=0.8)
healy(ax, title='Figure 4. The Continuous-Distance Regression: Distance Change vs. Overdose-Rate Change',
      xlabel='Change in distance to nearest legal border (50-mile units)', ylabel='Change in overdose rate per 100,000')
fig.text(0.02, -0.03, 'Negative x = county became closer to a newly-legal border.', fontsize=7, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig4_overdose_scatter.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()

# ── Fig 5: supplementary arrest coefficient plot (mj + alc, both specs) ──
supp = pd.read_csv(os.path.join(BASE, 'tables', 'tE2_arrest_supplementary.csv'))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, outcome, col in [(axes[0], 'Marijuana arrests', ACCENT2), (axes[1], 'Alcohol arrests', ACCENT)]:
    sub = supp[(supp['outcome'] == outcome) & (supp['spec'].isin(['switcher', 'dist_thresh', 'log_dist']))]
    sub = sub.set_index('spec').reindex(['switcher', 'dist_thresh', 'log_dist'])
    labels = ['Switcher\n(binary)', 'Distance\n(per 50mi)', 'Log-distance\n(robust.)']
    y = np.arange(len(labels))
    lo = sub['coef'] - 1.96*sub['se']
    hi = sub['coef'] + 1.96*sub['se']
    colors = [ACCENT if p < 0.05 else GRAY for p in sub['p']]
    for i in range(len(labels)):
        ax.errorbar(sub['coef'].iloc[i], y[i], xerr=[[sub['coef'].iloc[i]-lo.iloc[i]], [hi.iloc[i]-sub['coef'].iloc[i]]],
                     fmt='o', color=colors[i], capsize=4, ms=8, elinewidth=1.6)
    ax.axvline(0, color='#bbb', lw=1, linestyle='--')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    healy(ax, title=outcome, xlabel='Change in arrest rate per 100,000')
fig.suptitle('Figure 5. Supplementary: Marijuana and Alcohol Arrest Rates at the Same 50-Mile Threshold', fontsize=10, x=0.02, ha='left', y=1.05, color=INK)
fig.text(0.02, -0.05, 'SE clustered by state. Colored marker = significant at 5%. N=1,894 counties (157 switchers, 1,737 controls).',
         fontsize=7, color=MUTED, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig5_arrest_supp_coefplot.png'), dpi=170, bbox_inches='tight', facecolor='white')
plt.close()

print("All main1 figures saved.")
