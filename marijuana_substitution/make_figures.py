import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

BLUE = '#1f77b4'; RED = '#d62728'; GRAY = '#7f7f7f'; ORANGE = '#ff7f0e'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors='#444444')
    if title:  ax.set_title(title, fontsize=9, loc='left', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#555555')
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color='#555555')

# ═══════════════════════════════════════════════════════════
# PART A FIGURES (switcher/control county panel)
# ═══════════════════════════════════════════════════════════
partA = pd.read_csv(os.path.join(DATA, 'partA_county_panel.csv'))

# Fig A0a: Treatment map -- counties colored by switcher/control status
fig, ax = plt.subplots(figsize=(8, 5))
ctrl = partA[partA['switcher'] == 0].dropna(subset=['lat', 'lon'])
sw = partA[partA['switcher'] == 1].dropna(subset=['lat', 'lon'])
ax.scatter(ctrl['lon'], ctrl['lat'], c=GRAY, s=6, alpha=0.35, linewidths=0, label='Control (>=100 mi)')
ax.scatter(sw['lon'], sw['lat'], c=RED, s=22, alpha=0.85, linewidths=0, label='Switcher (became <100 mi)')
ax.legend(fontsize=8, frameon=False, loc='lower right')
healy(ax, title='Figure A0a. Treatment Assignment by Distance to Border',
      xlabel='Longitude', ylabel='Latitude')
fig.text(0.02, -0.02, 'Switcher = county whose distance to the nearest recreational-marijuana '
         'state border fell below 100 mi between 2020 and 2023.', fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA0a_map.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig A0b: Distribution of the continuous treatment (miles distance) and its change
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
ax = axes[0]
for grp, col, ls, lab in [(0, GRAY, '-', 'Control'), (1, RED, '--', 'Switcher')]:
    s = partA[partA['switcher'] == grp]['dist_rec_2023_mi'].dropna()
    if len(s) > 5:
        from scipy.stats import gaussian_kde
        d = s.clip(0, s.quantile(0.98))
        kde = gaussian_kde(d, bw_method=0.35)
        xr = np.linspace(0, d.max(), 300)
        ax.plot(xr, kde(xr), color=col, linestyle=ls, lw=1.8, label=lab)
ax.axvline(100, color='#888', linestyle=':', lw=1, label='100-mile threshold')
ax.legend(fontsize=7, frameon=False)
healy(ax, title='(a) 2023 distance to nearest border', xlabel='Miles', ylabel='Density')

ax = axes[1]
for grp, col, ls, lab in [(0, GRAY, '-', 'Control'), (1, RED, '--', 'Switcher')]:
    s = partA[partA['switcher'] == grp]['d_dist_mi'].dropna()
    if len(s) > 5:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s, bw_method=0.35)
        xr = np.linspace(s.quantile(0.01), s.quantile(0.99), 300)
        ax.plot(xr, kde(xr), color=col, linestyle=ls, lw=1.8, label=lab)
ax.axvline(0, color='#888', linestyle=':', lw=1)
ax.legend(fontsize=7, frameon=False)
healy(ax, title='(b) Change in distance, 2020-2023', xlabel='Miles (negative = became closer)', ylabel='Density')

fig.suptitle('Figure A0b. The Continuous Treatment: Miles to Nearest Recreational Border',
             fontsize=9, x=0.02, ha='left', y=1.03)
fig.text(0.02, -0.02, 'd_dist_mi = dist_2023_mi - dist_2020_mi is the primary continuous treatment used in Table A2.',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA0b_distance_dist.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Treatment-design figures saved.")

# Fig A1: MJ vs alcohol trend, switcher vs control
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, base, label in [(axes[0], 'mj', 'Marijuana'), (axes[1], 'alc', 'Alcohol')]:
    tr = partA.groupby('switcher')[[f'{base}_rate20', f'{base}_rate23']].mean()
    for sw, col, ls in [(0, GRAY, '-'), (1, RED, '--')]:
        lab = 'Control' if sw == 0 else 'Switcher'
        ax.plot([2020, 2023], [tr.loc[sw, f'{base}_rate20'], tr.loc[sw, f'{base}_rate23']],
                color=col, linestyle=ls, marker='o', ms=7, lw=2.2, label=lab)
    ax.set_xticks([2020, 2023])
    ax.legend(fontsize=7, frameon=False)
    healy(ax, title=f'{label} arrest rate', ylabel='Arrests per 100,000')
fig.suptitle('Figure A1. Marijuana vs. Alcohol Arrest Rate Trends: Switchers vs. Controls',
             fontsize=9, x=0.02, ha='left', y=1.03)
fig.text(0.02, -0.02, 'Codes 153+158 (marijuana); 270+280 (alcohol). Never-legalizing states, main sample.',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA1_trends.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig A2: DiD coefficient plot (switcher, both outcomes) with 95% CI
import json
with open(os.path.join(DATA, 'partA_results.json')) as f:
    res = json.load(f)
fig, ax = plt.subplots(figsize=(6, 4))
labels, coefs, los, his, cols = [], [], [], [], []
for key, lab, col in [('mj_switcher', 'Marijuana\n(switcher)', BLUE),
                       ('alc_switcher', 'Alcohol\n(switcher)', RED)]:
    b, se, p, n = res[key]
    labels.append(lab); coefs.append(b); los.append(b - 1.96*se); his.append(b + 1.96*se); cols.append(col)
y = np.arange(len(labels))
for i in range(len(labels)):
    ax.errorbar(coefs[i], y[i], xerr=[[coefs[i]-los[i]], [his[i]-coefs[i]]],
                fmt='o', color=cols[i], capsize=4, ms=8, elinewidth=1.6)
ax.axvline(0, color='#999', lw=1, linestyle='--')
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
healy(ax, title='Figure A2. Switcher Effect on Arrest Rates (95% CI)', xlabel='Change in arrests per 100,000')
fig.text(0.02, -0.03, 'SE clustered by state. First-difference (2020->2023).', fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA2_coefplot.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig A3: scatter of d_mj vs d_alc (substitution check)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(partA['d_mj'], partA['d_alc'], s=8, alpha=0.35, color=BLUE, linewidths=0)
corr = partA[['d_mj', 'd_alc']].corr().iloc[0, 1]
z = np.polyfit(partA['d_mj'].dropna(), partA.loc[partA['d_mj'].notna(), 'd_alc'], 1)
xr = np.linspace(partA['d_mj'].min(), partA['d_mj'].max(), 50)
ax.plot(xr, np.polyval(z, xr), color=RED, lw=1.8)
ax.axhline(0, color='#ccc', lw=0.8); ax.axvline(0, color='#ccc', lw=0.8)
healy(ax, title=f'Figure A3. County-Level Co-Movement (r = {corr:.2f})',
      xlabel='Change in marijuana arrest rate', ylabel='Change in alcohol arrest rate')
fig.text(0.02, -0.02, 'A negative slope would indicate substitution (counties with falling MJ arrests seeing rising alcohol arrests).',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA3_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig A4: continuous treatment (miles) vs. outcome change -- the regression itself
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, base, label, col in [(axes[0], 'd_mj', 'Marijuana', BLUE), (axes[1], 'd_alc', 'Alcohol', RED)]:
    x = partA['d_dist_mi']
    y = partA[base]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=7, alpha=0.3, color=col, linewidths=0)
    z = np.polyfit(x[ok], y[ok], 1)
    xr = np.linspace(x[ok].quantile(0.02), x[ok].quantile(0.98), 50)
    ax.plot(xr, np.polyval(z, xr), color='black', lw=1.8)
    ax.axvline(0, color='#ccc', lw=0.8); ax.axhline(0, color='#ccc', lw=0.8)
    healy(ax, title=f'{label}', xlabel='Change in distance to border (mi)',
          ylabel='Change in arrest rate per 100k')
fig.suptitle('Figure A4. The Continuous-Treatment Regression: Distance Change vs. Arrest-Rate Change',
             fontsize=9, x=0.02, ha='left', y=1.03)
fig.text(0.02, -0.02, 'Negative x = county became closer to a newly-legal border. Black line = OLS fit '
         '(county-level; reported coefficients use 100-mile units and cluster SE by state).',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figA4_treatment_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Part A figures saved.")

# ═══════════════════════════════════════════════════════════
# PART B FIGURES (legalizing states, event-study)
# ═══════════════════════════════════════════════════════════
event = pd.read_csv(os.path.join(DATA, 'partB_event_study.csv'))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, col, lab, color in [(axes[0], 'marijuana_mean', 'Marijuana', BLUE),
                             (axes[1], 'alcohol_mean', 'Alcohol', RED)]:
    ax.plot(event['event_year'], event[col], color=color, marker='o', ms=4, lw=1.8)
    ax.axvline(0, color='#999', lw=1, linestyle='--')
    healy(ax, title=f'{lab} arrest rate', xlabel='Years since own legalization', ylabel='Arrests per 100,000')
fig.suptitle("Figure B1. Event Study: Arrest Rates Around Each State's Own Legalization Date",
             fontsize=9, x=0.02, ha='left', y=1.03)
fig.text(0.02, -0.02, '21 legalizing states, pooled. Vertical line = year of own recreational legalization.',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figB1_eventstudy.png'), dpi=150, bbox_inches='tight')
plt.close()

# Fig B2: Colorado monthly trend with legalization date + 2021 NIBRS transition shaded
trends = pd.read_csv(os.path.join(DATA, 'state_trends_mj_alcohol.csv'))
co = trends[trends.state_abbr == 'CO'].copy()
co['t'] = pd.to_datetime(co['month'], format='%m-%Y')
co = co.sort_values('t')

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for offense, color in [('marijuana', BLUE), ('alcohol', RED)]:
    s = co[co.offense == offense]
    ax.plot(s['t'], s['rate_per_100k'], color=color, lw=1.3, label=offense.capitalize())
ax.axvline(pd.Timestamp('2014-01-01'), color=ORANGE, lw=1.5, linestyle='--', label='CO legalization (2014)')
ax.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-06-01'), color='#ffcccc', alpha=0.3,
           label='NIBRS transition (low coverage)')
ax.legend(fontsize=7, frameon=False)
healy(ax, title='Figure B2. Colorado Monthly Arrest Rates, 2005-2023', xlabel='', ylabel='Arrests per 100,000')
fig.text(0.02, -0.02, 'Shaded region = 2021-2022 mandatory NIBRS transition period (state coverage fell to ~54% in 2021).',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'figB2_colorado.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Part B figures saved.")
