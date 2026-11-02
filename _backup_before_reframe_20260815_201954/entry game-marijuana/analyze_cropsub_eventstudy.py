# ============================================================
# Event study for crop substitution -- the weakest-powered of the four
# outcomes in this design, and worth being explicit about why: unlike
# revenue (2 states, stackable on a common market-age clock) or land
# value (21 states), crop substitution has only ONE state (Oregon,
# the only state with usable cultivation/border data), so there is no
# cross-state stacking to build a market-age clock from -- there is
# only Oregon's own single time series. NASS gives exactly 3 waves
# (2012, 2017, 2022), which we place relative to Oregon's own 2015
# retail launch (event_year = -3, +2, +7). With only 3 border counties
# (Baker, Malheur, Wallowa) this is necessarily a thin, low-powered
# event study -- reported for completeness and comparability with the
# other three outcomes, not because it is well-powered.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')

df = pd.read_csv(os.path.join(TABLES, 'crop_substitution_border_panel.csv'), dtype={'fips5': str})

# long panel: one row per county-year
long = pd.concat([
    df.assign(year=2012, cropland=df['cropland_2012']),
    df.assign(year=2017, cropland=df['cropland_2017']),
    df.assign(year=2022, cropland=df['cropland_2022']),
], ignore_index=True)
long['log_cropland'] = np.log(long['cropland'])
OR_LAUNCH = 2015
long['event_year'] = long['year'] - OR_LAUNCH  # -3, +2, +7
long['g'] = pd.Categorical(long['fips5']).codes

print(f"Panel: {len(long)} county-year obs, {long['fips5'].nunique()} counties, "
      f"event years {sorted(long['event_year'].unique())}")
print(f"Border counties: {df.loc[df['border_70mi']==1,'county_key'].tolist()}")

long['ey'] = pd.Categorical(long['event_year'])
formula = 'log_cropland ~ log_population + C(ey) + border_70mi:C(ey)'
m = smf.ols(formula, data=long).fit(cov_type='HC1')
mc = smf.ols(formula, data=long).fit(cov_type='cluster', cov_kwds={'groups': long['g']})

print("\n" + "=" * 70)
print("EVENT STUDY -- border_70mi x event-year interaction, outcome: log(cropland)")
print("=" * 70)
rows = []
for ey in sorted(long['event_year'].unique()):
    term = f'border_70mi:C(ey)[{ey}]'
    if term not in m.params.index:
        continue
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    se_c, p_c = mc.bse[term], mc.pvalues[term]
    print(f"event year {ey:+d} (year {ey+OR_LAUNCH})   b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
          f"| se(county-cl)={se_c:.4f} p={p_c:.4f} n_border=3")
    rows.append(dict(event_year=ey, calendar_year=ey + OR_LAUNCH, coef=b, se_hc1=se, p_hc1=p,
                      se_county_cl=se_c, p_county_cl=p_c))
pd.DataFrame(rows).to_csv(os.path.join(TABLES, 'cropsub_eventstudy_results.csv'), index=False)
print(f"\nR2={m.rsquared:.3f} n={int(m.nobs)}  (NOTE: only 3 border counties -- treat as illustrative, not well-powered)")

# ---------- Coefficient plot ----------
res = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(-0.5, color='gray', linewidth=0.8, linestyle='--')
ax.errorbar(res['event_year'], res['coef'], yerr=1.96 * res['se_county_cl'], fmt='o', color='black',
            ecolor='gray', capsize=4, markersize=7)
ax.set_xticks(res['event_year'])
ax.set_xticklabels([f"{y} ({y+OR_LAUNCH})" for y in res['event_year']])
ax.set_xlabel("Years relative to Oregon's 2015 retail launch (calendar year)")
ax.set_ylabel('Border $\\times$ event-time coefficient')
ax.set_title("Crop substitution event study (Oregon only, 3 border counties)\n"
             "Illustrative, not well-powered -- see Section on crop substitution", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'fig9_cropsub_eventstudy.png'), dpi=200)
plt.close(fig)

# ---------- Trends-separately plot (same style as the other three outcomes) ----------
long['y_adj'] = smf.ols('log_cropland ~ log_population', data=long).fit().resid
summary = long.groupby(['event_year', 'border_70mi']).agg(
    y_mean=('y_adj', 'mean'), y_se=('y_adj', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan),
    n=('y_adj', 'size')
).reset_index()

fig2, ax2 = plt.subplots(figsize=(7, 4.2))
for bval, label, color, marker in [(0, 'Interior (control, n=33)', 'steelblue', 'o'),
                                     (1, 'Border ($\\leq$70mi, n=3)', 'firebrick', 's')]:
    sub = summary[summary['border_70mi'] == bval].sort_values('event_year')
    ax2.errorbar(sub['event_year'], sub['y_mean'], yerr=1.96 * sub['y_se'], fmt=f'{marker}-', color=color,
                 ecolor=color, alpha=0.85, capsize=4, markersize=6, linewidth=1.3, label=label)
ax2.axvline(-0.5, color='gray', linewidth=0.8, linestyle='--')
ax2.set_xticks(sorted(summary['event_year'].unique()))
ax2.set_xticklabels([f"{y} ({y+OR_LAUNCH})" for y in sorted(summary['event_year'].unique())])
ax2.set_xlabel("Years relative to Oregon's 2015 retail launch (calendar year)")
ax2.set_ylabel('log(harvested cropland), adj. for log(pop)')
ax2.set_title("Crop substitution: border vs. interior trends, separately (Oregon only)\n"
              "Only 3 border counties -- illustrative, not well-powered", fontsize=10)
ax2.legend(fontsize=9, loc='best')
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES, 'fig10_cropsub_trends.png'), dpi=200)
plt.close(fig2)
print("\nSaved fig9_cropsub_eventstudy.png and fig10_cropsub_trends.png")
