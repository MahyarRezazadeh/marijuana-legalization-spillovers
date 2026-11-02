# ============================================================
# Replication check: apply this project's own revenue-per-store
# construction to the exact Hansen, Miller & Weber (2020) setting --
# WA border-with-OR counties around OR's 2015-10-01 retail opening --
# using WA's annual (fiscal-year, Jul-Jun) county revenue data. This
# is coarser than HMW's monthly difference-in-discontinuities design
# (FY2016 straddles the Oct-2015 switch: ~3 pre-months, 9 post-months,
# so it is a genuine transition year, not a clean post period), but a
# similar-sign, similar-magnitude result from an independently built
# dataset is a useful validation of the switch-off mechanism before
# applying it to the new CO-NM setting.
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'wa_or_switch_panel.csv'), dtype={'fips5': str})
print("FY years available:", sorted(panel['fy_year'].unique()))
print("Only 1 pre-period year (FY2015) -- event-year coefficients before 0 cannot be estimated "
      "beyond that single year; this limits pre-trend testing relative to the CO-NM design.")

d = panel.copy()
d['g'] = pd.Categorical(d['fips5']).codes

# ---------- DiD, main spec: FY2015 (pre) vs FY2016+ (post, all years) ----------
formula = 'log_revenue_per_store ~ or_border_70mi*post_any + id_border_70mi + log_population + C(fy_year)'
m = smf.ols(formula, data=d).fit(cov_type='HC1')
mc = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
term = 'or_border_70mi:post_any'
pct = (np.exp(m.params[term]) - 1) * 100
print(f"\n--- DiD, post = FY2016+ (includes FY2016 transition year) ---")
print(f"n={int(m.nobs)}  b={m.params[term]:+.4f} ({pct:+.1f}%)  se(HC1)={m.bse[term]:.4f} p={m.pvalues[term]:.4f} "
      f"| se(county-cl)={mc.bse[term]:.4f} p={mc.pvalues[term]:.4f}")
row1 = dict(spec='post=FY2016+ (incl. transition yr)', coef=m.params[term], pct_change=pct,
            se_hc1=m.bse[term], p_hc1=m.pvalues[term], se_county_cl=mc.bse[term], p_county_cl=mc.pvalues[term],
            n=int(m.nobs))

# ---------- DiD, robustness: drop FY2016 transition year, post = FY2017+ ----------
d2 = d[d['fy_year'] != 2016].copy()
d2['g'] = pd.Categorical(d2['fips5']).codes
formula2 = 'log_revenue_per_store ~ or_border_70mi*post_clean + id_border_70mi + log_population + C(fy_year)'
m2 = smf.ols(formula2, data=d2).fit(cov_type='HC1')
mc2 = smf.ols(formula2, data=d2).fit(cov_type='cluster', cov_kwds={'groups': d2['g']})
term2 = 'or_border_70mi:post_clean'
pct2 = (np.exp(m2.params[term2]) - 1) * 100
print(f"\n--- DiD, FY2016 transition year dropped, post = FY2017+ ---")
print(f"n={int(m2.nobs)}  b={m2.params[term2]:+.4f} ({pct2:+.1f}%)  se(HC1)={m2.bse[term2]:.4f} p={m2.pvalues[term2]:.4f} "
      f"| se(county-cl)={mc2.bse[term2]:.4f} p={mc2.pvalues[term2]:.4f}")
row2 = dict(spec='FY2016 dropped, post=FY2017+', coef=m2.params[term2], pct_change=pct2,
            se_hc1=m2.bse[term2], p_hc1=m2.pvalues[term2], se_county_cl=mc2.bse[term2], p_county_cl=mc2.pvalues[term2],
            n=int(m2.nobs))

pd.DataFrame([row1, row2]).to_csv(os.path.join(BASE, 'wa_or_did_summary.csv'), index=False)

# ---------- Event-year path (relative to FY2015, the only pre-year) ----------
d['ey_cat'] = pd.Categorical(d['fy_year'])
formula3 = 'log_revenue_per_store ~ log_population + id_border_70mi + or_border_70mi:C(ey_cat, Treatment(reference=2015))'
m3 = smf.ols(formula3, data=d).fit(cov_type='HC1')
mc3 = smf.ols(formula3, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
rows = []
for yr in sorted(d['fy_year'].unique()):
    if yr == 2015:
        continue
    term3 = f'or_border_70mi:C(ey_cat, Treatment(reference=2015))[{yr}]'
    if term3 not in m3.params.index:
        continue
    rows.append(dict(fy_year=yr, coef=m3.params[term3], se_hc1=m3.bse[term3], p_hc1=m3.pvalues[term3],
                      se_county_cl=mc3.bse[term3], p_county_cl=mc3.pvalues[term3]))
res = pd.DataFrame(rows)
print(f"\n--- Event-year path relative to FY2015 (n={int(m3.nobs)}, R2={m3.rsquared:.3f}) ---")
print(res.round(4).to_string(index=False))
res.to_csv(os.path.join(BASE, 'wa_or_eventstudy_path.csv'), index=False)

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(res['fy_year'], res['coef'], marker='o', color='tab:blue')
ax.fill_between(res['fy_year'], res['coef'] - 1.96 * res['se_county_cl'],
                 res['coef'] + 1.96 * res['se_county_cl'], alpha=0.15, color='tab:blue')
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(2015.75, color='red', ls='--', lw=1, label='OR retail opens (2015-10-01, within FY2016)')
ax.set_xlabel('Fiscal year (ref. = FY2015, last fully pre-OR-legal year)')
ax.set_ylabel('Coefficient on OR-border (log revenue/store)')
ax.set_title('WA counties bordering Oregon: revenue-per-store around OR legalization\n(replication of Hansen, Miller & Weber 2020 mechanism, own data)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_wa_or_eventstudy.png'), dpi=150)
print("\nSaved fig_wa_or_eventstudy.png")
