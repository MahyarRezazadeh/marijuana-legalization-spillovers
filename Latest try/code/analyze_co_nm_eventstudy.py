# ============================================================
# Event study: does the revenue-per-store premium of CO counties
# bordering New Mexico shrink once NM's own retail market opens
# (2022-04-01)? This is the novel switch-off extension to Hansen,
# Miller & Weber (2020) -- a second, independent test of their
# "captive demand disappears once the neighbor legalizes" mechanism,
# on a border/event they did not study.
#
# Three groups: NM-border (treatment, 6 counties), other-nonlegal-
# border (control 1: still-illegal neighbor throughout -- UT/WY/OK/NE/
# KS, 5 counties), interior (control 2, 31 counties). Panel: CO
# 2014/2015-2025, annual, revenue_per_store from the project's
# existing 70mi-threshold construction (current-snapshot store count,
# same convention as the rest of the paper).
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'co_nm_switch_panel.csv'), dtype={'fips5': str})

print("Years available:", sorted(panel['fy_year'].unique()))

# ---------- Event-study spec: NM-border vs interior ----------
def run_eventstudy(df, treat_col, label, lo=-7, hi=3):
    d = df.copy()
    d['ey'] = d['event_year'].clip(lower=lo, upper=hi)
    d['g'] = pd.Categorical(d['fips5']).codes
    d['ey_cat'] = pd.Categorical(d['ey'])
    # drop event_year == -1 (2021, last full pre-year) as reference
    formula = f'log_revenue_per_store ~ log_population + C(fy_year) + {treat_col}:C(ey_cat, Treatment(reference=-1))'
    m = smf.ols(formula, data=d).fit(cov_type='HC1')
    mc = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
    rows = []
    for ey in sorted(d['ey'].unique()):
        if ey == -1:
            continue
        term = f'{treat_col}:C(ey_cat, Treatment(reference=-1))[{ey}]'
        if term not in m.params.index:
            continue
        b, p = m.params[term], m.pvalues[term]
        se_c, p_c = mc.bse[term], mc.pvalues[term]
        rows.append(dict(event_year=ey, coef=b, se_hc1=m.bse[term], p_hc1=p,
                          se_county_cl=se_c, p_county_cl=p_c))
    res = pd.DataFrame(rows).sort_values('event_year')
    print(f"\n--- Event study: {label} (n={int(m.nobs)}, R2={m.rsquared:.3f}) ---")
    print(res.round(4).to_string(index=False))
    return res

res_vs_interior = run_eventstudy(
    panel[(panel['nm_border_70mi'] == 1) | (panel['interior'] == 1)],
    'nm_border_70mi', 'NM-border vs interior')
res_vs_interior.to_csv(os.path.join(BASE, 'co_nm_eventstudy_vs_interior.csv'), index=False)

res_vs_otherborder = run_eventstudy(
    panel[(panel['nm_border_70mi'] == 1) | (panel['other_border_70mi'] == 1)],
    'nm_border_70mi', 'NM-border vs other-nonlegal-border (UT/WY/OK/NE/KS)')
res_vs_otherborder.to_csv(os.path.join(BASE, 'co_nm_eventstudy_vs_otherborder.csv'), index=False)

# ---------- Simple pre/post DiD (drop 2022 as transition year) ----------
d = panel[panel['fy_year'] != 2022].copy()
d['post'] = (d['fy_year'] >= 2023).astype(int)
d['g'] = pd.Categorical(d['fips5']).codes
formula = 'log_revenue_per_store ~ nm_border_70mi*post + other_border_70mi + log_population + C(fy_year)'
m = smf.ols(formula, data=d).fit(cov_type='HC1')
mc = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
term = 'nm_border_70mi:post'
print(f"\n--- Simple DiD (2022 dropped as transition year) ---")
print(f"n={int(m.nobs)} R2={m.rsquared:.3f}")
print(f"{term}: b={m.params[term]:+.4f}  se(HC1)={m.bse[term]:.4f} p={m.pvalues[term]:.4f}  "
      f"| se(county-cl)={mc.bse[term]:.4f} p={mc.pvalues[term]:.4f}")
pct_change = (np.exp(m.params[term]) - 1) * 100
print(f"Implied change in revenue-per-store for NM-border counties after NM opens: {pct_change:+.1f}%")

did_summary = pd.DataFrame([dict(
    spec='NM-border x post(2023+), 2022 dropped', coef=m.params[term], pct_change=pct_change,
    se_hc1=m.bse[term], p_hc1=m.pvalues[term], se_county_cl=mc.bse[term], p_county_cl=mc.pvalues[term],
    n=int(m.nobs))])
did_summary.to_csv(os.path.join(BASE, 'co_nm_did_summary.csv'), index=False)

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
for res, lbl, color in [(res_vs_interior, 'vs. interior counties', 'tab:blue'),
                         (res_vs_otherborder, 'vs. other-nonlegal-border counties', 'tab:orange')]:
    ax.plot(res['event_year'], res['coef'], marker='o', color=color, label=lbl)
    ax.fill_between(res['event_year'], res['coef'] - 1.96 * res['se_county_cl'],
                     res['coef'] + 1.96 * res['se_county_cl'], alpha=0.15, color=color)
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0.25, color='red', ls='--', lw=1, label="NM retail opens (2022-04-01)")
ax.set_xlabel('Event year (0 = 2022, NM legalization year)')
ax.set_ylabel('Coefficient on NM-border (log revenue/store), rel. to 2021')
ax.set_title('CO counties bordering New Mexico: revenue-per-store event study')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_co_nm_eventstudy.png'), dpi=150)
print("\nSaved fig_co_nm_eventstudy.png")
