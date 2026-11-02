# ============================================================
# Alcohol placebo, full check: does the cannabis border revenue-per-store
# premium (the paper's central finding, substitution.tex) show up in
# ALCOHOL retail too, or is it cannabis-specific? Same logic as the
# existing CO-only general-retail placebo (analyze_placebo_general_retail.py),
# but alcohol is conceptually the sharper test: it's a similarly
# regulated, excise-taxed good with its own well-documented cross-border
# tax-arbitrage literature, so a null here is harder to wave away than
# a null in ordinary grocery/clothing retail.
#
# Coverage across all 4 states in substitution.tex, checked directly
# (not assumed):
#   CO -- county-level alcohol data does NOT exist publicly. CDOR's
#         sales-tax dataset (already used for the general-retail placebo)
#         has no liquor-store-specific NAICS breakout, and its separate
#         Liquor Excise Tax Report (data.colorado.gov 43i8-hhvf) is
#         statewide-only, no county field at all.
#   WA -- county-level alcohol data does NOT exist publicly either.
#         WSLCB's own "Liquor Sales, Revenue, and Distributions"
#         dashboard (lcb.wa.gov/research/liquor-revenue-dashboard) embed
#         is dead (blank on load, checked directly in-browser -- same
#         class of failure as MA's dead cannabis-sales Tableau map
#         elsewhere in this project). DOR's spirits-tax statistics page
#         confirms its own downloadable file is statewide-only. Beer/wine
#         "sales to importers and distributors" reports are statewide
#         monthly totals, not county, not even retail (upstream of it).
#   OR -- STRONG. OLCC (control state for spirits, same agency that
#         regulates cannabis retail) publishes monthly per-store spirits
#         sales with a native county field (data.oregon.gov gyqr-vac8),
#         2019-2026, all 36 counties incl. all 3 cannabis-design border
#         counties (Baker, Malheur, Wallowa). Built in
#         build_oregon_alcohol_panel.py -- revenue-per-capita AND
#         revenue-per-store (store count = distinct active store numbers).
#   MT -- MODERATE. Also a control state (private "agency liquor
#         stores"), same ABCD division as the cannabis CARD reports. No
#         bulk API -- scraped from the annual "Enterprise Fund Report of
#         Operations" PDFs' by-county schedule (scrape_montana_alcohol.py).
#         FY2021 unavailable (scanned image, not machine-readable text).
#         Revenue-per-capita only (no reliable per-year store count).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')

oregon = pd.read_csv(os.path.join(TABLES, 'oregon_alcohol_panel.csv'))
montana = pd.read_csv(os.path.join(TABLES, 'montana_alcohol_panel.csv'))

# ---------- Pooled OR+MT, revenue per capita ----------
keep_cols = ['fips5', 'county_name', 'state_abbr', 'fy_year', 'population',
             'dist_nonlegal_mi', 'border_70mi', 'revenue_per_capita',
             'log_revenue_per_capita', 'log_population']
pooled = pd.concat([oregon[keep_cols], montana[keep_cols]], ignore_index=True)
pooled['county_g'] = pd.Categorical(pooled['fips5']).codes

print(f"Pooled OR+MT alcohol panel: {len(pooled)} county-year obs, {pooled['fips5'].nunique()} counties")
print(f"Border (<=70mi): {pooled.loc[pooled['border_70mi']==1,'fips5'].nunique()} counties, "
      f"interior: {pooled.loc[pooled['border_70mi']==0,'fips5'].nunique()} counties")
print("\nUnconditional means, revenue per capita, by state x border:")
print(pooled.groupby(['state_abbr', 'border_70mi'])['revenue_per_capita'].mean().unstack().round(2))

print("\n" + "=" * 70)
print("PLACEBO -- outcome: log(alcohol revenue per capita), OR+MT pooled")
print("=" * 70)
specs = {
    'No controls':                     'log_revenue_per_capita ~ border_70mi',
    '+ log(pop) + year FE':            'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)',
    '+ log(pop) + year FE + state FE': 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
}
rows = []
for name, formula in specs.items():
    m = smf.ols(formula, data=pooled).fit(cov_type='HC1')
    mc = smf.ols(formula, data=pooled).fit(cov_type='cluster', cov_kwds={'groups': pooled['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:34s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(outcome='revenue_per_capita', sample='OR+MT pooled', spec=name, coef=b,
                      se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

print("\nPer-state (own-state FE not needed, single state):")
for st, sub in pooled.groupby('state_abbr'):
    sub = sub.copy()
    sub['county_g'] = pd.Categorical(sub['fips5']).codes
    formula = 'log_revenue_per_capita ~ border_70mi + log_population + C(fy_year)'
    m = smf.ols(formula, data=sub).fit(cov_type='HC1')
    mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    n_border = sub.loc[sub['border_70mi']==1, 'fips5'].nunique()
    print(f"  {st:3s} (border counties={n_border}) b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} "
          f"| se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(outcome='revenue_per_capita', sample=st, spec='+ log(pop) + year FE', coef=b,
                      se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

# ---------- Oregon-only, revenue per store (matches the paper's central outcome) ----------
print("\n" + "=" * 70)
print("PLACEBO -- outcome: log(alcohol revenue per STORE), Oregon only")
print("(MT has no reliable per-year store count; this is OR's own comparison")
print(" to the cannabis design's central revenue-per-store finding)")
print("=" * 70)
oregon['county_g'] = pd.Categorical(oregon['fips5']).codes
specs2 = {
    'No controls':          'log_revenue_per_store ~ border_70mi',
    '+ log(pop) + year FE': 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year)',
}
for name, formula in specs2.items():
    m = smf.ols(formula, data=oregon).fit(cov_type='HC1')
    mc = smf.ols(formula, data=oregon).fit(cov_type='cluster', cov_kwds={'groups': oregon['county_g']})
    b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
    se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
    print(f"{name:34s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
    rows.append(dict(outcome='revenue_per_store', sample='OR', spec=name, coef=b,
                      se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TABLES, 'placebo_alcohol_results.csv'), index=False)
print(f"\nSaved to {os.path.join(TABLES, 'placebo_alcohol_results.csv')}")
