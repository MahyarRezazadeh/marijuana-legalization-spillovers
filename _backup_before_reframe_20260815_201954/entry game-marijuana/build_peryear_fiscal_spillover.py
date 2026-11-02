# ============================================================
# Per-YEAR (calendar year, not market age, and not a single 2025
# snapshot) fiscal spillover: estimates a separate border_70mi
# coefficient for each calendar year via a full border x year
# interaction, then applies each year's OWN coefficient to that
# year's ACTUAL border revenue. This avoids assuming the pooled,
# average premium (b=0.341 for CO+WA, b=0.654 for MT) held constant
# across the whole sample -- we already know from the market-age
# event study that it didn't (CO+WA premium wasn't significant until
# market-age year 7).
#
# CO+WA: revenue per dispensary, corrected panel, 2015-2025.
# Montana: revenue per capita, 2022-2025 (only 4 years available).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')


def run_peryear(panel, outcome, treat, controls, cluster_col, label):
    d = panel.copy()
    d['yr'] = pd.Categorical(d['fy_year'])
    d['g'] = pd.Categorical(d[cluster_col]).codes
    formula = f'{outcome} ~ {controls} + C(yr) + {treat}:C(yr)'
    m = smf.ols(formula, data=d).fit(cov_type='HC1')
    mc = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['g']})
    rows = []
    for yr in sorted(d['fy_year'].unique()):
        term = f'{treat}:C(yr)[{yr}]'
        if term not in m.params.index:
            continue
        b, p = m.params[term], m.pvalues[term]
        se_c, p_c = mc.bse[term], mc.pvalues[term]
        actual_border_rev = d.loc[(d['fy_year'] == yr) & (d[treat] == 1), 'total_sales'].sum()
        premium = np.exp(b) - 1
        attributable = actual_border_rev * (premium / (1 + premium))
        rows.append(dict(year=int(yr), coef=b, p_hc1=p, p_county_cl=p_c, premium_pct=premium * 100,
                          actual_border_revenue=actual_border_rev, attributable=attributable))
        print(f"[{label}] {int(yr)}: b={b:+.4f} p(cl)={p_c:.4f} premium={premium*100:+.1f}%  "
              f"border_rev=${actual_border_rev:,.0f}  attributable=${attributable:,.0f}")
    return pd.DataFrame(rows)


# ---------- CO+WA: revenue per dispensary, calendar-year interaction ----------
print("=" * 70)
print("CO+WA -- revenue per dispensary, per-calendar-year border effect")
print("=" * 70)
cowa = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), dtype={'fips5': str})
cowa_peryear = run_peryear(cowa, 'log_revenue_per_store', 'border_70mi',
                            'log_population + C(state_abbr)', 'fips5', 'CO+WA')
cowa_peryear.to_csv(os.path.join(TABLES, 'cowa_peryear_fiscal_spillover.csv'), index=False)

# ---------- Montana: revenue per capita, calendar-year interaction ----------
print("\n" + "=" * 70)
print("MONTANA -- revenue per capita, per-calendar-year border effect")
print("=" * 70)
mt_panel = pd.read_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'), dtype={'fips5': str})
mt_peryear = run_peryear(mt_panel, 'log_revenue_per_capita', 'border_70mi',
                          'log_population', 'fips5', 'Montana')
mt_peryear.to_csv(os.path.join(TABLES, 'mt_peryear_fiscal_spillover.csv'), index=False)

# ---------- Totals: sum of per-year attributable dollars vs. the flat pooled-premium estimate ----------
print("\n" + "=" * 70)
print("SUM OF PER-YEAR ATTRIBUTABLE $ vs. FLAT-PREMIUM ESTIMATE (2025 only, from prior table)")
print("=" * 70)
print(f"CO+WA: sum across all years with a significant OR insignificant per-year coef = "
      f"${cowa_peryear['attributable'].sum():,.0f}")
print(f"CO+WA: sum across years actually significant at p<0.10 (county-cl.) = "
      f"${cowa_peryear.loc[cowa_peryear['p_county_cl'] < 0.10, 'attributable'].sum():,.0f}")
print(f"Montana: sum across all 4 years = ${mt_peryear['attributable'].sum():,.0f}")
print(f"Montana: sum across years significant at p<0.10 (county-cl.) = "
      f"${mt_peryear.loc[mt_peryear['p_county_cl'] < 0.10, 'attributable'].sum():,.0f}")
