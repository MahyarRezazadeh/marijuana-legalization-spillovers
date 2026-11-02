# ============================================================
# Rebuilds the CO+WA revenue-per-store panel with corrected Washington
# dispensary counts (see rebuild_wa_dispensary_counts.py), then
# re-estimates the exact same specifications as
# analyze_revenue_perstore_border70.py for direct before/after
# comparison. Colorado is untouched (its dispensary data checked out
# against real-world totals).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# ---------- Colorado: keep exactly as before (unaffected) ----------
old_panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
co_part = old_panel[old_panel['state_abbr'] == 'CO'].copy()

# ---------- Washington: rebuild from the full revenue series + corrected counts ----------
rev = pd.read_csv(os.path.join(BASE, 'border revenue', 'data', 'county_revenue_panel.csv'), dtype={'fips5': str})
wa_rev = rev[rev['state_abbr'] == 'WA'].copy()
wa_rev = wa_rev[wa_rev['revenue_per_capita'] > 0].copy()

wa_counts = pd.read_csv(os.path.join(TABLES, 'wa_dispensary_counts_corrected.csv'))
wa_rev['county_key_up'] = wa_rev['county_key'].str.strip().str.upper()
wa_rev = wa_rev.merge(wa_counts, left_on='county_key_up', right_on='county_key', how='left', suffixes=('', '_y'))
wa_rev['dispensary_count'] = wa_rev['dispensary_count_new'].fillna(0)
wa_rev = wa_rev[wa_rev['dispensary_count'] > 0].copy()

wa_rev['revenue_per_store'] = wa_rev['total_sales'] / wa_rev['dispensary_count']
wa_rev['log_revenue_per_store'] = np.log(wa_rev['revenue_per_store'])
wa_rev['log_population'] = np.log(wa_rev['population'])
wa_rev['border_70mi'] = (wa_rev['dist_nonlegal_mi'] <= 70).astype(int)

wa_cols = ['fips5', 'state_abbr', 'county_name', 'lat', 'lon', 'population', 'land_area_sqmi',
           'dist_nonlegal_mi', 'nearest_nonlegal_state', 'county_key', 'fy_year', 'total_sales',
           'dispensary_count', 'revenue_per_store', 'log_revenue_per_store', 'log_population', 'border_70mi']
wa_part = wa_rev[wa_cols].copy()

n_before = old_panel[old_panel['state_abbr'] == 'WA']['fips5'].nunique()
n_after = wa_part['fips5'].nunique()
print(f"WA counties in panel: {n_before} (old) -> {n_after} (corrected)")
print(f"WA border counties: {old_panel[(old_panel.state_abbr=='WA')&(old_panel.border_70mi==1)]['fips5'].nunique()} (old) -> "
      f"{wa_part[wa_part.border_70mi==1]['fips5'].nunique()} (corrected)")

new_panel = pd.concat([co_part[wa_cols], wa_part], ignore_index=True)
new_panel['county_g'] = pd.Categorical(new_panel['fips5']).codes
new_panel.to_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel_CORRECTED.csv'), index=False)

print(f"\nTotal panel: old n={len(old_panel)}, corrected n={len(new_panel)}")
print(f"Border obs: old={( old_panel.border_70mi==1).sum()}, corrected={(new_panel.border_70mi==1).sum()}")
print(f"Interior obs: old={(old_panel.border_70mi==0).sum()}, corrected={(new_panel.border_70mi==0).sum()}")

# ---------- Re-estimate, same specs as analyze_revenue_perstore_border70.py ----------
def run(panel, label):
    print(f"\n{'='*70}\n{label} (n={len(panel)})\n{'='*70}")
    rows = []
    for name, formula in {
        'No controls': 'log_revenue_per_store ~ border_70mi',
        '+ log(pop) + year FE + state FE': 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year) + C(state_abbr)',
    }.items():
        m = smf.ols(formula, data=panel).fit(cov_type='HC1')
        mc = smf.ols(formula, data=panel).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(panel['fips5']).codes})
        b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
        se_c, p_c = mc.bse['border_70mi'], mc.pvalues['border_70mi']
        print(f"{name:40s} b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} | se(county-cl)={se_c:.4f} p={p_c:.4f} n={int(m.nobs)}")
        rows.append(dict(spec=name, coef=b, se_hc1=se, p_hc1=p, se_county_cl=se_c, p_county_cl=p_c, n=int(m.nobs)))

    # single-state estimates
    for st in ['CO', 'WA']:
        sub = panel[panel.state_abbr == st]
        formula = 'log_revenue_per_store ~ border_70mi + log_population + C(fy_year)'
        m = smf.ols(formula, data=sub).fit(cov_type='HC1')
        mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': pd.Categorical(sub['fips5']).codes})
        b, p = m.params['border_70mi'], m.pvalues['border_70mi']
        p_c = mc.pvalues['border_70mi']
        print(f"{st} only, +log(pop)+year FE{'':17s} b={b:+.4f} p(HC1)={p:.4f} p(county-cl)={p_c:.4f} n={int(m.nobs)}")
        rows.append(dict(spec=f'{st} only', coef=b, p_hc1=p, p_county_cl=p_c, n=int(m.nobs)))
    return pd.DataFrame(rows)

old_res = run(old_panel, "OLD (undercounted WA dispensaries)")
new_res = run(new_panel, "CORRECTED (real WSLCB WA dispensary counts)")

new_res.to_csv(os.path.join(TABLES, 'revenue_perstore_border70_results_CORRECTED.csv'), index=False)
print("\nSaved revenue_perstore_border70_panel_CORRECTED.csv and results_CORRECTED.csv")
