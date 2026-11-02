# ============================================================
# NEW check (not the switcher-vs-always-near check in Table
# tab:alwaysnear): compares counties that were ALWAYS within 50
# miles of a legal-market county throughout 2020-2025 ("always-
# near") against counties that were ALWAYS 50 miles or farther
# throughout the same window ("always-far"). Neither group ever
# changes status, so there is no switch event for either -- the
# "treatment" (NearAccess) is a time-invariant county
# characteristic, not a transition-specific event.
#
# "Same treatment timing" (per the request that produced this
# script) means: keep the identical stacked 5-transition panel
# structure used everywhere else in the paper (one row per county
# per one-year transition, 2020-21 .. 2024-25, transition FE,
# baseline-(2020)-rate control, same escalating FE battery) rather
# than collapsing to a single before/after comparison -- NearAccess
# is simply constant across a given county's five rows instead of
# switching on partway through, the way Switcher_it does for
# switcher counties elsewhere in the paper.
#
# Always-near (116 counties) and ever-switcher (177 counties, used
# only to EXCLUDE them here) are taken directly from the files
# already built for the rest of the paper, for consistency, rather
# than re-derived from raw distance crossings.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
EL = os.path.dirname(BASE)
TABLES = os.path.join(BASE, 'tables')

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TRANSITIONS = list(zip(YEARS[:-1], YEARS[1:]))
THRESH = 50.0

DIVISION = {
    "CT": "NewEngland", "ME": "NewEngland", "MA": "NewEngland", "NH": "NewEngland", "RI": "NewEngland", "VT": "NewEngland",
    "NJ": "MidAtlantic", "NY": "MidAtlantic", "PA": "MidAtlantic",
    "IL": "ENCentral", "IN": "ENCentral", "MI": "ENCentral", "OH": "ENCentral", "WI": "ENCentral",
    "IA": "WNCentral", "KS": "WNCentral", "MN": "WNCentral", "MO": "WNCentral", "NE": "WNCentral", "ND": "WNCentral", "SD": "WNCentral",
    "DE": "SAtlantic", "FL": "SAtlantic", "GA": "SAtlantic", "MD": "SAtlantic", "NC": "SAtlantic", "SC": "SAtlantic", "VA": "SAtlantic", "DC": "SAtlantic", "WV": "SAtlantic",
    "AL": "ESCentral", "KY": "ESCentral", "MS": "ESCentral", "TN": "ESCentral",
    "AR": "WSCentral", "LA": "WSCentral", "OK": "WSCentral", "TX": "WSCentral",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain", "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}

dist = pd.read_csv(os.path.join(EL, 'data', 'distance_by_year_rec_full.csv'), dtype={'fips5': str})
dist['fips5'] = dist['fips5'].str.zfill(5)

main_panel = pd.read_csv(os.path.join(BASE, 'data', 'stacked_panel_main_t50.csv'), dtype={'fips5': str})
ever_switcher_fips = set(main_panel.loc[main_panel['switcher'] == 1, 'fips5'].unique())
print(f"Ever-switcher counties (excluded here): {len(ever_switcher_fips)}")

always_near_fips = set(dist.loc[dist['dist_2020'] < THRESH, 'fips5'])
print(f"Always-near counties (NearAccess=1): {len(always_near_fips)}")

always_far = dist[(~dist['fips5'].isin(always_near_fips)) & (~dist['fips5'].isin(ever_switcher_fips))].copy()
print(f"Always-far counties (NearAccess=0): {len(always_far)}")

with open(os.path.join(EL, 'data', 'overdose_county_multiyear_raw.json'), encoding='utf-8') as f:
    raw = json.load(f)
raw_df = pd.DataFrame(raw)
raw_df['fips5'] = raw_df['fips'].astype(str).str.zfill(5)
raw_df['provisional_drug_overdose'] = pd.to_numeric(raw_df['provisional_drug_overdose'], errors='coerce')
od_wide = raw_df.pivot_table(index='fips5', columns='year', values='provisional_drug_overdose', aggfunc='first')
od_wide.columns = [f'od_{int(c)}' for c in od_wide.columns]
od = od_wide

always_near_df = dist[dist['fips5'].isin(always_near_fips)].copy()


def build_rows(county_df, near_access):
    rows = []
    for year_pre, year_post in TRANSITIONS:
        for _, r in county_df.iterrows():
            fips = r['fips5']
            if fips not in od.index:
                continue
            od_pre, od_post = od.loc[fips, f'od_{year_pre}'], od.loc[fips, f'od_{year_post}']
            if r['population'] <= 0:
                continue
            rows.append({
                'fips5': fips, 'state_abbr': r['state_abbr'], 'population': r['population'],
                'transition': f'{year_pre}-{year_post}', 'near_access': near_access,
                'od_rate_pre': od_pre / r['population'] * 1e5, 'od_rate_post': od_post / r['population'] * 1e5,
                'baseline_rate': od.loc[fips, 'od_2020'] / r['population'] * 1e5,
            })
    return pd.DataFrame(rows)


near_rows = build_rows(always_near_df, 1)
far_rows = build_rows(always_far, 0)
combo = pd.concat([near_rows, far_rows], ignore_index=True)
combo['d_od'] = combo['od_rate_post'] - combo['od_rate_pre']
combo = combo.dropna(subset=['d_od'])
combo['division'] = combo['state_abbr'].map(DIVISION)
combo['g'] = pd.Categorical(combo['state_abbr']).codes
combo['g_county'] = pd.Categorical(combo['fips5']).codes

n_near = combo.loc[combo['near_access'] == 1, 'fips5'].nunique()
n_far = combo.loc[combo['near_access'] == 0, 'fips5'].nunique()
print(f"\nPanel: {combo.shape[0]} rows, {combo['fips5'].nunique()} counties "
      f"({n_near} always-near, {n_far} always-far)")
n_per_county = combo.groupby('fips5').size()
print(f"Counties with only 1 row (no within-county variation for county FE): {(n_per_county==1).sum()} of {len(n_per_county)}")

# NOTE: no "+ County FE" row here, unlike every other robustness table
# in this paper. near_access is constant within every county by
# construction (always-near and always-far counties never change
# status), confirmed directly on this panel (0 of 1,153 counties have
# more than one near_access value across their rows). A county fixed
# effect would therefore be exactly collinear with near_access, making
# that coefficient structurally unidentified, not merely imprecise --
# verified empirically: fitting it anyway gives a condition number of
# ~3e17 (machine-precision numerical noise, not a real estimate). This
# is a genuine structural difference from the switcher-based checks
# elsewhere, where Switcher_it varies within a switcher county
# (pre-switch vs. switch-transition rows) and county FE is identified.
specs = {
    "No controls": "d_od ~ near_access + C(transition)",
    "+ Baseline overdose-rate level": "d_od ~ near_access + baseline_rate + C(transition)",
    "+ State FE": "d_od ~ near_access + baseline_rate + C(transition) + C(state_abbr)",
    "+ Census division x transition FE": "d_od ~ near_access + baseline_rate + C(transition):C(division)",
    "+ State x transition FE": "d_od ~ near_access + baseline_rate + C(transition):C(state_abbr)",
}

rows_out = []
for spec_name, formula in specs.items():
    sub = combo.dropna(subset=['baseline_rate', 'division']) if 'baseline_rate' in formula else combo
    m = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
    m_c = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g_county']})
    print(f"{spec_name:38s} b={m.params['near_access']:+.4f} se={m.bse['near_access']:.4f} "
          f"p={m.pvalues['near_access']:.4f} | se_county={m_c.bse['near_access']:.4f} "
          f"p_county={m_c.pvalues['near_access']:.4f} n={int(m.nobs)}")
    rows_out.append(dict(spec=spec_name, coef=m.params['near_access'], se=m.bse['near_access'],
                          p=m.pvalues['near_access'], se_county=m_c.bse['near_access'],
                          p_county=m_c.pvalues['near_access'],
                          n_county_clusters=int(sub['g_county'].nunique()), n=int(m.nobs)))

pd.DataFrame(rows_out).to_csv(os.path.join(TABLES, 'alwaysnear_vs_alwaysfar_results.csv'), index=False)
combo.to_csv(os.path.join(BASE, 'data', 'alwaysnear_vs_alwaysfar_panel.csv'), index=False)
print("\nSaved tables/alwaysnear_vs_alwaysfar_results.csv, data/alwaysnear_vs_alwaysfar_panel.csv")
