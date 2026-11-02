import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

s = pd.read_csv(os.path.join(DATA, "stacked_panel_dispensary.csv"), dtype={"fips5": str})
s = s[s["d_od"].notna() & (s["population"] > 0)].copy()

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
s["division"] = s["state_abbr"].map(DIVISION)
s["g"] = pd.Categorical(s["state_abbr"]).codes  # state-clustered SEs, matching the main design
s["g_county"] = pd.Categorical(s["fips5"]).codes

n_per_county = s.groupby("fips5").size()
print(f"Counties with only 1 transition-row (no within-county variation for county FE): "
      f"{(n_per_county == 1).sum()} of {len(n_per_county)}")

specs = {
    "No controls (transition FE only)": "{y} ~ {x} + C(transition)",
    "+ baseline (2020) overdose rate": "{y} ~ {x} + baseline_rate + C(transition)",
    "+ state FE": "{y} ~ {x} + baseline_rate + C(transition) + C(state_abbr)",
    "+ census division x transition FE": "{y} ~ {x} + baseline_rate + C(transition):C(division)",
    "+ state x transition FE": "{y} ~ {x} + baseline_rate + C(transition):C(state_abbr)",
    "+ county FE": "{y} ~ {x} + baseline_rate + C(transition) + C(fips5)",
}

rows = []
for spec_name, tmpl in specs.items():
    for xname in ["d_dist_mi", "switcher"]:
        sub = s.dropna(subset=["baseline_rate", "division"])
        formula = tmpl.format(y="d_od", x=xname)
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g"]})
        m_c = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g_county"]})
        rows.append(dict(spec=spec_name, x=xname, coef=m.params[xname], se=m.bse[xname],
                          p=m.pvalues[xname], se_county=m_c.bse[xname], p_county=m_c.pvalues[xname],
                          n_county_clusters=int(sub["g_county"].nunique()), n=int(m.nobs)))
        print(f"{spec_name:38s} [{xname:10s}] b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} "
              f"p={m.pvalues[xname]:.4f} | se_county={m_c.bse[xname]:.5f} p_county={m_c.pvalues[xname]:.4f} n={int(m.nobs)}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(DATA, "dispensary_robustness_results.csv"), index=False)
print("\nSaved dispensary_robustness_results.csv")
