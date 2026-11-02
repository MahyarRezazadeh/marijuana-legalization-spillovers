import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

THRESH = int(sys.argv[1])

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

s = pd.read_csv(f"data/stacked_panel_main_t{THRESH}.csv", dtype={"fips5": str})
s = s[s["d_od"].notna() & (s["population"] > 0)].copy()
s["division"] = s["state_abbr"].map(DIVISION)
s["g"] = pd.Categorical(s["state_abbr"]).codes
s["g_county"] = pd.Categorical(s["fips5"]).codes

specs = {
    "No controls": "{y} ~ {x} + C(transition)",
    "+ Baseline overdose-rate level": "{y} ~ {x} + baseline_rate + C(transition)",
    "+ State FE": "{y} ~ {x} + baseline_rate + C(transition) + C(state_abbr)",
    "+ Census division x transition FE": "{y} ~ {x} + baseline_rate + C(transition):C(division)",
    "+ State x transition FE": "{y} ~ {x} + baseline_rate + C(transition):C(state_abbr)",
    "+ County FE": "{y} ~ {x} + baseline_rate + C(transition) + C(fips5)",
}

rows = []
for spec_name, tmpl in specs.items():
    for xname in ["switcher", "d_dist_mi"]:
        sub = s.dropna(subset=["baseline_rate", "division"])
        formula = tmpl.format(y="d_od", x=xname)
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g"]})
        m_c = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g_county"]})
        n_county_clusters = int(sub["g_county"].nunique())
        rows.append(dict(spec=spec_name, x=xname, coef=m.params[xname], se=m.bse[xname],
                          p=m.pvalues[xname], se_county=m_c.bse[xname], p_county=m_c.pvalues[xname],
                          n_county_clusters=n_county_clusters, n=int(m.nobs)))
        print(f"T={THRESH} {spec_name:38s} [{xname:10s}] b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} "
              f"p={m.pvalues[xname]:.4f} | se_county={m_c.bse[xname]:.5f} p_county={m_c.pvalues[xname]:.4f} "
              f"n={int(m.nobs)}")

pd.DataFrame(rows).to_csv(f"data/main_results_t{THRESH}.csv", index=False)
