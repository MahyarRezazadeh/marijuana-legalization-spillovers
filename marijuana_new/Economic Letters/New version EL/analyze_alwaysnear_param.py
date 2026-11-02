import sys
import pandas as pd
import numpy as np
import json
import statsmodels.formula.api as smf

THRESH = float(sys.argv[1])
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
TRANSITIONS = list(zip(YEARS[:-1], YEARS[1:]))

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

with open(r"..\data\overdose_county_multiyear_raw.json", encoding="utf-8") as f:
    raw = json.load(f)
raw_df = pd.DataFrame(raw)
raw_df["fips5"] = raw_df["fips"].astype(str).str.zfill(5)
raw_df["provisional_drug_overdose"] = pd.to_numeric(raw_df["provisional_drug_overdose"], errors="coerce")
od_wide = raw_df.pivot_table(index="fips5", columns="year", values="provisional_drug_overdose", aggfunc="first")
od_wide.columns = [f"od_{int(c)}" for c in od_wide.columns]
od = od_wide


def build_and_run(dist_file, switcher_panel_file, label, outfile):
    dist = pd.read_csv(dist_file, dtype={"fips5": str})
    dist["fips5"] = dist["fips5"].str.zfill(5)
    always_near = dist[dist["dist_2020"] < THRESH].copy()

    rows = []
    for year_pre, year_post in TRANSITIONS:
        for _, r in always_near.iterrows():
            fips = r["fips5"]
            if fips not in od.index:
                continue
            od_pre, od_post = od.loc[fips, f"od_{year_pre}"], od.loc[fips, f"od_{year_post}"]
            rows.append({
                "fips5": fips, "state_abbr": r["state_abbr"], "population": r["population"],
                "transition": f"{year_pre}-{year_post}", "switcher": 0,
                "od_rate_pre": od_pre/r["population"]*1e5, "od_rate_post": od_post/r["population"]*1e5,
                "baseline_rate": od.loc[fips, "od_2020"]/r["population"]*1e5,
                "group": "always_near_control",
            })
    control_df = pd.DataFrame(rows)
    control_df["d_od"] = control_df["od_rate_post"] - control_df["od_rate_pre"]

    standard = pd.read_csv(switcher_panel_file, dtype={"fips5": str})
    if "baseline_rate" not in standard.columns:
        standard["baseline_rate"] = standard["od_2020"] / (standard["population"] / 1e5)
    ever_switcher_fips = standard.loc[standard["switcher"] == 1, "fips5"].unique()
    treated = standard[standard["fips5"].isin(ever_switcher_fips)].copy()
    treated["group"] = "switcher_treatment"

    combo = pd.concat([
        treated[["fips5", "state_abbr", "population", "transition", "switcher", "d_od", "baseline_rate", "group"]],
        control_df[["fips5", "state_abbr", "population", "transition", "switcher", "d_od", "baseline_rate", "group"]],
    ], ignore_index=True)
    combo = combo.dropna(subset=["d_od"])
    combo["g"] = pd.Categorical(combo["state_abbr"]).codes
    combo["g_county"] = pd.Categorical(combo["fips5"]).codes
    combo["division"] = combo["state_abbr"].map(DIVISION)

    n_switch = combo.loc[combo["group"] == "switcher_treatment", "fips5"].nunique()
    n_near = combo.loc[combo["group"] == "always_near_control", "fips5"].nunique()
    print(f"[{label} T={THRESH}] {combo.shape[0]} rows, {n_switch} ever-switcher + {n_near} always-near counties")

    results = []
    for spec_name, formula in [
        ("No controls", "d_od ~ switcher + C(transition)"),
        ("+ baseline overdose rate", "d_od ~ switcher + baseline_rate + C(transition)"),
        ("+ state FE", "d_od ~ switcher + baseline_rate + C(transition) + C(state_abbr)"),
        ("+ census division x transition FE", "d_od ~ switcher + baseline_rate + C(transition):C(division)"),
        ("+ state x transition FE", "d_od ~ switcher + baseline_rate + C(transition):C(state_abbr)"),
        ("+ county FE", "d_od ~ switcher + baseline_rate + C(transition) + C(fips5)"),
    ]:
        sub = combo.dropna(subset=["baseline_rate", "division"]) if "baseline_rate" in formula else combo
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g"]})
        m_c = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g_county"]})
        print(f"  {spec_name:38s} b={m.params['switcher']:+.4f} se={m.bse['switcher']:.4f} "
              f"p={m.pvalues['switcher']:.4f} | se_county={m_c.bse['switcher']:.4f} p_county={m_c.pvalues['switcher']:.4f} n={int(m.nobs)}")
        results.append(dict(spec=spec_name, coef=m.params["switcher"], se=m.bse["switcher"],
                             p=m.pvalues["switcher"], se_county=m_c.bse["switcher"], p_county=m_c.pvalues["switcher"],
                             n_county_clusters=int(sub["g_county"].nunique()), n=int(m.nobs)))
    pd.DataFrame(results).to_csv(outfile, index=False)


build_and_run(r"..\data\distance_by_year_rec_full.csv", f"data/stacked_panel_main_t{int(THRESH)}.csv",
              "PanelA-state", f"data/alwaysnear_panelA_t{int(THRESH)}.csv")
build_and_run("data/distance_by_year_dispensary.csv", f"data/stacked_panel_dispensary_t{int(THRESH)}.csv",
              "PanelB-dispensary", f"data/alwaysnear_panelB_t{int(THRESH)}.csv")
