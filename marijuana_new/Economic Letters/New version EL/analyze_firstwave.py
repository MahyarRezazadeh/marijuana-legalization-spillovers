import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

panel = pd.read_csv("data/stacked_panel_firstwave.csv", dtype={"fips5": str})
panel = panel.dropna(subset=["d_rate"])
panel["g"] = pd.Categorical(panel["state_abbr"]).codes

print(f"N={panel.shape[0]}, switch events={panel['switcher'].sum()}, "
      f"states={panel['state_abbr'].nunique()}, counties={panel['fips5'].nunique()}")

results = []
for spec_name, formula_tmpl in [
    ("No controls (transition FE only)", "d_rate ~ {x} + C(transition)"),
    ("+ baseline (2013) rate", "d_rate ~ {x} + baseline_rate + C(transition)"),
    ("+ state FE", "d_rate ~ {x} + baseline_rate + C(transition) + C(state_abbr)"),
]:
    for xname in ["switcher", "d_dist_mi"]:
        sub = panel.dropna(subset=["baseline_rate"]) if "baseline_rate" in formula_tmpl else panel
        formula = formula_tmpl.format(x=xname)
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["g"]})
        print(f"{spec_name:35s} [{xname:10s}] b={m.params[xname]:+.5f} se={m.bse[xname]:.5f} "
              f"p={m.pvalues[xname]:.4f} n={int(m.nobs)}")
        results.append(dict(spec=spec_name, x=xname, coef=m.params[xname], se=m.bse[xname],
                             p=m.pvalues[xname], n=int(m.nobs)))

pd.DataFrame(results).to_csv("data/firstwave_results.csv", index=False)
