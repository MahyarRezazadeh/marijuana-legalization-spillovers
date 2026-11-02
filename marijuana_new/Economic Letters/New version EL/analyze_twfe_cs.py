import sys
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from csdid.att_gt import ATTgt

np.random.seed(20260726)

THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 50

panel = pd.read_csv(f"data/panel_twfe_cs_t{THRESH}.csv", dtype={"fips5": str})
panel["g_state"] = pd.Categorical(panel["state_abbr"]).codes
panel["g_county"] = pd.Categorical(panel["fips5"]).codes

# ---------------- TWFE ----------------
twfe = smf.ols("od_rate ~ treated + C(fips5) + C(year)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["g_state"]}
)
twfe_c = smf.ols("od_rate ~ treated + C(fips5) + C(year)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["g_county"]}
)
twfe_coef, twfe_se, twfe_p = twfe.params["treated"], twfe.bse["treated"], twfe.pvalues["treated"]
twfe_se_county, twfe_p_county = twfe_c.bse["treated"], twfe_c.pvalues["treated"]
print(f"[T={THRESH}] TWFE: b={twfe_coef:+.4f} se={twfe_se:.4f} p={twfe_p:.4f} "
      f"| se_county={twfe_se_county:.4f} p_county={twfe_p_county:.4f} n={int(twfe.nobs)}")


def run_cs(xformla, clustervar="state_abbr"):
    att = ATTgt(
        yname="od_rate", tname="year", idname="fips_int", gname="g",
        data=panel, control_group=["nevertreated"], clustervar=clustervar,
        xformla=xformla,
        panel=True, allow_unbalanced_panel=False, bstrap=True, biters=1000, cband=False,
    )
    fit = att.fit()

    # log raw group-time cells and flag any group whose cells are entirely
    # NaN (seen at 100mi under the covariate spec -- a cohort-level DR-DID
    # estimation failure, not a missing-data fluke)
    mp = fit.MP
    g_arr = np.asarray(mp["group"]).flatten()
    t_arr = np.asarray(mp["t"]).flatten()
    a_arr = np.asarray(mp["att"]).flatten()
    gt_cells = [{"g": int(g), "t": int(t), "att": (None if np.isnan(a) else float(a))}
                for g, t, a in zip(g_arr, t_arr, a_arr)]
    nan_groups = sorted({int(g) for g, a in zip(g_arr, a_arr) if np.isnan(a)})

    simple = att.aggte(typec="simple", na_rm=True)
    cs_att = float(simple.atte["overall_att"])
    cs_se = float(np.asarray(simple.atte["overall_se"]).flatten()[0])
    z = cs_att / cs_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    dyn = att.aggte(typec="dynamic", na_rm=True)
    d = dyn.atte
    egt = np.asarray(d["egt"]).flatten()
    att_egt = np.asarray(d["att_egt"]).flatten()
    se_egt = np.asarray(d["se_egt"]).flatten()
    dyn_table = []
    for e, a, s in zip(egt, att_egt, se_egt):
        zz = a / s
        pp = 2 * (1 - stats.norm.cdf(abs(zz)))
        dyn_table.append({"event_time": int(e), "att": float(a), "se": float(s), "p": float(pp)})

    return {"att": cs_att, "se": cs_se, "p": float(p), "n": int(panel.shape[0]), "dynamic": dyn_table,
            "gt_cells": gt_cells, "nan_groups": nan_groups}


print(f"[T={THRESH}] Running CS, no covariates, state-clustered...")
cs_nocov = run_cs(None, clustervar="state_abbr")
print(f"[T={THRESH}] CS (no covariates): b={cs_nocov['att']:+.4f} se={cs_nocov['se']:.4f} p={cs_nocov['p']:.4f}")
for row in cs_nocov["dynamic"]:
    print(f"    t={row['event_time']:+d}: att={row['att']:+.4f} se={row['se']:.4f} p={row['p']:.4f}")

print(f"[T={THRESH}] Running CS, no covariates, county-clustered...")
cs_nocov_county = run_cs(None, clustervar="fips5")
print(f"[T={THRESH}] CS (no covariates, county-clustered): b={cs_nocov_county['att']:+.4f} "
      f"se={cs_nocov_county['se']:.4f} p={cs_nocov_county['p']:.4f}")

print(f"[T={THRESH}] Running CS, + baseline-rate covariate, state-clustered...")
cs_cov = run_cs("~baseline_rate", clustervar="state_abbr")
print(f"[T={THRESH}] CS (+baseline_rate): b={cs_cov['att']:+.4f} se={cs_cov['se']:.4f} p={cs_cov['p']:.4f}")
for row in cs_cov["dynamic"]:
    print(f"    t={row['event_time']:+d}: att={row['att']:+.4f} se={row['se']:.4f} p={row['p']:.4f}")

print(f"[T={THRESH}] Running CS, + baseline-rate covariate, county-clustered...")
cs_cov_county = run_cs("~baseline_rate", clustervar="fips5")
print(f"[T={THRESH}] CS (+baseline_rate, county-clustered): b={cs_cov_county['att']:+.4f} "
      f"se={cs_cov_county['se']:.4f} p={cs_cov_county['p']:.4f}")

out = {
    "thresh": THRESH,
    "twfe": {"coef": twfe_coef, "se": twfe_se, "p": twfe_p,
             "se_county": twfe_se_county, "p_county": twfe_p_county, "n": int(twfe.nobs)},
    "cs_nocov": cs_nocov,
    "cs_nocov_county": cs_nocov_county,
    "cs_cov": cs_cov,
    "cs_cov_county": cs_cov_county,
}
outpath = f"data/twfe_cs_result_t{THRESH}.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {outpath}")
