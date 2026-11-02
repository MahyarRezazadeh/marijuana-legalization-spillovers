import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

THRESH = 50

s = pd.read_csv(f"data/stacked_panel_main_t{THRESH}.csv", dtype={"fips5": str})
s = s[s["d_od"].notna() & s["d_dist_mi"].notna() & s["baseline_rate"].notna() & (s["population"] > 0)].copy()
s["g"] = pd.Categorical(s["state_abbr"]).codes
print(f"Dose-response sample: {s.shape[0]} county-transition rows, {s['fips5'].nunique()} counties")
print(s["d_dist_mi"].describe())

# ---------- (1) flexible polynomial specification, county + transition FE ----------
s["d2"] = s["d_dist_mi"] ** 2
s["d3"] = s["d_dist_mi"] ** 3

lin = smf.ols("d_od ~ d_dist_mi + baseline_rate + C(transition) + C(fips5)", data=s).fit(
    cov_type="cluster", cov_kwds={"groups": s["g"]}
)
cubic = smf.ols("d_od ~ d_dist_mi + d2 + d3 + baseline_rate + C(transition) + C(fips5)", data=s).fit(
    cov_type="cluster", cov_kwds={"groups": s["g"]}
)
print("\nLinear dose:")
print(f"  d_dist_mi: b={lin.params['d_dist_mi']:+.5f} se={lin.bse['d_dist_mi']:.5f} p={lin.pvalues['d_dist_mi']:.4f}")
print("\nCubic dose:")
for term in ["d_dist_mi", "d2", "d3"]:
    print(f"  {term}: b={cubic.params[term]:+.6g} se={cubic.bse[term]:.4g} p={cubic.pvalues[term]:.4f}")

f_test = cubic.f_test("d2 = 0, d3 = 0")
joint_p = float(np.asarray(f_test.pvalue))
print(f"\nJoint F-test (d2=d3=0), i.e. linearity test: F={float(f_test.fvalue):.3f} p={joint_p:.4f}")

# ---------- (2) Frisch-Waugh-Lovell residualization for the plot ----------
resid_y = smf.ols("d_od ~ baseline_rate + C(transition) + C(fips5)", data=s).fit().resid
resid_x = smf.ols("d_dist_mi ~ baseline_rate + C(transition) + C(fips5)", data=s).fit().resid
rdf = pd.DataFrame({"rx": resid_x, "ry": resid_y}).dropna()

# 20 equal-frequency bins for a binned-scatter view of the same relationship
rdf["bin"] = pd.qcut(rdf["rx"], 20, duplicates="drop")
binned = rdf.groupby("bin", observed=True).agg(
    x_mean=("rx", "mean"), y_mean=("ry", "mean"), y_se=("ry", lambda v: v.std(ddof=1) / np.sqrt(len(v))), n=("ry", "size")
).reset_index(drop=True)

# cubic fit on the residualized (FE-partialled-out) scale, for overlaying a smooth
# curve on the binned scatter in the same units as the binned points
rdf["rx2"] = rdf["rx"] ** 2
rdf["rx3"] = rdf["rx"] ** 3
curve_fit = smf.ols("ry ~ rx + rx2 + rx3", data=rdf).fit()
grid = np.linspace(rdf["rx"].quantile(0.01), rdf["rx"].quantile(0.99), 200)
curve_y = curve_fit.predict(pd.DataFrame({"rx": grid, "rx2": grid**2, "rx3": grid**3}))
curve = [{"x": float(a), "y": float(b)} for a, b in zip(grid, curve_y)]

# ---------- (3) implied marginal effect, dy/d(dose), from the clustered raw-scale cubic fit ----------
b1, b2, b3 = cubic.params["d_dist_mi"], cubic.params["d2"], cubic.params["d3"]
cov = cubic.cov_params().loc[["d_dist_mi", "d2", "d3"], ["d_dist_mi", "d2", "d3"]].values
dose_grid = np.linspace(s["d_dist_mi"].quantile(0.01), s["d_dist_mi"].quantile(0.99), 200)
marg = []
for x in dose_grid:
    m = b1 + 2 * b2 * x + 3 * b3 * x**2
    grad = np.array([1.0, 2 * x, 3 * x**2])
    se = float(np.sqrt(grad @ cov @ grad))
    marg.append({"x": float(x), "m": float(m), "se": se, "lo": float(m - 1.96 * se), "hi": float(m + 1.96 * se)})

out = {
    "n": int(s.shape[0]),
    "n_counties": int(s["fips5"].nunique()),
    "linear": {"coef": float(lin.params["d_dist_mi"]), "se": float(lin.bse["d_dist_mi"]), "p": float(lin.pvalues["d_dist_mi"])},
    "cubic": {
        "d1": {"coef": float(cubic.params["d_dist_mi"]), "se": float(cubic.bse["d_dist_mi"]), "p": float(cubic.pvalues["d_dist_mi"])},
        "d2": {"coef": float(cubic.params["d2"]), "se": float(cubic.bse["d2"]), "p": float(cubic.pvalues["d2"])},
        "d3": {"coef": float(cubic.params["d3"]), "se": float(cubic.bse["d3"]), "p": float(cubic.pvalues["d3"])},
    },
    "linearity_test": {"F": float(f_test.fvalue), "p": joint_p},
    "binned": binned.to_dict(orient="records"),
    "curve": curve,
    "marginal": marg,
}
with open("data/dose_response_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved data/dose_response_result.json")
