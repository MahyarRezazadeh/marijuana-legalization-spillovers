import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

res = pd.read_csv("data/dispensary_robustness_results.csv")
cont = res[res["x"] == "d_dist_mi"].reset_index(drop=True)

# main-design (state-legal-county-centroid) continuous coefficients, from
# Table 2 / robustness-summary of the existing main.tex, for visual comparison
main_design = {
    "No controls (transition FE only)": (0.0128, (0.0128 - 1.96*0.0035, 0.0128 + 1.96*0.0035)),
    "+ baseline (2020) overdose rate": (0.0122, (0.0122 - 1.96*0.0034, 0.0122 + 1.96*0.0034)),
    "+ state FE": (0.0129, (0.0129 - 1.96*0.0034, 0.0129 + 1.96*0.0034)),
    "+ county FE": (0.0122, (0.0122 - 1.96*0.0053, 0.0122 + 1.96*0.0053)),
}

specs_order = ["No controls (transition FE only)", "+ baseline (2020) overdose rate",
               "+ state FE", "+ census division x transition FE", "+ state x transition FE", "+ county FE"]
cont = cont.set_index("spec").loc[specs_order].reset_index()
cont["lo"] = cont["coef"] - 1.96 * cont["se"]
cont["hi"] = cont["coef"] + 1.96 * cont["se"]

fig, ax = plt.subplots(figsize=(8, 5))
y = list(range(len(specs_order)))[::-1]

ax.errorbar(cont["coef"], [v + 0.12 for v in y], xerr=1.96 * cont["se"],
            fmt="o", color="#2c7fb8", capsize=3, label="Dispensary-distance design (this check)")

main_coefs, main_los, main_his = [], [], []
for s in specs_order:
    if s in main_design:
        c, (lo, hi) = main_design[s]
        main_coefs.append(c); main_los.append(c - lo); main_his.append(hi - c)
    else:
        main_coefs.append(None); main_los.append(None); main_his.append(None)
main_y = [v - 0.12 for i, v in enumerate(y) if main_coefs[i] is not None]
main_c = [c for c in main_coefs if c is not None]
main_err = [[l for l in main_los if l is not None], [h for h in main_his if h is not None]]
ax.errorbar(main_c, main_y, xerr=main_err, fmt="s", color="#969696", capsize=3,
            label="Main design (state-legal-county distance)")

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(y)
ax.set_yticklabels(specs_order)
ax.set_xlabel("Coefficient on distance-to-nearest-legal-market (per mile), effect on Δ overdose deaths/100k")
ax.set_title("Continuous distance coefficient: dispensary-date design vs. main (state-date) design")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("figures/figG_dispensary_vs_main_coefplot.png", dpi=200)
print("Saved figures/figG_dispensary_vs_main_coefplot.png")
