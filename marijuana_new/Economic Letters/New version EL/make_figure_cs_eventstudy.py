import json
import matplotlib.pyplot as plt

with open("data/twfe_cs_result_t50.json") as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

panels = [
    (axes[0], res["cs_nocov"]["dynamic"], "No covariates"),
    (axes[1], res["cs_cov"]["dynamic"], "+ baseline-rate covariate"),
]

for ax, dyn, subtitle in panels:
    et = [d["event_time"] for d in dyn]
    att = [d["att"] for d in dyn]
    se = [d["se"] for d in dyn]
    ci = [1.96 * s for s in se]
    ax.errorbar(et, att, yerr=ci, fmt="o", color="#2c7fb8", ecolor="#2c7fb8", capsize=4, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Event time (years since crossing the 50-mile threshold)")
    ax.set_title(subtitle)
    ax.set_xticks(et)

axes[0].set_ylabel("ATT: overdose deaths per 100,000")
fig.suptitle("Callaway–Sant'Anna event-study coefficients (never-treated controls)")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figures/figH_cs_eventstudy.png", dpi=200)
print("Saved figures/figH_cs_eventstudy.png")
