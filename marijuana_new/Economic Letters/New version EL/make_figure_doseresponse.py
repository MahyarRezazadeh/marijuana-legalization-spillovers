import json
import matplotlib.pyplot as plt

with open("data/dose_response_result.json") as f:
    res = json.load(f)

binned = res["binned"]
curve = res["curve"]
marg = res["marginal"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
cx = [c["x"] for c in curve]
cy = [c["y"] for c in curve]
ax.plot(cx, cy, color="#2c7fb8", linewidth=2, zorder=2, label="Cubic fit")
bx = [b["x_mean"] for b in binned]
by = [b["y_mean"] for b in binned]
bse = [1.96 * b["y_se"] for b in binned]
ax.errorbar(bx, by, yerr=bse, fmt="o", color="#252525", ecolor="#969696", capsize=3,
            markersize=5, zorder=3, label="Binned means (20 equal-N bins)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Change in distance to nearest legal-market\ncounty (miles), residualized on county FE,\ntransition FE, baseline overdose rate")
ax.set_ylabel("Change in overdose deaths per 100,000\n(residualized)")
ax.set_title("(a) Dose-response function")
ax.legend(loc="best", fontsize=8)

ax2 = axes[1]
mx = [m["x"] for m in marg]
mm = [m["m"] for m in marg]
mlo = [m["lo"] for m in marg]
mhi = [m["hi"] for m in marg]
ax2.plot(mx, mm, color="#d95f02", linewidth=2)
ax2.fill_between(mx, mlo, mhi, color="#d95f02", alpha=0.2, label="95% CI")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Change in distance to nearest legal-market\ncounty (miles), raw dose")
ax2.set_ylabel("Marginal effect on overdose deaths\nper 100,000 per mile")
ax2.set_title("(b) Implied marginal effect")
ax2.legend(loc="best", fontsize=8)

fig.suptitle("Dose-response function: distance change and overdose mortality")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figures/figA_doseresponse_multiperiod.png", dpi=200)
print("Saved figures/figA_doseresponse_multiperiod.png")
