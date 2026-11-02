import pandas as pd
import numpy as np
import json
from scipy import stats
from csdid.att_gt import ATTgt

np.random.seed(20260726)

panel = pd.read_csv("data/panel_twfe_cs.csv", dtype={"fips5": str})

att = ATTgt(
    yname="od_rate", tname="year", idname="fips_int", gname="g",
    data=panel, control_group=["nevertreated"], clustervar="state_abbr",
    panel=True, allow_unbalanced_panel=False, biters=1000, cband=False,
)
att.fit()
simple = att.aggte(typec="simple")

print([a for a in dir(simple) if not a.startswith("_")])
cs_att = float(simple.atte["overall_att"])
cs_se = float(np.asarray(simple.atte["overall_se"]).flatten()[0])
z = cs_att / cs_se
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"CS simple ATT: {cs_att:.4f}  se={cs_se:.4f}  z={z:.3f}  p={p:.4f}")

dyn = att.aggte(typec="dynamic")
d = dyn.atte
print("\nDynamic event-time keys:", list(d.keys()))
egt = np.asarray(d["egt"]).flatten()
att_egt = np.asarray(d["att_egt"]).flatten()
se_egt = np.asarray(d["se_egt"]).flatten()
dyn_table = []
for e, a, s in zip(egt, att_egt, se_egt):
    zz = a / s
    pp = 2 * (1 - stats.norm.cdf(abs(zz)))
    dyn_table.append({"event_time": int(e), "att": float(a), "se": float(s), "p": float(pp)})
    print(f"  t={int(e):+d}: att={a:+.4f} se={s:.4f} p={pp:.4f}")

with open("data/cs_result.json", "w") as f:
    json.dump({"cs_att": cs_att, "cs_se": cs_se, "cs_p": p, "dynamic": dyn_table}, f, indent=2)
