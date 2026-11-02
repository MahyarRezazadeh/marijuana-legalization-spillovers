import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

ranks = [1, 2, 3, 4]
hr = [0.974, 0.696, 0.848, 0.895]
se_log = [0.1065, 0.1236, 0.1487, 0.1624]  # se of the log(HR) coefficient
coef = [np.log(h) for h in hr]
lo = [np.exp(c - 1.96 * s) for c, s in zip(coef, se_log)]
hi = [np.exp(c + 1.96 * s) for c, s in zip(coef, se_log)]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(ranks, hr, yerr=[[h - l for h, l in zip(hr, lo)], [u - h for h, u in zip(hr, hi)]],
            fmt='o-', color='tab:red', capsize=4, markersize=7)
ax.axhline(1.0, color='gray', ls='--', lw=1, label='HR = 1 (no distance effect)')
ax.set_xticks(ranks)
ax.set_xticklabels(['1st\n(launch->1st)', '2nd\n(1st->2nd)', '3rd\n(2nd->3rd)', '4th\n(3rd->4th)'])
ax.set_ylabel('Hazard ratio on log(distance to nonlegal border)')
ax.set_xlabel('Entrant rank')
ax.set_title('Distance effect on entry speed, by entrant rank\n(HR < 1: closer to border = faster entry)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_hazard_by_rank.png'), dpi=150)
print('Saved fig_hazard_by_rank.png')
