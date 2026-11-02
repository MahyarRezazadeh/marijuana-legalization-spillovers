# ============================================================
# Final welfare numbers for the Economic Inquiry draft, using the
# mean-reversion-adjusted (baseline overdose-level control)
# coefficients that survive the pre-trends check, applied to the
# SAME switcher-county population base as the original calculation
# (keeps the population scope interpretable and comparable).
# ============================================================
import os, json
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

main_ctrl = pd.read_csv(os.path.join(DATA, 'main_with_controls.csv'))
sw_adj = main_ctrl[(main_ctrl['x'] == 'switcher') & (main_ctrl['spec'] == '+ baseline overdose level')].iloc[0]
b, se = sw_adj['coef'], sw_adj['se']
print(f"Mean-reversion-adjusted switcher coefficient: b={b:.4f} se={se:.4f} p={sw_adj['p']:.4f}")

with open(os.path.join(DATA, 'welfare_summary_ei.json')) as f:
    old = json.load(f)
pop = old['switcher_population']
le_diff = old['le_diff']

def averted(coef):
    return -coef * pop / 1e5

pt = averted(b)
lo = averted(b + 1.96 * se)
hi = averted(b - 1.96 * se)
yll = pt * le_diff

print(f"Switcher population (unchanged): {pop:,.0f}")
print(f"Averted deaths (mean-reversion-adjusted): {pt:.1f} [95% CI: {lo:.1f}, {hi:.1f}]")
print(f"Years of life saved: {yll:,.0f} [{lo*le_diff:,.0f}, {hi*le_diff:,.0f}]")

out = dict(coef=b, se=se, p=float(sw_adj['p']), switcher_population=pop,
           averted_deaths=pt, averted_lo=lo, averted_hi=hi,
           le_diff=le_diff, yll=yll, yll_lo=lo*le_diff, yll_hi=hi*le_diff)
with open(os.path.join(DATA, 'welfare_summary_ei_adjusted.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved welfare_summary_ei_adjusted.json")
