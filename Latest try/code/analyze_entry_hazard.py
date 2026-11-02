# ============================================================
# Cox proportional-hazards models: do border counties enter FASTER
# after their state's retail market opens (racing to capture cross-
# border demand), and does a SECOND entrant follow faster once a
# border county's first mover reveals the opportunity (a literal
# "competed for it" test)? Distinct from the existing static entry-
# density/count test, which is null but cannot speak to timing.
# State fixed effects via Cox strata (matches the state-FE convention
# used throughout analyze_entry_density.py).
# ============================================================
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'entry_hazard_panel.csv'))

results = []

def run_cox(df, duration_col, event_col, covariates, label):
    d = df[[duration_col, event_col, 'state_abbr'] + covariates].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=duration_col, event_col=event_col, strata=['state_abbr'])
    print(f"\n=== {label} (n={len(d)}, events={int(d[event_col].sum())}) ===")
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].round(4).to_string())
    for cov in covariates:
        results.append(dict(
            model=label, covariate=cov,
            coef=cph.summary.loc[cov, 'coef'], hazard_ratio=cph.summary.loc[cov, 'exp(coef)'],
            se=cph.summary.loc[cov, 'se(coef)'], p=cph.summary.loc[cov, 'p'],
            n=len(d), events=int(d[event_col].sum())))
    return cph

# ---------- First-entry hazard: border_70mi (binary) ----------
run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['border_70mi', 'log_population'], 'First entry, border_70mi (binary)')

# ---------- First-entry hazard: continuous log-distance ----------
run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['log_dist_nonlegal', 'log_population'], 'First entry, log(distance to nonlegal border)')

# ---------- Second-entrant hazard (gap from 1st to 2nd entrant) ----------
second_sample = panel[panel['first_entry_event'] == 1].copy()
run_cox(second_sample, 'second_gap_time', 'second_gap_event',
        ['border_70mi', 'log_population'], 'Second entrant gap, border_70mi (binary)')
run_cox(second_sample, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_population'], 'Second entrant gap, log(distance to nonlegal border)')

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(BASE, 'entry_hazard_results.csv'), index=False)
print("\nSaved entry_hazard_results.csv")
print(res_df.round(4).to_string(index=False))

# ---------- Kaplan-Meier figure: first entry, border vs interior ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (dur, ev, title) in zip(axes, [
        ('first_entry_time', 'first_entry_event', 'Time to FIRST dispensary\n(days since state retail launch)'),
        ('second_gap_time', 'second_gap_event', 'Time from 1st to 2nd dispensary\n(days, among counties with a 1st entrant)')]):
    sample = panel if dur == 'first_entry_time' else second_sample
    for grp, label, color in [(1, 'border (<=70mi)', 'tab:red'), (0, 'interior', 'tab:blue')]:
        kmf = KaplanMeierFitter()
        mask = sample['border_70mi'] == grp
        kmf.fit(sample.loc[mask, dur], sample.loc[mask, ev], label=label)
        kmf.plot_survival_function(ax=ax, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Days')
    ax.set_ylabel('Share not yet entered')
fig.suptitle('Entry timing, border vs. interior counties (Kaplan-Meier)')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'fig_entry_hazard_km.png'), dpi=150)
print("Saved fig_entry_hazard_km.png")
