# ============================================================
# Two questions the original entry_hazard analysis could not answer,
# both required before writing up "the second entrant" as an
# economically distinct object rather than just describing a single
# regression coefficient:
#
# (1) IS THE EFFECT SPECIFIC TO ENTRANT #2? Compare border-proximity
#     hazard ratios across the 1st, 2nd, 3rd, and 4th entrant gaps. If
#     the effect is roughly constant or monotonically increasing across
#     all ranks, that is a generic "more agglomeration nearby -> always
#     faster entry" story, not something special about the SECOND
#     entrant per se. If it peaks or is only present at rank 2, that
#     supports a genuinely distinct second-entrant mechanism.
#
# (2) IS THE 2ND-ENTRANT EFFECT ABOUT CROSS-BORDER DEMAND SPECIFICALLY,
#     OR JUST RAW DISTANCE? Re-estimate the 2nd-entrant hazard with
#     log(captive nonlegal population within 70mi) in place of / added
#     to log(distance). If the effect is really about a captive
#     cross-border demand pool (not merely "there happens to be some
#     other state nearby, regardless of its population"), the
#     captive-population measure should do at least as well.
# ============================================================
import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

BASE = os.path.dirname(os.path.abspath(__file__))
panel = pd.read_csv(os.path.join(BASE, 'entry_hazard_panel_v2.csv'))

results = []

def run_cox(df, duration_col, event_col, covariates, label, strata=('state_abbr',)):
    cols = [duration_col, event_col] + list(strata) + covariates
    d = df[cols].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=duration_col, event_col=event_col, strata=list(strata))
    print(f"\n=== {label} (n={len(d)}, events={int(d[event_col].sum())}) ===")
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].round(4).to_string())
    for cov in covariates:
        results.append(dict(model=label, covariate=cov, coef=cph.summary.loc[cov, 'coef'],
                             hazard_ratio=cph.summary.loc[cov, 'exp(coef)'],
                             se=cph.summary.loc[cov, 'se(coef)'], p=cph.summary.loc[cov, 'p'],
                             n=len(d), events=int(d[event_col].sum())))
    return cph

print("#" * 70)
print("PART 1: hazard ratio on log(distance) across entrant rank 1,2,3,4")
print("#" * 70)

run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 1 (state-launch -> 1st entrant)')

r2 = panel[panel['n_timed_entrants'] >= 1]
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 2 (1st -> 2nd entrant)')

r3 = panel[panel['n_timed_entrants'] >= 2]
run_cox(r3, 'gap_2to3_time', 'gap_2to3_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 3 (2nd -> 3rd entrant)')

r4 = panel[panel['n_timed_entrants'] >= 3]
run_cox(r4, 'gap_3to4_time', 'gap_3to4_event',
        ['log_dist_nonlegal', 'log_population'], 'Rank 4 (3rd -> 4th entrant)')

# same with the binary border measure, for comparability with the original table
run_cox(panel, 'first_entry_time', 'first_entry_event',
        ['border_70mi', 'log_population'], 'Rank 1 (binary border)')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['border_70mi', 'log_population'], 'Rank 2 (binary border)')
run_cox(r3, 'gap_2to3_time', 'gap_2to3_event',
        ['border_70mi', 'log_population'], 'Rank 3 (binary border)')
run_cox(r4, 'gap_3to4_time', 'gap_3to4_event',
        ['border_70mi', 'log_population'], 'Rank 4 (binary border)')

print("\n" + "#" * 70)
print("PART 2: is the 2nd-entrant effect about captive cross-border")
print("population specifically, or just raw distance to any nonlegal county?")
print("#" * 70)

run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_captive_pop', 'log_population'], 'Rank 2, log(captive nonlegal population)')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['has_captive_pop', 'log_population'], 'Rank 2, has_captive_pop (binary, >0 within 70mi)')
run_cox(r2, 'second_gap_time', 'second_gap_event',
        ['log_dist_nonlegal', 'log_captive_pop', 'log_population'],
        'Rank 2, distance AND captive population jointly')

# subsample: among border counties only (border_70mi==1), does captive
# population size (not just being a border county) predict 2nd-entrant speed?
r2_border = r2[r2['border_70mi'] == 1]
print(f"\nBorder-only subsample for rank-2 gap: n={len(r2_border)} counties, "
      f"of which {r2_border['has_captive_pop'].sum()} have any captive population")
try:
    run_cox(r2_border, 'second_gap_time', 'second_gap_event',
            ['log_captive_pop', 'log_population'], 'Rank 2, BORDER COUNTIES ONLY, log(captive population)')
except Exception as e:
    print(f"[skipped -- {e}]")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(BASE, 'entry_hazard_by_rank_and_captivepop.csv'), index=False)
print("\nSaved entry_hazard_by_rank_and_captivepop.csv")
print(res_df.round(4).to_string(index=False))
