"""Plot border vs. interior (control) counties' revenue-per-store trends SEPARATELY over market age,
stacking CO (launched 2014) and WA (launched 2015) on a common market-age clock. This shows the
control group's own trajectory directly -- the closest thing to a pre-trend check available given
neither state has any pre-launch revenue data (see analysis discussion)."""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')

panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
launch_offset = {'CO': 2014, 'WA': 2015}
panel['event_year'] = panel.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)
panel['event_year_bin'] = panel['event_year'].clip(upper=9)
panel.loc[panel['event_year_bin'] >= 9, 'event_year_bin'] = 9

# control for population and state level differences (so CO's and WA's own baseline revenue
# levels don't dominate the plot), leaving year and border/interior variation intact
panel['g'] = pd.Categorical(panel['fips5']).codes
resid_formula = 'log_revenue_per_store ~ log_population + C(state_abbr)'
panel['y_adj'] = smf.ols(resid_formula, data=panel).fit().resid

summary = panel.groupby(['event_year_bin', 'border_70mi']).agg(
    y_mean=('y_adj', 'mean'), y_se=('y_adj', lambda x: x.std() / np.sqrt(len(x))), n=('y_adj', 'size')
).reset_index()

fig, ax = plt.subplots(figsize=(8, 4.5))
for border_val, label, color, marker in [(0, 'Interior (control)', 'steelblue', 'o'),
                                           (1, 'Border ($\\leq$70mi, treatment)', 'firebrick', 's')]:
    sub = summary[summary['border_70mi'] == border_val].sort_values('event_year_bin')
    x = sub['event_year_bin'].to_numpy()
    y = sub['y_mean'].to_numpy()
    yerr = 1.96 * sub['y_se'].to_numpy()
    ax.errorbar(x, y, yerr=yerr, fmt=f'{marker}-', color=color, ecolor=color, alpha=0.85,
                capsize=3, markersize=5, linewidth=1.3, label=label)

ax.set_xlabel("Years since state's own retail launch (CO launched 2014, WA launched 2015, stacked on a common clock)")
ax.set_ylabel('log(revenue per store), adjusted for log(pop) and state level')
ax.set_title("Border vs. interior counties' revenue-per-store trends, separately\n"
             "(control group's own trajectory is the closest available pre-trend check -- no county-level\n"
             "revenue data exists for either state before its own retail launch, see Section on data limits)",
             fontsize=10)
ax.set_xticks(sorted(summary['event_year_bin'].unique()))
ax.set_xticklabels(['9+' if v == 9 else str(int(v)) for v in sorted(summary['event_year_bin'].unique())])
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
out_path = os.path.join(FIGURES, 'fig6_border_vs_interior_trends.png')
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved {out_path}")

print("\nInterior (control) trend, by market-age year:")
print(summary[summary['border_70mi'] == 0][['event_year_bin', 'y_mean', 'n']].to_string(index=False))
print("\nBorder (treatment) trend, by market-age year:")
print(summary[summary['border_70mi'] == 1][['event_year_bin', 'y_mean', 'n']].to_string(index=False))
