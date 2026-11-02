"""Border vs. interior trend lines, plotted SEPARATELY (not as a difference), for all three
market-age event studies: revenue per store (already done in make_bordercontrol_trend_figure.py,
reproduced here for consistency), revenue per capita, and land value (which also includes genuine
pre-launch years, unlike the two revenue outcomes)."""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')


def two_line_plot(panel, event_col, border_col, y_col, resid_formula, out_path, title, xlabel, ylabel,
                   x_tick_labels=None, launch_vline=None):
    panel = panel.copy()
    panel['y_adj'] = smf.ols(resid_formula, data=panel).fit().resid
    summary = panel.groupby([event_col, border_col]).agg(
        y_mean=('y_adj', 'mean'), y_se=('y_adj', lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan),
        n=('y_adj', 'size')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for bval, label, color, marker in [(0, 'Interior (control)', 'steelblue', 'o'),
                                        (1, 'Border (treatment)', 'firebrick', 's')]:
        sub = summary[summary[border_col] == bval].sort_values(event_col)
        x = sub[event_col].to_numpy()
        y = sub['y_mean'].to_numpy()
        yerr = 1.96 * sub['y_se'].to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt=f'{marker}-', color=color, ecolor=color, alpha=0.85,
                    capsize=3, markersize=5, linewidth=1.3, label=label)
    if launch_vline is not None:
        ax.axvline(launch_vline, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    xt = sorted(summary[event_col].unique())
    ax.set_xticks(xt)
    if x_tick_labels:
        ax.set_xticklabels([x_tick_labels.get(v, str(int(v))) for v in xt], rotation=0)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return summary


# ---------- Revenue per store (reproduced for consistency) ----------
rp = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_border70_panel.csv'), dtype={'fips5': str})
launch_offset = {'CO': 2014, 'WA': 2015}
rp['event_year'] = rp.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)
rp['event_year_bin'] = rp['event_year'].clip(upper=9)
rp.loc[rp['event_year_bin'] >= 9, 'event_year_bin'] = 9
two_line_plot(
    rp, 'event_year_bin', 'border_70mi', 'log_revenue_per_store',
    'log_revenue_per_store ~ log_population + C(state_abbr)',
    os.path.join(FIGURES, 'fig6_border_vs_interior_trends_perstore.png'),
    "Revenue per store: border vs. interior trends, separately (CO launched 2014, WA 2015, stacked)",
    "Years since state's own retail launch", 'log(revenue/store), adj. for log(pop) + state',
    x_tick_labels={9: '9+'},
)

# ---------- Revenue per capita ----------
rc = pd.read_csv(os.path.join(BASE, 'border revenue', 'data', 'county_revenue_panel.csv'), dtype={'fips5': str})
rc['fips5'] = rc['fips5'].str.zfill(5)
rc = rc[rc['revenue_per_capita'] > 0].copy()
rc['log_population'] = np.log(rc['population'])
rc['border_70mi'] = (rc['dist_nonlegal_mi'] <= 70).astype(int)
rc['event_year'] = rc.apply(lambda r: int(r['fy_year']) - launch_offset[r['state_abbr']], axis=1)
rc['event_year_bin'] = rc['event_year'].clip(upper=9)
rc.loc[rc['event_year_bin'] >= 9, 'event_year_bin'] = 9
two_line_plot(
    rc, 'event_year_bin', 'border_70mi', 'log_revenue_per_capita',
    'log_revenue_per_capita ~ log_population + C(state_abbr)',
    os.path.join(FIGURES, 'fig7_border_vs_interior_trends_percapita.png'),
    "Revenue per capita: border vs. interior trends, separately (CO launched 2014, WA 2015, stacked)",
    "Years since state's own retail launch", 'log(revenue/capita), adj. for log(pop) + state',
    x_tick_labels={9: '9+'},
)

# ---------- Land value (has genuine pre-launch years) ----------
lv = pd.read_csv(os.path.join(TABLES, 'land_value_border_panel.csv'), dtype={'fips5': str})
legal = pd.read_csv(os.path.join(DATA, 'legal_dates_new.csv'), parse_dates=['retail_date'])
launch_year = dict(zip(legal['state_abbr'], legal['retail_date'].dt.year))
lv['launch_year'] = lv['state_abbr'].map(launch_year)
lv['event_year'] = lv['year'] - lv['launch_year']
lv['event_year_bin'] = lv['event_year'].clip(lower=-4, upper=8)
two_line_plot(
    lv, 'event_year_bin', 'border_70mi', 'log_landvalue',
    'log_landvalue ~ log_population + log_land_area + C(state_abbr)',
    os.path.join(FIGURES, 'fig8_border_vs_interior_trends_landvalue.png'),
    "Land value: border vs. interior trends, separately, including real pre-launch years",
    "Years relative to state's own retail launch", 'log(land value/acre), adj. for log(pop) + land area + state',
    x_tick_labels={-4: '−4 or earlier', 8: '8+'},
    launch_vline=-0.5,
)
