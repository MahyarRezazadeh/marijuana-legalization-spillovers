"""Event-study coefficient plots (with 95% CI, county-clustered SEs) for both
land value (with pre-launch bins) and revenue-per-store market-age event studies."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')
os.makedirs(FIGURES, exist_ok=True)


def plot_eventstudy(csv_path, out_path, title, xlabel, drop_labels=None, x_order=None):
    df = pd.read_csv(csv_path)
    if drop_labels:
        df = df[~df['event_year'].isin(drop_labels)]
    if x_order is None:
        x_order = df['event_year'].tolist()
    df = df.set_index('event_year').loc[x_order].reset_index()

    x = np.arange(len(df))
    coef = df['coef'].to_numpy()
    se = df['se_county_cl'].to_numpy()
    ci = 1.96 * se

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
    if '0' in x_order:
        zero_idx = x_order.index('0')
        ax.axvline(zero_idx - 0.5, color='gray', linewidth=0.8, linestyle='--')
    ax.errorbar(x, coef, yerr=ci, fmt='o', color='black', ecolor='gray', elinewidth=1.2,
                capsize=3, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(df['event_year'], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Border $\\times$ event-time coefficient')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# Land value: full event-year order including pre-launch bins, drop degenerate (unidentified) bins
lv_order = ['-4 or earlier', '-3', '-2', '-1', '0', '2', '3', '4', '7', '8+']
plot_eventstudy(
    os.path.join(TABLES, 'land_value_border_eventstudy_results.csv'),
    os.path.join(FIGURES, 'fig1_landvalue_eventstudy.png'),
    'Land value: border $\\times$ market-age coefficients (with 95% CI)\n'
    'Dashed line = state\'s own retail launch; years 1 and 5 omitted (unidentified, single-state bins)',
    "Years relative to state's own retail launch",
    drop_labels=['1', '5'],
    x_order=lv_order,
)

# Revenue per store: no pre-launch data exists (no legal sales before launch)
rev_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9+']
plot_eventstudy(
    os.path.join(TABLES, 'revenue_perstore_eventstudy_results.csv'),
    os.path.join(FIGURES, 'fig2_revenue_perstore_eventstudy.png'),
    'Revenue per store: border $\\times$ market-age coefficients (with 95% CI)\n'
    'No pre-launch bins exist (no legal cannabis sales before a state\'s own retail market opens)',
    "Years since state's own retail launch",
    x_order=rev_order,
)

# Revenue per capita: same no-pre-launch structure, clean null throughout
plot_eventstudy(
    os.path.join(TABLES, 'revenue_percapita_eventstudy_results.csv'),
    os.path.join(FIGURES, 'fig5_revenue_percapita_eventstudy.png'),
    'Revenue per capita: border $\\times$ market-age coefficients (with 95% CI)\n'
    'No pre-launch bins exist; shown for comparison with the per-store premium in Figure 2',
    "Years since state's own retail launch",
    x_order=rev_order,
)
