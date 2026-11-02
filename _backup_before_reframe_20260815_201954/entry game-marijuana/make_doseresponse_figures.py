"""Dose-response binned-scatter plots for the two revenue-per-store mechanism checks:
(1) log(distance to non-legal border), (2) log(captive non-legal population within 70mi).
Standard binscatter approach: partial out controls (Frisch-Waugh-Lovell) via residualization,
bin the residualized x-variable into deciles, plot mean residualized y vs mean residualized x
per bin, overlay the fitted regression line from the actual (unbinned) regression."""
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
os.makedirs(FIGURES, exist_ok=True)


def binscatter(df, xvar, controls_formula_rhs, yvar, n_bins, title, xlabel, ylabel, out_path, exclude_sedgwick_note=False):
    d = df.dropna(subset=[xvar, yvar]).copy()
    # residualize y and x on the same controls (Frisch-Waugh-Lovell)
    y_resid = smf.ols(f'{yvar} ~ {controls_formula_rhs}', data=d).fit().resid
    x_resid = smf.ols(f'{xvar} ~ {controls_formula_rhs}', data=d).fit().resid
    d['y_resid'] = y_resid
    d['x_resid'] = x_resid

    d['bin'] = pd.qcut(d['x_resid'], q=n_bins, duplicates='drop')
    binned = d.groupby('bin', observed=True).agg(x_mean=('x_resid', 'mean'), y_mean=('y_resid', 'mean'),
                                                    n=('x_resid', 'size')).reset_index()

    # fitted line from the actual regression coefficient on the residualized relationship
    slope, intercept = np.polyfit(d['x_resid'], d['y_resid'], 1)
    xs = np.linspace(d['x_resid'].min(), d['x_resid'].max(), 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(binned['x_mean'], binned['y_mean'], s=binned['n'] * 3, color='black', zorder=3,
               label=f'{n_bins}-bin means (size = obs. count)')
    ax.plot(xs, slope * xs + intercept, color='firebrick', linewidth=1.5, zorder=2,
            label=f'fitted slope = {slope:+.3f}')
    ax.axhline(0, color='gray', linewidth=0.6)
    ax.axvline(0, color='gray', linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path} (slope={slope:+.4f}, n={len(d)})")


panel = pd.read_csv(os.path.join(TABLES, 'revenue_perstore_captivepop_panel.csv'), dtype={'fips5': str})
controls = 'log_population + C(fy_year) + C(state_abbr)'

binscatter(
    panel, 'log_dist_nonlegal', controls, 'log_revenue_per_store', n_bins=10,
    title='Dose-response: revenue per store vs. distance to non-legal border\n(residualized on log(pop), year FE, state FE; 10-bin means)',
    xlabel='log(distance to nearest non-legal county), residualized',
    ylabel='log(revenue per store), residualized',
    out_path=os.path.join(FIGURES, 'fig3_doseresponse_distance.png'),
)

binscatter(
    panel, 'log_captive_pop', controls, 'log_revenue_per_store', n_bins=10,
    title='Dose-response: revenue per store vs. captive non-legal population within 70mi\n(residualized on log(pop), year FE, state FE; 10-bin means)',
    xlabel='log(captive non-legal population, 70mi), residualized',
    ylabel='log(revenue per store), residualized',
    out_path=os.path.join(FIGURES, 'fig4_doseresponse_captivepop.png'),
)
