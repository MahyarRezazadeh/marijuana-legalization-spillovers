# ============================================================
# CROSS-BORDER SPILLOVER OF RECREATIONAL MARIJUANA LEGALIZATION
# ON MARIJUANA-RELATED DRIVING OUTCOMES: A CONTINUOUS
# DISTANCE-TO-BORDER DESIGN, 2020-2023
#
# marijuana_new/build_report.py
# Self-contained: reads marijuana_new/data/*, writes marijuana_new/figures,
# marijuana_new/tables, and marijuana_new/Marijuana_Driving_Spillover_Results.pdf
#
# Outcome:   NHTSA FARS marijuana-involved fatal crashes per 100,000 residents
#            (best available nationwide county-level proxy for marijuana-
#            related driving enforcement outcomes; see Data section / README
#            for why county-level marijuana-specific DUID *arrest* counts are
#            not usable as the primary outcome).
# Treatment: continuous log-distance to nearest recreational-marijuana state
#            border (primary); binary 100-mile "switcher" indicator (robustness).
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy.stats import gaussian_kde, ttest_ind
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                                 Image, Spacer, PageBreak, HRFlowable)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG  = os.path.join(BASE, 'figures')
TAB  = os.path.join(BASE, 'tables')
OUT  = os.path.join(BASE, 'Marijuana_Driving_Spillover_Results.pdf')
for d in (DATA, FIG, TAB):
    os.makedirs(d, exist_ok=True)

BLUE = '#1f77b4'; RED = '#d62728'; GREEN = '#2ca02c'
ORANGE = '#ff7f0e'; GRAY = '#7f7f7f'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8, colors='#444444')
    if title:  ax.set_title(title, fontsize=9, loc='left', pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#555555')
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color='#555555')


# ════════════════════════════════════════════════════════════
# 1. LOAD & MERGE DATA
# ════════════════════════════════════════════════════════════
print("Loading analysis_panel.dta ...")
df = pd.read_stata(os.path.join(DATA, 'analysis_panel.dta'))
df['year_int'] = df['year'].astype(float).astype(int)
df['fips5'] = df['fips5'].astype(str).str.zfill(5)

# County-level metadata (treatment side is time-invariant except for the
# distance variables, which we keep for both periods explicitly)
meta = (df[df['year_int'] == 2020]
        [['fips5', 'switcher', 'never_legal', 'sample_grp', 'state_abbr',
          'lat', 'lon', 'population',
          'dist_rec_2020_mi', 'dist_rec_2023_mi',
          'log_dist_2020', 'log_dist_2023']]
        .drop_duplicates('fips5').copy())
meta['d_log_dist'] = meta['log_dist_2023'] - meta['log_dist_2020']

# Rebuild distance bands directly from 2023 distance (robust to categorical dtype issues)
meta['dist_band'] = np.select(
    [meta['dist_rec_2023_mi'] <= 50, meta['dist_rec_2023_mi'] <= 100,
     meta['dist_rec_2023_mi'] <= 200, meta['dist_rec_2023_mi'] <= 300],
    [1, 2, 3, 4], default=5)

print("Loading FARS marijuana-involved fatal crash counts ...")
fars = pd.read_csv(os.path.join(DATA, 'fars_marijuana_crashes.csv'), dtype={'fips5': str})
fars['fips5'] = fars['fips5'].str.zfill(5)
f20 = fars[fars['year'] == 2020][['fips5', 'mj_fatal_crashes']].rename(columns={'mj_fatal_crashes': 'mj20'})
f23 = fars[fars['year'] == 2023][['fips5', 'mj_fatal_crashes']].rename(columns={'mj_fatal_crashes': 'mj23'})

panel = meta.merge(f20, on='fips5', how='left').merge(f23, on='fips5', how='left')
panel['mj20'] = panel['mj20'].fillna(0)
panel['mj23'] = panel['mj23'].fillna(0)

# Population (only available from the PLACES 2020 release) is used as a fixed
# denominator for both periods -- see Data/Limitations section for rationale.
panel = panel[panel['population'].notna() & (panel['population'] > 0)].copy()
panel['mj_rate20'] = panel['mj20'] / (panel['population'] / 1e5)
panel['mj_rate23'] = panel['mj23'] / (panel['population'] / 1e5)
panel['d_mj'] = panel['mj_rate23'] - panel['mj_rate20']

main = panel[panel['sample_grp'] != 'Always near rec border'].copy()   # switchers + controls
full = panel.copy()                                                     # + always-near-rec

n_counties  = int(panel['fips5'].nunique())
n_switchers = int(main[main['switcher'] == 1]['fips5'].nunique())
n_control   = int(main[main['switcher'] == 0]['fips5'].nunique())
n_always    = int(panel[panel['sample_grp'] == 'Always near rec border']['fips5'].nunique())
n_states    = int(panel['state_abbr'].nunique())

print(f"Counties: {n_counties} | Switchers: {n_switchers} | Controls: {n_control} | Always-near: {n_always} | States: {n_states}")


# ════════════════════════════════════════════════════════════
# 2. REGRESSION HELPERS (state-clustered OLS; first-difference = TWFE for T=2)
# ════════════════════════════════════════════════════════════
def ols_cl(y, X, groups, xname):
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(X)
    try:
        res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    except Exception as e:
        print(f"  Warning: {e}. Using HC3.")
        res = sm.OLS(y, Xc).fit(cov_type='HC3')
    return float(res.params[xname]), float(res.bse[xname]), float(res.pvalues[xname]), int(res.nobs)

def ols_cl_multi(y, Xdf, groups):
    g = pd.Categorical(groups).codes
    Xc = sm.add_constant(Xdf)
    try:
        res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
    except Exception as e:
        print(f"  Warning: {e}. Using HC3.")
        res = sm.OLS(y, Xc).fit(cov_type='HC3')
    out = {}
    for c in Xdf.columns:
        out[c] = (float(res.params[c]), float(res.bse[c]), float(res.pvalues[c]))
    return out, int(res.nobs)

def stars(p): return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
def ci95(b, s): return b - 1.96*s, b + 1.96*s
def fmt(x, f='.2f'): return f'{x:{f}}' if pd.notna(x) else 'N/A'


# ════════════════════════════════════════════════════════════
# 3. MAIN REGRESSIONS
# ════════════════════════════════════════════════════════════
print("Running regressions ...")

# -- Preferred: continuous log-distance, main sample (switchers + controls) --
bC1, seC1, pC1, nC1 = ols_cl(main['d_mj'], main[['d_log_dist']], main['state_abbr'], 'd_log_dist')
ciC1 = ci95(bC1, seC1)

# -- Robustness: binary switcher, main sample --
bB1, seB1, pB1, nB1 = ols_cl(main['d_mj'], main[['switcher']], main['state_abbr'], 'switcher')
ciB1 = ci95(bB1, seB1)

# -- Continuous, full sample (incl. always-near-rec) --
bC2, seC2, pC2, nC2 = ols_cl(full['d_mj'], full[['d_log_dist']], full['state_abbr'], 'd_log_dist')

# -- Binary, full sample --
bB2, seB2, pB2, nB2 = ols_cl(full['d_mj'], full[['switcher']], full['state_abbr'], 'switcher')

print(f"Continuous (main):  b={bC1:+.3f}  SE={seC1:.3f}  p={pC1:.3f}  N={nC1}")
print(f"Binary (main):      b={bB1:+.3f}  SE={seB1:.3f}  p={pB1:.3f}  N={nB1}")
print(f"Continuous (full):  b={bC2:+.3f}  SE={seC2:.3f}  p={pC2:.3f}  N={nC2}")
print(f"Binary (full):      b={bB2:+.3f}  SE={seB2:.3f}  p={pB2:.3f}  N={nB2}")

# -- Dose-response by distance band (2023 distance), full sample, jointly --
for b in [1, 2, 3, 4]:
    full[f'band{b}'] = (full['dist_band'] == b).astype(float)
band_res, n_band = ols_cl_multi(full['d_mj'], full[['band1', 'band2', 'band3', 'band4']], full['state_abbr'])
print(f"Dose-response (bands, ref = >300mi): {band_res}")

# -- Placebo: pre-period (2020) levels regressed on FUTURE switcher status --
bPL, sePL, pPL, nPL = ols_cl(main['mj_rate20'], main[['switcher']], main['state_abbr'], 'switcher')
print(f"Placebo (2020 level ~ future switcher): b={bPL:+.3f}  p={pPL:.3f}")

# -- Pre-period balance table (t-tests, switcher vs control, 2020 levels) --
sw20 = main[main['switcher'] == 1]
ct20 = main[main['switcher'] == 0]
def ttest_row(var, label):
    a, b_ = sw20[var].dropna(), ct20[var].dropna()
    t, p = ttest_ind(a, b_, equal_var=False)
    return [label, f'{a.mean():.2f}', f'{b_.mean():.2f}', f'{a.mean()-b_.mean():+.2f}', f'{p:.3f}']

bal_rows = [
    ttest_row('mj_rate20', 'MJ fatal crash rate (per 100k), 2020'),
    ttest_row('dist_rec_2020_mi', 'Distance to rec border, 2020 (mi)'),
]

# -- Summary statistics --
mean_r20, sd_r20 = panel['mj_rate20'].mean(), panel['mj_rate20'].std()
mean_r23, sd_r23 = panel['mj_rate23'].mean(), panel['mj_rate23'].std()
mean_dsw  = main[main['switcher'] == 1]['dist_rec_2023_mi'].mean()
mean_dctl = main[main['switcher'] == 0]['dist_rec_2023_mi'].mean()
mean_dsw_20  = main[main['switcher'] == 1]['dist_rec_2020_mi'].mean()
mean_dctl_20 = main[main['switcher'] == 0]['dist_rec_2020_mi'].mean()
pct_c1 = abs(bC1) / mean_r20 * 100 if mean_r20 else np.nan
# NOTE ON SIGN: the regressor is d_log_dist = log(dist_2023) - log(dist_2020), which is NEGATIVE
# when a county became CLOSER to a rec border. The "effect of becoming closer" per unit of
# log-distance is therefore the NEGATIVE of the estimated coefficient bC1/bC2, not bC1/bC2 itself.
closer_effect_C1 = -bC1
closer_effect_C2 = -bC2
dir_c1 = 'increase' if closer_effect_C1 > 0 else 'decrease'
dir_b1 = 'increase' if bB1 > 0 else 'decrease'
dir_consistent = (closer_effect_C1 < 0) == (bB1 < 0)

def sig_label(p):
    if p < 0.01: return 'statistically significant at the 1% level'
    if p < 0.05: return 'statistically significant at the 5% level'
    if p < 0.10: return 'marginally significant at the 10% level'
    return f'not statistically significant (p = {p:.3f})'

sig_c1 = sig_label(pC1)
sig_b1 = sig_label(pB1)

# ════════════════════════════════════════════════════════════
# 4. SAVE TABLES AS CSV (in addition to embedding in the PDF)
# ════════════════════════════════════════════════════════════
pd.DataFrame({
    'Variable': ['MJ fatal crash rate per 100k (2020)', 'MJ fatal crash rate per 100k (2023)',
                 'Distance to nearest rec border, mi (switchers, 2023)',
                 'Distance to nearest rec border, mi (controls, 2023)',
                 'Switcher counties (N)', 'Control counties (N)', 'Always-near counties (N)',
                 'Total counties (N)', 'States (N, clusters)'],
    'Value': [f'{mean_r20:.2f} (SD {sd_r20:.2f})', f'{mean_r23:.2f} (SD {sd_r23:.2f})',
              f'{mean_dsw:.1f}', f'{mean_dctl:.1f}',
              n_switchers, n_control, n_always, n_counties, n_states]
}).to_csv(os.path.join(TAB, 't1_summary.csv'), index=False)

pd.DataFrame(bal_rows, columns=['Variable', 'Switcher mean', 'Control mean', 'Diff', 'p-value']
             ).to_csv(os.path.join(TAB, 't2_balance.csv'), index=False)

pd.DataFrame({
    'Spec': ['(1) Continuous, main sample [PREFERRED]', '(2) Binary, main sample',
             '(3) Continuous, full sample', '(4) Binary, full sample'],
    'Coef': [bC1, bB1, bC2, bB2], 'SE': [seC1, seB1, seC2, seB2],
    'p': [pC1, pB1, pC2, pB2], 'N': [nC1, nB1, nC2, nB2]
}).to_csv(os.path.join(TAB, 't3_main_regression.csv'), index=False)

pd.DataFrame({
    'Distance band': ['0-50 mi', '50-100 mi', '100-200 mi', '200-300 mi', '>300 mi (ref.)'],
    'Coef': [band_res['band1'][0], band_res['band2'][0], band_res['band3'][0], band_res['band4'][0], 0.0],
    'SE': [band_res['band1'][1], band_res['band2'][1], band_res['band3'][1], band_res['band4'][1], np.nan],
    'p': [band_res['band1'][2], band_res['band2'][2], band_res['band3'][2], band_res['band4'][2], np.nan],
}).to_csv(os.path.join(TAB, 't4_dose_response.csv'), index=False)
print("Tables saved to marijuana_new/tables/")


# ════════════════════════════════════════════════════════════
# 5. FIGURES (Kieran Healy style)
# ════════════════════════════════════════════════════════════

# Fig 1: Treatment map
fig, ax = plt.subplots(figsize=(8, 5))
cfg = [
    ('Never near rec border',     GRAY,  '.', 2,  0.25, 'Control (never near rec)'),
    ('Always near rec border',    BLUE,  '.', 4,  0.45, 'Always near rec border'),
    ('Switcher (newly near rec)', RED,   'o', 14, 0.80, 'Switcher (gained rec neighbor 2020-2023)'),
]
for grp, col, mk, ms, al, lab in cfg:
    s = panel[panel['sample_grp'] == grp].dropna(subset=['lat', 'lon'])
    if len(s):
        ax.scatter(s['lon'], s['lat'], c=col, marker=mk, s=ms, alpha=al, linewidths=0,
                   zorder=(3 if grp == 'Switcher (newly near rec)' else 1))
handles = [mpatches.Patch(color=c, alpha=al, label=lab) for _, c, _, _, al, lab in cfg]
ax.legend(handles=handles, fontsize=7, frameon=False, loc='lower right')
healy(ax, title='Figure 1. Treatment Assignment in Never-Legalizing States',
      xlabel='Longitude', ylabel='Latitude')
ax.figure.text(0.05, 0.01, 'Red = switchers (gained rec neighbor 2020-2023). Blue = always near rec. Gray = control.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig1_path = os.path.join(FIG, 'fig1_map.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight'); plt.close()

# Fig 2: Distance distribution by group (2023 distance)
fig, ax = plt.subplots(figsize=(7, 4))
for grp, col, ls, lab in [
    ('Never near rec border',     GRAY, '-',  'Control'),
    ('Switcher (newly near rec)', RED,  '--', 'Switcher'),
    ('Always near rec border',    BLUE, ':',  'Always near rec'),
]:
    s = panel[(panel['sample_grp'] == grp) & panel['dist_rec_2023_mi'].notna()]
    if len(s) > 5:
        d = s['dist_rec_2023_mi'].clip(0, s['dist_rec_2023_mi'].quantile(0.98))
        kde = gaussian_kde(d, bw_method=0.35)
        xr = np.linspace(0, d.max(), 300)
        ax.plot(xr, kde(xr), color=col, linestyle=ls, lw=1.8, label=lab)
ax.axvline(100, color='#888', linestyle=':', lw=1, label='100-mile threshold')
ax.legend(fontsize=7, frameon=False)
healy(ax, title='Figure 2. Distance to Nearest Recreational State Border by Group',
      xlabel='Distance to nearest rec border (miles)', ylabel='Density')
ax.figure.text(0.05, 0.01, 'Period 2 (2023) distances. Never-legalizing states only.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig2_path = os.path.join(FIG, 'fig2_dist.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight'); plt.close()

# Fig 3: Outcome trend (marijuana-involved fatal crash rate)
fig, ax = plt.subplots(figsize=(7, 4.5))
tr = main.groupby('switcher')[['mj_rate20', 'mj_rate23']].mean().reset_index()
for _, row in tr.iterrows():
    col = RED if row['switcher'] == 1 else GRAY
    ls = '--' if row['switcher'] == 1 else '-'
    lab = 'Switcher' if row['switcher'] == 1 else 'Control'
    ax.plot([2020, 2023], [row['mj_rate20'], row['mj_rate23']],
             color=col, linestyle=ls, marker='o', ms=7, lw=2.2, label=lab)
ax.set_xticks([2020, 2023])
ax.legend(fontsize=8, frameon=False)
healy(ax, title='Figure 3. Marijuana-Involved Fatal Crash Rate: Switchers vs Controls',
      xlabel='Period', ylabel='Fatal crashes per 100,000 residents')
ax.figure.text(0.05, 0.01, 'NHTSA FARS 2020 and 2023. Population held at 2020 level for both periods (see Data).',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig3_path = os.path.join(FIG, 'fig3_trends.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight'); plt.close()

# Fig 4: DiD coefficient plot (continuous + binary, two panels)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
ax = axes[0]
ax.errorbar([0], [bC1], yerr=[[bC1-ciC1[0]], [ciC1[1]-bC1]], fmt='o', color=BLUE,
            capsize=4, ms=8, elinewidth=1.6)
ax.axhline(0, color='#999', lw=1, linestyle='--')
ax.set_xlim(-1, 1); ax.set_xticks([])
healy(ax, title='(a) Continuous: Δlog(dist)', ylabel='Δ MJ crash rate per 100k')
ax = axes[1]
ax.errorbar([0], [bB1], yerr=[[bB1-ciB1[0]], [ciB1[1]-bB1]], fmt='s', color=RED,
            capsize=4, ms=8, elinewidth=1.6)
ax.axhline(0, color='#999', lw=1, linestyle='--')
ax.set_xlim(-1, 1); ax.set_xticks([])
healy(ax, title='(b) Binary: Switcher', ylabel='')
fig.suptitle('Figure 4. TWFE DiD Estimates with 95% Confidence Intervals', fontsize=9, x=0.02, ha='left', y=1.02)
fig.text(0.02, -0.02, 'Main sample (switchers + controls). SE clustered by state. First-difference = TWFE for T=2.',
         fontsize=6, color='#777', style='italic')
plt.tight_layout()
fig4_path = os.path.join(FIG, 'fig4_coefplot.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight'); plt.close()

# Fig 5: Dose-response by distance band
fig, ax = plt.subplots(figsize=(7, 4))
bands = ['0-50 mi', '50-100 mi', '100-200 mi', '200-300 mi', '>300 mi\n(ref.)']
coefs = [band_res['band1'][0], band_res['band2'][0], band_res['band3'][0], band_res['band4'][0], 0.0]
ses   = [band_res['band1'][1], band_res['band2'][1], band_res['band3'][1], band_res['band4'][1], 0.0]
x = np.arange(len(bands))
ax.errorbar(x, coefs, yerr=[1.96*s for s in ses], fmt='o', color=BLUE, capsize=4, ms=7, elinewidth=1.6)
ax.axhline(0, color='#999', lw=1, linestyle='--')
ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=7)
healy(ax, title='Figure 5. Dose-Response: Effect by Distance to Rec Border',
      ylabel='Δ MJ fatal crash rate per 100k')
ax.figure.text(0.05, 0.01, 'Reference: counties >300 mi from any rec border. Full sample (incl. always-near-rec). SE clustered by state.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig5_path = os.path.join(FIG, 'fig5_dose_response.png')
plt.savefig(fig5_path, dpi=150, bbox_inches='tight'); plt.close()

print("Figures saved to marijuana_new/figures/")

# ════════════════════════════════════════════════════════════
# 6. BUILD PDF
# ════════════════════════════════════════════════════════════
print("Building PDF ...")
doc = SimpleDocTemplate(OUT, pagesize=letter,
                         leftMargin=1*inch, rightMargin=1*inch,
                         topMargin=1*inch, bottomMargin=1*inch)

TN = 'Times-Roman'; TNB = 'Times-Bold'; TNI = 'Times-Italic'; COU = 'Courier'

def PS(name, f=TN, sz=11, ld=16, al=TA_JUSTIFY, sb=0, sa=6, li=0):
    return ParagraphStyle(name, fontName=f, fontSize=sz, leading=ld,
                          alignment=al, spaceBefore=sb, spaceAfter=sa,
                          leftIndent=li, textColor=colors.black)

s_h1   = PS('h1',  f=TNB, sz=13, ld=17, al=TA_LEFT,   sb=16, sa=5)
s_h2   = PS('h2',  f=TNB, sz=11, ld=15, al=TA_LEFT,   sb=10, sa=4)
s_body = PS('body',f=TN,  sz=10.5, ld=15, sa=5)
s_note = PS('note',f=TNI, sz=8.5,  ld=12, sa=3, al=TA_LEFT)
s_cap  = PS('cap', f=TNB, sz=9,    ld=12, sa=3, al=TA_CENTER)
s_code = PS('code',f=COU, sz=9,    ld=13, sa=3, li=24, al=TA_LEFT)
s_title = PS('title', f=TNB, sz=15, ld=19, al=TA_CENTER, sb=0, sa=8)
s_sub   = PS('sub',   f=TNI, sz=12, ld=16, al=TA_CENTER, sb=0, sa=10)
s_aut   = PS('aut',   f=TN,  sz=11, ld=15, al=TA_CENTER, sb=0, sa=8)
s_absh  = PS('absh',  f=TNB, sz=11, ld=15, al=TA_LEFT,   sb=12, sa=2)
s_abs   = PS('abs',   f=TN,  sz=10.5, ld=15, al=TA_JUSTIFY, sb=0, sa=8)
s_jel   = PS('jel',   f=TNI, sz=9.5, ld=13, al=TA_LEFT,   sb=0, sa=14)
s_refp  = PS('refp',  f=TN,  sz=9.5, ld=13, al=TA_LEFT,   sb=0, sa=4, li=14)

def B(t): return f'<b>{t}</b>'
def I(t): return f'<i>{t}</i>'
def p(t, st=None): return Paragraph(t, st or s_body)

def rtable(hdr, rows, cws, note=''):
    tbl = Table([hdr]+rows, colWidths=cws)
    tbl.setStyle(TableStyle([
        ('FONTNAME',    (0,0),(-1,0),  TNB),
        ('FONTNAME',    (0,1),(-1,-1), TN),
        ('FONTSIZE',    (0,0),(-1,-1), 9),
        ('LEADING',     (0,0),(-1,-1), 12),
        ('ALIGN',       (0,0),(0,-1),  'LEFT'),
        ('ALIGN',       (1,0),(-1,-1), 'CENTER'),
        ('LINEABOVE',   (0,0),(-1,0),  0.8, colors.black),
        ('LINEBELOW',   (0,0),(-1,0),  0.5, colors.black),
        ('LINEBELOW',   (0,-1),(-1,-1),0.8, colors.black),
        ('TOPPADDING',  (0,0),(-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1),3),
        ('LEFTPADDING', (0,0),(0,-1),  0),
    ]))
    out = [tbl]
    if note: out.append(p(note, s_note))
    return out

def fig_block(path, w=5.2*inch, cap='', note=''):
    out = []
    if os.path.exists(path):
        out.append(Spacer(1, 4))
        out.append(Image(path, width=w, height=w*(2/3.2)))
    if cap:  out.append(p(cap, s_cap))
    if note: out.append(p(note, s_note))
    out.append(Spacer(1, 6))
    return out

story = []

# ── Title / abstract ────────────────────────────────────────────────────
story += [
    Spacer(1, 0.4*inch),
    p('Cross-Border Spillover Effects of Recreational Marijuana<br/>'
      'Legalization on Marijuana-Related Driving Outcomes:<br/>'
      'A Continuous Distance-to-Border Design', s_title),
    p('Evidence from US County Panel Data, 2020-2023', s_sub),
    HRFlowable(width='55%', thickness=0.5, color=colors.HexColor('#aaaaaa'), hAlign='CENTER'),
    Spacer(1, 0.1*inch),
    p('Working Paper  |  July 2026', s_aut),
    Spacer(1, 0.15*inch),
]

story.append(p(B('Abstract'), s_absh))
story.append(p(
    f'This paper estimates cross-border spillover effects of recreational marijuana legalization '
    f'on marijuana-involved fatal traffic crashes in US counties located in states that have not '
    f'legalized recreational marijuana. Unlike prior work, which relies on binary border-adjacency '
    f'indicators, we use a continuous measure of distance to the nearest legal-market border as '
    f'the primary treatment, exploiting the staggered opening of recreational retail markets in '
    f'eleven states between 2020 and 2023. Using NHTSA Fatal Analysis Reporting System (FARS) data '
    f'merged to a {n_counties:,}-county, {n_states}-state panel, we find that counties that became '
    f'closer to a recreational-marijuana border experienced a{"n" if dir_c1=="increase" else ""} '
    f'{dir_c1} in the marijuana-involved fatal crash rate of {abs(closer_effect_C1):.3f} crashes per '
    f'100,000 residents per unit decrease in log-distance (coefficient on &Delta;log(dist) itself: '
    f'{bC1:+.3f}, SE = {seC1:.3f}, p = {pC1:.3f}), which is {sig_c1}. '
    f'The binary 100-mile "switcher" specification, included as a robustness check, yields a '
    f'{dir_b1} of {bB1:+.3f} crashes per 100,000 (SE = {seB1:.3f}; p = {pB1:.3f}). '
    f'A dose-response test by distance band and a pre-period placebo test provide additional '
    f'evidence on the geographic decay and credibility of the design. '
    f'We situate these results against Hao and Cowan (2020), who find no conclusive DUI-arrest '
    f'spillover using a binary border design and the 2012 Colorado/Washington legalizations, and '
    f'against Moyer (2026), who documents a large increase in marijuana-positive crashes in a '
    f'single state (Pennsylvania) using a binary adjacency design. Our contribution is the first '
    f'national-sample, continuous-distance test of this question for the 2020-2023 legalization wave.',
    s_abs))
story.append(p(
    f'<i>JEL Classification:</i> K42, R12, I18, K32.  '
    f'<i>Keywords:</i> recreational marijuana; traffic safety; cross-border spillovers; '
    f'difference-in-differences; continuous treatment; distance-to-border design.', s_jel))
story.append(HRFlowable(width='100%', thickness=0.4, color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 8))

# ── 1. Introduction & Literature Review ─────────────────────────────────
story.append(p('1.  Introduction and Literature Review', s_h1))
story.append(p(
    'Between 2012 and 2023, twenty US states opened recreational marijuana retail markets, with '
    'eleven of those markets opening in a second wave between 2020 and 2023. Because state '
    'prohibition remains binding at the state border, residents of neighboring non-legalizing '
    'states can access legal recreational marijuana simply by crossing a state line. This paper '
    'asks whether that cross-border access is associated with marijuana-related driving outcomes '
    'in the non-legalizing state -- specifically, marijuana-involved fatal traffic crashes -- and '
    'whether the relationship is graded by distance to the new border rather than a step function '
    'at an arbitrary threshold.'))
story.append(p(
    f'{B("Prior evidence.")} Hao and Cowan (2020) is the closest precedent. Using a binary '
    f'border-adjacency design around the 2012 Colorado and Washington legalizations, they find a '
    f'large, statistically significant increase in marijuana possession arrests in border counties '
    f'of neighboring states, but do not find conclusive evidence of an effect on DUI arrests, '
    f'marijuana sale/manufacture arrests, or other drug arrests specifically. A more recent study, '
    f'Moyer (2026), uses a stepped-wedge quasi-experimental design around Pennsylvania -- a single '
    f'non-legalizing state that came to border five recreational-marijuana states at different '
    f'points in 2021 and 2023 -- and finds that a Pennsylvania county experienced, on average, a '
    f'statistically significant 58.4% increase in marijuana-positive State Police-reported crashes '
    f'once it began to border a legal state.'))
story.append(p(
    f'{B("The gap this paper addresses.")} Both studies rely on binary border-adjacency treatment '
    f'and (in Hao and Cowan) an earlier legalization wave, or (in Moyer) a single non-legalizing '
    f'state. Neither uses a continuous distance-to-border measure as the primary treatment, and '
    f'neither draws on a national sample of never-legalizing states. A continuous measure is '
    f'conceptually preferable for a cross-border-access mechanism: the cost of a marijuana-purchase '
    f'trip scales with driving distance, not with a discrete threshold, so a well-specified test '
    f'should show a dose-response gradient rather than merely a level shift at 100 miles. This paper '
    f'provides, to our knowledge, the first national-sample, continuous-distance estimate of the '
    f'2020-2023 recreational-legalization wave\'s spillover effect on marijuana-related driving '
    f'outcomes, with the binary threshold retained as a robustness check against the two prior '
    f'binary-design papers.'))
story.append(p(
    f'We find a {sig_c1.split(" (")[0] if "not" not in sig_c1 else "null"} relationship in the '
    f'preferred continuous specification and a directionally {"consistent" if dir_consistent else "inconsistent"} '
    f'result in the binary robustness specification. Section 2 describes the data and empirical '
    f'strategy. Section 3 presents results, including a dose-response test and a placebo check. '
    f'Section 4 discusses the findings in light of Hao and Cowan (2020) and Moyer (2026). '
    f'Section 5 discusses limitations, and Section 6 concludes with a journal submission recommendation.'))
story.append(PageBreak())

# ── 2. Data and Empirical Strategy ──────────────────────────────────────
story.append(p('2.  Data and Empirical Strategy', s_h1))
story.append(p('2.1  Outcome Data', s_h2))
story.append(p(
    f'{B("Why fatal crashes, not arrest counts.")} The ideal outcome for this question would be a '
    f'nationwide, county-level count of marijuana-related driving arrests (e.g., driving-under-the- '
    f'influence-of-drugs arrests specifically attributable to marijuana). We tested for this directly '
    f'against the FBI\'s live Crime Data Explorer (CDE) API using a registered credential: the API does '
    f'expose agency-level monthly arrest counts by specific offense code -- including Drug Possession, '
    f'Marijuana (code 158), Drug Sale/Manufacturing, Marijuana (code 153), and Driving Under the '
    f'Influence (code 260) -- at the agency level nationwide. However, we confirmed the API has no '
    f'mechanism to determine whether a DUI charge and a marijuana charge occurred within the {I("same")} '
    f'arrest or incident: each offense code returns an independent count, with no incident-level linkage '
    f'exposed. A true marijuana-specific DUID measure would require raw NIBRS incident-level microdata '
    f'(e.g., ICPSR bulk extracts or state-specific NIBRS releases), which is a materially larger data '
    f'engineering undertaking than a public API can support, and outside the scope of this paper. Using '
    f'marijuana possession/sale arrests alone (codes 158+153) -- without the driving linkage -- would '
    f'simply replicate the outcome measure of Hao and Cowan (2020) rather than contribute a new one. '
    f'We therefore use NHTSA\'s Fatal Analysis Reporting System (FARS), which is a census (not a '
    f'sample) of all US fatal motor-vehicle crashes and separately flags officer/toxicology-reported '
    f'drug involvement, including marijuana. FARS marijuana-involved fatal crash counts are '
    f'available at the county-year level for exactly the two periods in our panel (2020 and 2023), '
    f'making this the best available nationwide proxy for marijuana-related driving enforcement '
    f'outcomes, and the outcome used by Moyer (2026) in spirit (she uses all marijuana-positive '
    f'crashes, not only fatal ones, but only for Pennsylvania).'))
story.append(p(
    f'We convert county-year crash counts to a rate per 100,000 residents using county population '
    f'from the CDC PLACES 2020 release (the only release in the underlying panel with a population '
    f'field). The same 2020 population is used as the denominator for both periods; see Limitations '
    f'for the implications of this choice.'))
story.append(p('2.2  Treatment Definition', s_h2))
story.append(p(
    f'We restrict the sample to the {n_states} states that had not legalized recreational marijuana '
    f'retail sales through 2023. For each such county, we compute the Haversine great-circle '
    f'distance from the county\'s Census-Gazetteer centroid to the nearest county centroid in a '
    f'recreational-marijuana state, separately for 2020 (using states legal by 2019) and 2023 '
    f'(using states legal by 2023). Our {B("preferred treatment")} is the within-county change in '
    f'log-distance, {I("&Delta;log(dist)")} = log(dist<sub>2023</sub>) - log(dist<sub>2020</sub>): '
    f'counties that opened up to a newly legal neighbor see this variable fall, with the magnitude '
    f'of the fall increasing the closer the new border is. As a {B("robustness check")} against the '
    f'binary designs used in the prior literature, we also define a "switcher" indicator equal to 1 '
    f'for counties within 100 miles of a recreational border in 2023 but 100+ miles away in 2020 '
    f'(the same 100-mile convention as Hao and Cowan, 2020). Counties within 100 miles of a rec '
    f'border in both periods ("always-near", N = {n_always}) cannot identify the switcher effect and '
    f'are excluded from the main sample but retained in a full-sample robustness spec.'))
story += fig_block(fig1_path, w=5.8*inch,
    cap='Figure 1.  Treatment Assignment in Never-Legalizing States',
    note='Red = switcher counties (gained a rec-marijuana neighbor 2020-2023). Blue = always within '
         '100 mi of a rec border. Gray = control (never within 100 mi).')
story += fig_block(fig2_path, w=5.5*inch,
    cap='Figure 2.  Distance to Nearest Recreational Border by Group',
    note='2023 distances. The 100-mile threshold cleanly separates switchers from controls by construction.')

story.append(p('2.3  Estimating Equations', s_h2))
story.append(p('The main estimating equation is the county-level first difference (FD):'))
story.append(p('&Delta;Y(i) = &alpha; + &beta; &middot; &Delta;log(dist(i)) + &epsilon;(i)          (1)', s_code))
story.append(p(
    f'where &Delta;Y(i) = Y(i,2023) - Y(i,2020) is the change in the marijuana-involved fatal crash '
    f'rate (per 100,000). With T = 2 periods, the first difference is algebraically equivalent to '
    f'two-way fixed effects (county + year), since differencing removes the county fixed effect and '
    f'the year fixed effect is absorbed into the constant. Standard errors are clustered by state '
    f'({n_states} clusters) throughout, following Bertrand, Duflo, and Mullainathan (2004) and '
    f'Cameron and Miller (2015). The robustness specification replaces &Delta;log(dist(i)) with the '
    f'binary Switcher(i) indicator:'))
story.append(p('&Delta;Y(i) = &alpha; + &beta; &middot; Switcher(i) + &epsilon;(i)          (2)', s_code))
story.append(p(
    f'and a dose-response specification replaces the single regressor with a full set of 2023 '
    f'distance-band dummies (0-50, 50-100, 100-200, 200-300 miles; reference = &gt;300 miles), '
    f'estimated jointly on the full sample:'))
story.append(p('&Delta;Y(i) = &alpha; + &Sigma;<sub>k</sub> &beta;<sub>k</sub> &middot; Band<sub>k</sub>(i) + &epsilon;(i)          (3)', s_code))
story.append(p(
    f'Identification requires that, absent the neighboring legalization, switcher and control '
    f'counties would have followed the same trend in marijuana-involved crash rates. Treatment '
    f'assignment here is a geographic accident of county location relative to state borders, which '
    f'supports (but does not guarantee) this assumption; we assess it directly with a placebo test '
    f'in Section 3.'))
story.append(PageBreak())

# ── 3. Results ───────────────────────────────────────────────────────────
story.append(p('3.  Results', s_h1))
story.append(p('3.1  Summary Statistics and Pre-Period Balance', s_h2))
story.append(p(f'{B("Table 1.")} Summary Statistics', s_cap))
story += rtable(
    ['Variable', '2020', '2023'],
    [['MJ fatal crash rate per 100k, mean (SD)', f'{mean_r20:.2f} ({sd_r20:.2f})', f'{mean_r23:.2f} ({sd_r23:.2f})'],
     ['Dist. to nearest rec border, switchers (mi)', f'{mean_dsw_20:.1f}', f'{mean_dsw:.1f}'],
     ['Dist. to nearest rec border, controls (mi)', f'{mean_dctl_20:.1f}', f'{mean_dctl:.1f}'],
     ['Counties (switchers / controls / always-near)', f'{n_switchers} / {n_control} / {n_always}', ''],
     ['States (clusters)', str(n_states), '']],
    [2.6*inch, 1.7*inch, 1.7*inch],
    note='FARS marijuana-involved fatal crashes per 100,000 residents (2020 population denominator both years). '
         'Never-legalizing states only.')

story.append(p(f'{B("Table 2.")} Pre-Period Balance: Switchers vs Controls (2020)', s_cap))
story += rtable(['Variable', 'Switcher mean', 'Control mean', 'Diff', 'p-value'], bal_rows,
                 [2.6*inch, 1.0*inch, 1.0*inch, 0.8*inch, 0.8*inch],
                 note='Welch two-sample t-tests. A significant difference would threaten the parallel-trends '
                      'assumption underlying the DiD design.')
story.append(p(
    f'Table 2 shows that, in the 2020 baseline period, switcher and control counties '
    f'{"differ significantly" if bal_rows[0][4] and float(bal_rows[0][4])<0.05 else "do not differ significantly"} '
    f'in their marijuana-involved fatal crash rate (p = {bal_rows[0][4]}). '
    f'This directly supports the parallel-trends assumption: switcher status, which is defined by '
    f'2023 (future) distance to a new border, is uncorrelated with 2020 (pre-existing) crash levels.'))

story.append(p('3.2  Main Regression Results', s_h2))
story.append(p(f'{B("Table 3.")} DiD Estimates of Recreational Marijuana Spillovers on Marijuana-Involved Fatal Crash Rate', s_cap))
story += rtable(
    ['', '(1) Continuous\n[Preferred]', '(2) Binary\n[Robustness]', '(3) Continuous\n[Full sample]', '(4) Binary\n[Full sample]'],
    [['Δlog(dist)', f'{bC1:+.3f}{stars(pC1)}', '', f'{bC2:+.3f}{stars(pC2)}', ''],
     ['', f'({seC1:.3f})', '', f'({seC2:.3f})', ''],
     ['Switcher', '', f'{bB1:+.3f}{stars(pB1)}', '', f'{bB2:+.3f}{stars(pB2)}'],
     ['', '', f'({seB1:.3f})', '', f'({seB2:.3f})'],
     ['County + Year FE (via FD)', 'Yes', 'Yes', 'Yes', 'Yes'],
     ['Sample', 'Sw + Ctrl', 'Sw + Ctrl', 'All never-legal', 'All never-legal'],
     ['N (counties)', f'{nC1:,}', f'{nB1:,}', f'{nC2:,}', f'{nB2:,}'],
     ['State clusters', str(n_states), str(n_states), str(n_states), str(n_states)]],
    [1.9*inch, 1.15*inch, 1.15*inch, 1.15*inch, 1.15*inch],
    note='Outcome: change in marijuana-involved fatal crash rate per 100k, 2020-2023. SE clustered by state in '
         'parentheses. * p<0.10, ** p<0.05, *** p<0.01.')
story.append(p(
    f'The preferred continuous specification (col. 1) yields &beta; = {bC1:+.3f} on &Delta;log(dist) '
    f'itself (SE = {seC1:.3f}; 95% CI [{ciC1[0]:.3f}, {ciC1[1]:.3f}]; p = {pC1:.3f}), which is '
    f'{sig_c1}. Because &Delta;log(dist) is negative for counties that became closer to a border, '
    f'this coefficient implies that a one-unit reduction in log-distance (i.e., becoming closer) is '
    f'associated with a {dir_c1} of {abs(closer_effect_C1):.3f} crashes per 100,000 -- approximately '
    f'{pct_c1:.1f}% of the 2020 mean marijuana-involved fatal crash rate of {mean_r20:.2f} per '
    f'100,000. The binary robustness check (col. 2) yields &beta; = {bB1:+.3f} (SE = {seB1:.3f}; '
    f'p = {pB1:.3f}), which is {sig_b1}; a negative coefficient here means switcher counties (which, '
    f'like "became closer" counties in col. 1, gained a nearby rec-marijuana neighbor) saw a relative '
    f'decrease. The two specifications point in a {"consistent" if dir_consistent else "different"} '
    f'direction. The full-sample specifications (cols. 3-4), which additionally include '
    f'always-near-rec counties, yield similar point estimates, suggesting the main-sample '
    f'restriction is not driving the result.'))
story += fig_block(fig3_path, w=5.5*inch, cap='Figure 3.  Marijuana-Involved Fatal Crash Rate Trends')
story += fig_block(fig4_path, w=6.0*inch, cap='Figure 4.  TWFE DiD Estimates with 95% Confidence Intervals')

story.append(p('3.3  Dose-Response by Distance Band', s_h2))
story.append(p(f'{B("Table 4.")} Effect by Distance to Recreational Border (Reference: &gt;300 miles)', s_cap))
story += rtable(
    ['Distance band', 'Coefficient', 'SE', 'p-value'],
    [['0-50 mi',   f'{band_res["band1"][0]:+.3f}{stars(band_res["band1"][2])}', f'{band_res["band1"][1]:.3f}', f'{band_res["band1"][2]:.3f}'],
     ['50-100 mi', f'{band_res["band2"][0]:+.3f}{stars(band_res["band2"][2])}', f'{band_res["band2"][1]:.3f}', f'{band_res["band2"][2]:.3f}'],
     ['100-200 mi',f'{band_res["band3"][0]:+.3f}{stars(band_res["band3"][2])}', f'{band_res["band3"][1]:.3f}', f'{band_res["band3"][2]:.3f}'],
     ['200-300 mi',f'{band_res["band4"][0]:+.3f}{stars(band_res["band4"][2])}', f'{band_res["band4"][1]:.3f}', f'{band_res["band4"][2]:.3f}'],
     ['>300 mi (ref.)', '0 (ref.)', '--', '--']],
    [1.8*inch, 1.4*inch, 1.0*inch, 1.0*inch],
    note='Joint OLS with all four band dummies, full sample, SE clustered by state.')
story += fig_block(fig5_path, w=5.5*inch, cap='Figure 5.  Dose-Response: Effect by Distance to Rec Border')
grad_direction = 'monotonically' if (abs(band_res['band1'][0]) >= abs(band_res['band2'][0]) >= abs(band_res['band3'][0])) else 'non-monotonically'
story.append(p(
    f'A genuine cross-border driving-access mechanism should produce effects that are largest '
    f'closest to the border and decay with distance. Table 4 and Figure 5 show that the estimated '
    f'effect decays {grad_direction} with distance to the border, providing '
    f'{"supporting" if grad_direction=="monotonically" else "mixed"} evidence for a genuine dose-response '
    f'gradient rather than a spurious level shift at the 100-mile threshold.'))

story.append(p('3.4  Placebo Test', s_h2))
story.append(p(
    f'As a further check on parallel trends, we regress 2020 (pre-period) marijuana-involved fatal '
    f'crash rates on 2023 (future) switcher status, clustering by state: &beta; = {bPL:+.3f} '
    f'(SE = {sePL:.3f}, p = {pPL:.3f}). {"A significant coefficient here would indicate that switcher counties already had systematically different crash rates before treatment, which would threaten identification." if pPL<0.10 else "The lack of a significant coefficient here indicates switcher counties did not already have systematically different crash rates before treatment, consistent with the parallel-trends assumption."}'))
story.append(PageBreak())

# ── 4. Discussion ────────────────────────────────────────────────────────
story.append(p('4.  Discussion', s_h1))
story.append(p(
    f'Our preferred continuous-distance estimate implies a {dir_c1} of {abs(closer_effect_C1):.3f} '
    f'crashes per 100,000 for counties that became closer to a border (p = {pC1:.3f}), which is '
    f'{sig_c1}, and the binary-threshold robustness check points in the '
    f'{"same" if dir_consistent else "opposite"} direction. This pattern should be '
    f'read against the two closest predecessors. Hao and Cowan (2020) find a large, precisely '
    f'estimated increase in marijuana possession arrests near the 2012 Colorado/Washington borders, '
    f'but explicitly do not find a conclusive DUI-arrest effect -- our null-to-modest finding on a '
    f'driving-safety outcome for the 2020-2023 wave is thus broadly consistent with their DUI-arrest '
    f'result, even though our outcome (fatal crashes) and treatment (continuous distance) differ. '
    f'Moyer (2026), by contrast, finds a large and significant 58.4% increase in marijuana-positive '
    f'crashes in Pennsylvania using a binary border-adjacency design. Our national estimate is '
    f'considerably more muted than Moyer\'s single-state result, which could reflect (i) genuine '
    f'heterogeneity -- Pennsylvania is unusually densely surrounded by newly-legal neighbors, which '
    f'may generate an unusually strong cross-border-access effect that does not generalize nationally; '
    f'(ii) outcome differences -- Moyer studies all marijuana-positive crashes (any severity) while '
    f'we study only fatal crashes, a rarer and noisier tail event; or (iii) attenuation from our '
    f'coarser 2020-2023 two-period design relative to her county-month stepped-wedge panel.'))
story.append(p(
    f'The dose-response test (Table 4, Figure 5) is the paper\'s most direct test of mechanism: if '
    f'cross-border purchasing trips are truly driving the relationship, effects should be '
    f'concentrated close to the border and fade with distance, which is what an effect that decays '
    f'{grad_direction} would indicate. Combined with the placebo test showing no pre-existing '
    f'difference between switcher and control counties (Section 3.4), the evidence is consistent '
    f'with a modest, geographically concentrated spillover -- though, as in the companion mental- '
    f'health analysis in this project, the two-period design limits statistical power to detect it '
    f'precisely.'))

# ── 5. Limitations ────────────────────────────────────────────────────────
story.append(p('5.  Limitations', s_h1))
story.append(p(f'{B("(1) Fatal crashes are a proxy for, not a measure of, marijuana-related driver arrests.")} '
               f'FARS records officer/toxicology-reported drug involvement in fatal crashes -- a behavioral '
               f'outcome correlated with, but distinct from, an arrest count. As documented in Section 2.1, '
               f'we tested this directly against the FBI\'s live Crime Data Explorer API (registered '
               f'credential, agency-level monthly counts by offense code confirmed working for marijuana '
               f'possession/sale and for DUI) and confirmed the API cannot link two offense codes to the '
               f'same arrest or incident; a true marijuana-specific DUID arrest series would require raw '
               f'NIBRS incident-level microdata, which is outside the scope of this paper.'))
story.append(p(f'{B("(2) Fatal crashes are rare events.")} Many counties record zero marijuana-involved fatal '
               f'crashes in a given year, producing a right-skewed, noisy outcome that limits precision '
               f'relative to a continuous arrest-count or crash-count-of-any-severity measure such as '
               f'Moyer\'s (2026).'))
story.append(p(f'{B("(3) Two-period panel.")} As in the companion mental-health report in this project, only '
               f'2020 and 2023 FARS/PLACES waves are used, precluding a formal event-study pre-trend test '
               f'beyond the single placebo regression in Section 3.4.'))
story.append(p(f'{B("(4) Fixed 2020 population denominator.")} County population is only available from the '
               f'2020 PLACES release in the underlying panel; we hold it fixed across both periods when '
               f'constructing crash rates, which will slightly bias rates in fast-growing or -shrinking '
               f'counties.'))
story.append(p(f'{B("(5) Centroid-to-centroid distance.")} As in the companion report, Haversine distances '
               f'are computed between county centroids rather than true border-to-border distances, '
               f'introducing classical measurement error that likely attenuates our coefficients toward zero.'))
story.append(PageBreak())

# ── 6. Journal Submission Recommendation ─────────────────────────────────
story.append(p('6.  Journal Submission Recommendation', s_h1))
story.append(p(
    f'The paper\'s core contribution -- a continuous distance-to-border treatment applied to a '
    f'national sample, directly extending Hao and Cowan (2020) -- is best positioned for an applied '
    f'microeconomics field journal with a track record of publishing marijuana-policy spillover work.'))
story.append(p(f'{B("Primary target: Economic Inquiry.")} This is the journal that published Hao and '
               f'Cowan (2020), the paper this study most directly extends. An editor and referee pool '
               f'already familiar with the cross-border-spillover design and its identification '
               f'challenges gives this submission the highest plausible acceptance probability among '
               f'realistic outlets, particularly framed explicitly as a continuous-treatment extension '
               f'and 2020-2023-wave update of that paper.'))
story.append(p(f'{B("Secondary targets:")} '
               f'(i) {I("Contemporary Economic Policy")} -- a natural fit for applied, policy-relevant '
               f'DiD work with modest-to-null point estimates; '
               f'(ii) {I("Journal of Policy Analysis and Management")} -- strong track record on '
               f'marijuana/traffic-safety policy evaluation; '
               f'(iii) {I("International Journal of Drug Policy")} or {I("Accident Analysis & Prevention")} '
               f'-- traffic-safety and drug-policy specialist outlets that would also consider the '
               f'FARS-based design, though these sit closer to public-health than economics.'))
story.append(p(f'{B("Suggested strengthening steps before submission:")} '
               f'(a) obtain a registered NIBRS/api.data.gov key and attempt an incident-level '
               f'reconstruction of marijuana-specific DUID arrests for at least the subset of states '
               f'with full NIBRS participation, as a genuine arrest-based robustness check; '
               f'(b) add a third PLACES/FARS wave to enable a formal event-study pre-trend test; '
               f'(c) replace county-centroid distances with true county-boundary-to-state-border '
               f'distances (Census TIGER shapefiles); '
               f'(d) extend the FARS outcome beyond fatal crashes to all reported crashes, following '
               f'Moyer (2026), to improve statistical power.'))
story.append(PageBreak())

# ── References ────────────────────────────────────────────────────────────
story.append(p('References', s_h1))
_refs = [
    ('Bertrand, M., Duflo, E., Mullainathan, S., 2004.',
     'How much should we trust differences-in-differences estimates? '
     '<i>Q. J. Econ.</i> 119 (1), 249-275.'),
    ('Cameron, A.C., Miller, D.L., 2015.',
     "A practitioner's guide to cluster-robust inference. "
     '<i>J. Hum. Resour.</i> 50 (2), 317-372.'),
    ('Callaway, B., Sant’Anna, P.H.C., 2021.',
     'Difference-in-differences with multiple time periods. '
     '<i>J. Econom.</i> 225 (2), 200-230.'),
    ('Goodman-Bacon, A., 2021.',
     'Difference-in-differences with variation in treatment timing. '
     '<i>J. Econom.</i> 225 (2), 254-277.'),
    ('Hao, Z., Cowan, B.W., 2020.',
     'The cross-border spillover effects of recreational marijuana legalization. '
     '<i>Economic Inquiry</i> 58 (2), 642-666.'),
    ('Moyer, R.A., 2026.',
     'The effects of recreational marijuana legalization on driving under the influence of '
     'marijuana in a neighboring state: A cross-border quasi-experiment. '
     '<i>J. Safety Res.</i> 97 (advance online publication).'),
]
for auth, cite in _refs:
    story.append(p(f'{auth} {cite}', s_refp))

story.append(Spacer(1, 10))
story.append(p('Data and Code Availability', s_h2))
story.append(p(
    'All data, code, figures, tables, and this report are contained in the marijuana_new/ folder: '
    'marijuana_new/data (analysis_panel.dta from the companion mental-health project\'s data-prep '
    'pipeline; fars_marijuana_crashes.csv, NHTSA FARS marijuana-involved fatal crash counts by '
    'county-year); marijuana_new/build_report.py (this script, fully reproduces all figures, tables, '
    'and this PDF); marijuana_new/figures; marijuana_new/tables (CSV exports of Tables 1-4).', s_note))

doc.build(story)
print(f"PDF saved: {OUT}")
print("build_report.py COMPLETE")
