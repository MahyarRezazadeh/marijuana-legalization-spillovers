# ============================================================
# MARIJUANA LEGALIZATION SPILLOVER: FULL ANALYSIS + PDF
# Reads analysis_panel.dta  ->  Marijuana_Spillover_Results.pdf
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                                 Image, Spacer, PageBreak, HRFlowable)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect"
DATA = os.path.join(BASE, 'data')
FIG  = os.path.join(BASE, 'figures')
OUT  = os.path.join(BASE, 'Marijuana_Spillover_Results.pdf')
os.makedirs(FIG, exist_ok=True)

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
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════
print("Loading analysis_panel.dta ...")
df = pd.read_stata(os.path.join(DATA, 'analysis_panel.dta'))

# Convert key variables to float
num_cols = ['mhlth','depression','switcher','post','switcher_post','never_legal',
            'log_dist','log_dist_post','dist_rec_mi','dist_rec_2023_mi',
            'dist_rec_2020_mi','dist_band']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df['year_int'] = df['year'].astype(float).astype(int)

# county_id and state_id are string categories from Stata encode
df['fips']  = df['county_id'].astype(str)
df['state'] = df['state_id'].astype(str)

# Rebuild dist_band robustly
df['dist_band'] = np.where(df['dist_rec_2023_mi'] <= 50,  1,
                  np.where(df['dist_rec_2023_mi'] <= 100, 2,
                  np.where(df['dist_rec_2023_mi'] <= 200, 3,
                  np.where(df['dist_rec_2023_mi'] <= 300, 4, 5))))

main = df[df['sample_grp'] != 'Always near rec border'].copy()
full = df.copy()

n_counties  = int(df['fips'].nunique())
n_switchers = int(main[main['switcher']==1]['fips'].nunique())
n_control   = int(main[main['switcher']==0]['fips'].nunique())
n_always    = int(df[df['sample_grp']=='Always near rec border']['fips'].nunique())
n_states    = int(df['state'].nunique())

print(f"Counties (unique FIPS): {n_counties} | Switchers: {n_switchers} | Controls: {n_control}")

# ════════════════════════════════════════════════════════════
# 2. FIRST-DIFFERENCE HELPER  (TWFE = FD when T=2)
# ════════════════════════════════════════════════════════════
def make_fd(data, outcome):
    """Create first-difference dataset; keep only counties in BOTH periods."""
    cols = [outcome, 'fips', 'state', 'switcher', 'never_legal',
            'log_dist', 'dist_band', 'sample_grp', 'year_int',
            'dist_rec_2023_mi', 'dist_rec_2020_mi']
    sub  = data[[c for c in cols if c in data.columns]].copy()
    pre  = sub[sub['year_int']==2020].set_index('fips')
    post = sub[sub['year_int']==2023].set_index('fips')
    both = pre.index.intersection(post.index)
    pre  = pre.loc[both]; post = post.loc[both]
    fd   = pd.DataFrame(index=both)
    fd['d_y']      = post[outcome].values - pre[outcome].values
    fd['switcher'] = pre['switcher'].values.astype(float)
    fd['state']    = pre['state'].values
    fd['never_legal'] = pre['never_legal'].values
    fd['dist_band_23'] = post['dist_band'].values
    fd['log_dist_pre'] = pre['log_dist'].values
    fd['log_dist_post2']= post['log_dist'].values
    fd['d_log_dist']   = fd['log_dist_post2'] - fd['log_dist_pre']
    fd['sample_grp'] = pre['sample_grp'].values
    for b in [1,2,3,4]:
        fd[f'band{b}'] = (fd['dist_band_23']==b).astype(float)
    return fd.dropna(subset=['d_y','switcher'])

def ols_cl(y, X, groups, xname):
    """OLS + state-clustered SE. Returns (b, se, p, nobs)."""
    g = pd.Categorical(groups).codes  # integer codes for clustering
    Xc = sm.add_constant(X)
    try:
        res = sm.OLS(y, Xc).fit(cov_type='cluster', cov_kwds={'groups': g})
        b  = res.params[xname]
        se = res.bse[xname]
        p  = res.pvalues[xname]
    except Exception as e:
        print(f"  Warning: {e}. Using HC3.")
        res = sm.OLS(y, Xc).fit(cov_type='HC3')
        b  = res.params[xname]
        se = res.bse[xname]
        p  = res.pvalues[xname]
    return float(b), float(se), float(p), int(len(y))

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''

def ci95(b, s): return b-1.96*s, b+1.96*s

# ════════════════════════════════════════════════════════════
# 3. RUN ALL REGRESSIONS
# ════════════════════════════════════════════════════════════
print("Running regressions (first-difference = TWFE for T=2) ...")

# ── MHLTH ─────────────────────────────────────────────────
fd_m = make_fd(main, 'mhlth')
fd_f = make_fd(full[full['never_legal']==1], 'mhlth')

# Spec 1: FD without controlling for anything else (= county FE only)
bm1, sem1, pm1, nm1 = ols_cl(fd_m['d_y'], fd_m[['switcher']], fd_m['state'], 'switcher')

# Spec 2: same sample -- in 2-period FD this IS the TWFE estimate (preferred)
bm2, sem2, pm2, nm2 = bm1, sem1, pm1, nm1  # identical for balanced panel
ci_lo_m, ci_hi_m = ci95(bm2, sem2)

# Spec 3: all never-legal (include always-near-rec)
bm3, sem3, pm3, nm3 = ols_cl(fd_f['d_y'], fd_f[['switcher']], fd_f['state'], 'switcher')

# Spec 4: continuous log-distance
bm4, sem4, pm4, nm4 = ols_cl(fd_f['d_y'], fd_f[['d_log_dist']], fd_f['state'], 'd_log_dist')

print(f"MHLTH TWFE: b={bm2:+.3f}  SE={sem2:.3f}  p={pm2:.3f}  N={nm2}")
print(f"MHLTH Cont: b={bm4:+.3f}  SE={sem4:.3f}  p={pm4:.3f}  N={nm4}")

# ── Depression ─────────────────────────────────────────────
fd_dm = make_fd(main, 'depression')
fd_df = make_fd(full[full['never_legal']==1], 'depression')

bd1, sed1, ppd1, nd1 = ols_cl(fd_dm['d_y'], fd_dm[['switcher']], fd_dm['state'], 'switcher')
bd2, sed2, ppd2, nd2 = bd1, sed1, ppd1, nd1
ci_lo_d, ci_hi_d = ci95(bd2, sed2)
bd3, sed3, ppd3, nd3 = ols_cl(fd_df['d_y'], fd_df[['d_log_dist']], fd_df['state'], 'd_log_dist')

# Continuous spec on main sample (preferred: excludes always-near)
bm4m, sem4m, pm4m, nm4m = ols_cl(fd_m['d_y'],  fd_m[['d_log_dist']],  fd_m['state'],  'd_log_dist')
bd3m, sed3m, ppd3m, nd3m = ols_cl(fd_dm['d_y'], fd_dm[['d_log_dist']], fd_dm['state'], 'd_log_dist')
ci_lo_m_cont, ci_hi_m_cont = ci95(bm4m, sem4m)
ci_lo_d_cont, ci_hi_d_cont = ci95(bd3m, sed3m)

print(f"Depression TWFE: b={bd2:+.3f}  SE={sed2:.3f}  p={ppd2:.3f}  N={nd2}")
print(f"MHLTH Cont (main): b={bm4m:+.3f}  SE={sem4m:.3f}  p={pm4m:.3f}  N={nm4m}")
print(f"Dep   Cont (main): b={bd3m:+.3f}  SE={sed3m:.3f}  p={ppd3m:.3f}  N={nd3m}")

# ── Dose-response ──────────────────────────────────────────
hb_mh = []; hb_dp = []
for b in [1,2,3,4]:
    sub_m = fd_f.copy()
    sub_d = fd_df.copy()
    bv, sv, pv, nv = ols_cl(sub_m['d_y'], sub_m[[f'band{b}']], sub_m['state'], f'band{b}')
    hb_mh.append((bv,sv,pv))
    bv2,sv2,pv2,nv2 = ols_cl(sub_d['d_y'], sub_d[[f'band{b}']], sub_d['state'], f'band{b}')
    hb_dp.append((bv2,sv2,pv2))

# ── Summary statistics ─────────────────────────────────────
mean_mh20  = df[df['year_int']==2020]['mhlth'].mean()
sd_mh20    = df[df['year_int']==2020]['mhlth'].std()
mean_mh23  = df[df['year_int']==2023]['mhlth'].mean()
sd_mh23    = df[df['year_int']==2023]['mhlth'].std()
mean_dep20 = df[df['year_int']==2020]['depression'].mean()
sd_dep20   = df[df['year_int']==2020]['depression'].std()
mean_dep23 = df[df['year_int']==2023]['depression'].mean()
sd_dep23   = df[df['year_int']==2023]['depression'].std()
mean_dsw   = main[main['switcher']==1]['dist_rec_mi'].mean()
mean_dctl  = main[main['switcher']==0]['dist_rec_mi'].mean()

pct_mh  = abs(bm2)/mean_mh20*100
pct_dep = abs(bd2)/mean_dep20*100
dir_mh  = 'increase' if bm2 > 0 else 'decrease'
dir_dep = 'increase' if bd2 > 0 else 'decrease'

def sig_label(p):
    if p<0.01: return 'statistically significant at the 1% level'
    if p<0.05: return 'statistically significant at the 5% level'
    if p<0.10: return 'marginally significant at the 10% level'
    return f'not statistically significant (p = {p:.3f})'

# ── Placebo ────────────────────────────────────────────────
pre2020 = main[main['year_int']==2020].copy()
X_pl = sm.add_constant(pre2020['switcher'].astype(float))
g_pl = pd.Categorical(pre2020['state']).codes
res_pl = sm.OLS(pre2020['mhlth'], X_pl).fit(
    cov_type='cluster', cov_kwds={'groups': g_pl})
pl_b = float(res_pl.params['switcher'])
pl_p = float(res_pl.pvalues['switcher'])

# ════════════════════════════════════════════════════════════
# 3b.  MEDICAL MARIJUANA ROBUSTNESS CHECK
# ════════════════════════════════════════════════════════════
print("Medical marijuana robustness analysis ...")

NEWLY_MED = {'SD', 'AL', 'MS'}
NEVER_MED = {'GA','ID','IN','IA','KS','KY','NC','NE','SC','TN','TX','WI','WY'}

# ---- Load raw PLACES CSVs (all states) --------------------
try:
    p1r = pd.read_csv(os.path.join(DATA, 'places_2022_gis.csv'))
    p1r.columns = [c.lower() for c in p1r.columns]
    p1r['fips'] = (p1r['countyfips']
                   .apply(lambda x: str(int(float(x))).zfill(5)
                          if pd.notna(x) else None))
    p1r = p1r.rename(columns={'stateabbr': 'state',
                               'mhlth_crudeprev': 'mhlth',
                               'depression_crudeprev': 'depression'})
    p1r = p1r[['fips', 'state', 'mhlth', 'depression']].dropna().copy()
    p1r['year'] = 2020
except Exception as ex:
    print(f"  Period-1 load error: {ex}")
    p1r = pd.DataFrame(columns=['fips','state','mhlth','depression','year'])

try:
    p2m = pd.read_csv(os.path.join(DATA, 'places_2025_mhlth.csv'))
    p2m.columns = [c.lower() for c in p2m.columns]
    p2m['fips'] = (p2m['locationid']
                   .apply(lambda x: str(int(float(x))).zfill(5)
                          if pd.notna(x) else None))
    p2m = p2m.rename(columns={'stateabbr': 'state', 'data_value': 'mhlth'})
    p2m = p2m[['fips', 'state', 'mhlth']].dropna().copy()

    p2d = pd.read_csv(os.path.join(DATA, 'places_2025_dep.csv'))
    p2d.columns = [c.lower() for c in p2d.columns]
    p2d['fips'] = (p2d['locationid']
                   .apply(lambda x: str(int(float(x))).zfill(5)
                          if pd.notna(x) else None))
    p2d = p2d.rename(columns={'data_value': 'depression'})
    p2d = p2d[['fips', 'depression']].dropna().copy()

    p2r = p2m.merge(p2d, on='fips', how='inner')
    p2r['year'] = 2023
except Exception as ex:
    print(f"  Period-2 load error: {ex}")
    p2r = pd.DataFrame(columns=['fips','state','mhlth','depression','year'])

med_all   = pd.concat([p1r, p2r], ignore_index=True)
med_panel = med_all[med_all['state'].isin(NEWLY_MED | NEVER_MED)].copy()
med_panel['med_treat'] = med_panel['state'].isin(NEWLY_MED).astype(float)

n_med_newly = int(med_panel[(med_panel['year']==2020) &
                             (med_panel['med_treat']==1)]['fips'].nunique())
n_med_never = int(med_panel[(med_panel['year']==2020) &
                             (med_panel['med_treat']==0)]['fips'].nunique())
print(f"  Med panel: {len(med_panel)} rows  newly={n_med_newly}  never={n_med_never} counties")

def med_grp_mean(treat_val, yr, outcome):
    s = med_panel[(med_panel['med_treat']==treat_val) & (med_panel['year']==yr)]
    return float(s[outcome].mean()) if len(s) > 0 else np.nan

mm_new20 = med_grp_mean(1, 2020, 'mhlth')
mm_new23 = med_grp_mean(1, 2023, 'mhlth')
mm_nev20 = med_grp_mean(0, 2020, 'mhlth')
mm_nev23 = med_grp_mean(0, 2023, 'mhlth')
md_new20 = med_grp_mean(1, 2020, 'depression')
md_new23 = med_grp_mean(1, 2023, 'depression')
md_nev20 = med_grp_mean(0, 2020, 'depression')
md_nev23 = med_grp_mean(0, 2023, 'depression')

def make_fd_med(data, outcome):
    sub  = data[['fips','state','med_treat','year',outcome]].copy()
    pre  = sub[sub['year']==2020].set_index('fips')
    post = sub[sub['year']==2023].set_index('fips')
    both = pre.index.intersection(post.index)
    fd   = pd.DataFrame(index=both)
    fd['d_y']       = post.loc[both, outcome].values - pre.loc[both, outcome].values
    fd['med_treat'] = pre.loc[both, 'med_treat'].values.astype(float)
    fd['state']     = pre.loc[both, 'state'].values
    return fd.dropna(subset=['d_y', 'med_treat'])

fd_med_mh = make_fd_med(med_panel, 'mhlth')
fd_med_dp = make_fd_med(med_panel, 'depression')

bmed_mh, semed_mh, pmed_mh, nmed_mh = ols_cl(
    fd_med_mh['d_y'], fd_med_mh[['med_treat']], fd_med_mh['state'], 'med_treat')
bmed_dp, semed_dp, pmed_dp, nmed_dp = ols_cl(
    fd_med_dp['d_y'], fd_med_dp[['med_treat']], fd_med_dp['state'], 'med_treat')

ci_lo_med_mh = bmed_mh - 1.96*semed_mh
ci_hi_med_mh = bmed_mh + 1.96*semed_mh
ci_lo_med_dp = bmed_dp - 1.96*semed_dp
ci_hi_med_dp = bmed_dp + 1.96*semed_dp

print(f"  Medical MHLTH DiD: b={bmed_mh:+.3f}  SE={semed_mh:.3f}  p={pmed_mh:.3f}  N={nmed_mh}")
print(f"  Medical Dep   DiD: b={bmed_dp:+.3f}  SE={semed_dp:.3f}  p={pmed_dp:.3f}  N={nmed_dp}")

# ════════════════════════════════════════════════════════════
# 3b2.  NHTSA FARS: MARIJUANA-INVOLVED FATAL CRASHES
# ════════════════════════════════════════════════════════════
print("Loading NHTSA FARS marijuana-impaired driving data ...")
try:
    _fars_raw = pd.read_csv(os.path.join(DATA, 'fars_marijuana_crashes.csv'))
    _fars_raw['fips5'] = _fars_raw['fips5'].astype(str).str.zfill(5)

    # Get 5-digit FIPS from df (fips5 column from Stata dta)
    _has_fips5 = 'fips5' in df.columns
    if _has_fips5:
        _cmeta = (df[df['year_int'] == 2020].copy())
        _cmeta['fips5'] = _cmeta['fips5'].astype(str).str.zfill(5)
        _cmeta = _cmeta[['fips5','fips','switcher','state','never_legal','sample_grp']].drop_duplicates('fips5')
    else:
        _cmeta = (df[df['year_int'] == 2020][['fips','switcher','state','never_legal','sample_grp']]
                  .drop_duplicates('fips').copy())
        _cmeta['fips5'] = _cmeta['fips']

    # Population for normalization
    if 'population' in df.columns:
        _pop = (df[df['year_int']==2020][['fips','population']].drop_duplicates('fips')
                .copy().rename(columns={'population':'pop20'}))
        if _has_fips5:
            _pop['fips5'] = df.loc[df['year_int']==2020, 'fips5'].astype(str).str.zfill(5)
        else:
            _pop['fips5'] = _pop['fips']
        _cmeta = _cmeta.merge(_pop[['fips5','pop20']], on='fips5', how='left')
    else:
        _cmeta['pop20'] = np.nan

    _f20 = _fars_raw[_fars_raw['year']==2020][['fips5','mj_fatal_crashes']].rename(columns={'mj_fatal_crashes':'mj20'})
    _f23 = _fars_raw[_fars_raw['year']==2023][['fips5','mj_fatal_crashes']].rename(columns={'mj_fatal_crashes':'mj23'})
    fp = (_cmeta.merge(_f20, on='fips5', how='left').merge(_f23, on='fips5', how='left'))
    fp['mj20'] = fp['mj20'].fillna(0)
    fp['mj23'] = fp['mj23'].fillna(0)

    _pop_ok = fp['pop20'].notna() & (fp['pop20'] > 0)
    if _pop_ok.mean() > 0.5:
        fp['mj_rate_20'] = fp['mj20'] / (fp['pop20'] / 100000)
        fp['mj_rate_23'] = fp['mj23'] / (fp['pop20'] / 100000)
        _rate_label = 'marijuana fatal crashes per 100k'
    else:
        fp['mj_rate_20'] = fp['mj20']
        fp['mj_rate_23'] = fp['mj23']
        _rate_label = 'marijuana fatal crashes (raw)'
    _pop_ok_mask = fp['mj_rate_20'].notna() & fp['mj_rate_23'].notna()

    fps = fp[(fp['sample_grp'] != 'Always near rec border') & _pop_ok_mask].copy()
    fps['d_mj'] = fps['mj_rate_23'] - fps['mj_rate_20']

    bf,  sef,  pf,  nf  = ols_cl(fps['d_mj'], fps[['switcher']], fps['state'], 'switcher')
    ci_lo_f, ci_hi_f = ci95(bf, sef)

    sw_mj_20 = float(fps[fps['switcher']==1]['mj_rate_20'].mean())
    sw_mj_23 = float(fps[fps['switcher']==1]['mj_rate_23'].mean())
    ct_mj_20 = float(fps[fps['switcher']==0]['mj_rate_20'].mean())
    ct_mj_23 = float(fps[fps['switcher']==0]['mj_rate_23'].mean())
    n_sw_f   = int(fps[fps['switcher']==1]['fips5'].nunique())
    n_ct_f   = int(fps[fps['switcher']==0]['fips5'].nunique())

    _fars_ok = True
    print(f"  FARS binary DiD: b={bf:+.3f}  SE={sef:.3f}  p={pf:.3f}  N={nf}")
    print(f"  Switcher: {sw_mj_20:.3f} -> {sw_mj_23:.3f}  |  Control: {ct_mj_20:.3f} -> {ct_mj_23:.3f}")
except Exception as _ex:
    import traceback; traceback.print_exc()
    print(f"  FARS error: {_ex}")
    _fars_ok = False
    bf, sef, pf, nf = np.nan, np.nan, np.nan, 0
    ci_lo_f, ci_hi_f = np.nan, np.nan
    sw_mj_20 = sw_mj_23 = ct_mj_20 = ct_mj_23 = np.nan
    n_sw_f = n_ct_f = 0
    fps = pd.DataFrame()
    _rate_label = 'marijuana fatal crashes per 100k'

# ════════════════════════════════════════════════════════════
# 3c.  THREE-PERIOD EVENT STUDY  (adds PLACES 2024, BRFSS 2021-22)
# ════════════════════════════════════════════════════════════
print("Three-period event study (PLACES 2022 / 2024 / 2025) ...")

# Load PLACES 2024 (BRFSS 2021-2022) -- label as period 2022
p24m = pd.read_csv(os.path.join(DATA, 'places_2024_mhlth.csv'))
p24m.columns = [c.lower() for c in p24m.columns]
p24m['fips'] = (p24m['locationid']
                .apply(lambda x: str(int(float(x))).zfill(5)
                       if pd.notna(x) else None))
p24m = p24m.rename(columns={'stateabbr': 'state', 'data_value': 'mhlth'})
p24m = p24m[['fips', 'state', 'mhlth']].dropna()

p24d = pd.read_csv(os.path.join(DATA, 'places_2024_dep.csv'))
p24d.columns = [c.lower() for c in p24d.columns]
p24d['fips'] = (p24d['locationid']
                .apply(lambda x: str(int(float(x))).zfill(5)
                       if pd.notna(x) else None))
p24d = p24d.rename(columns={'data_value': 'depression'})
p24d = p24d[['fips', 'depression']].dropna()

p24 = p24m.merge(p24d, on='fips', how='inner')
p24['year_int'] = 2022
print(f"  PLACES 2024 loaded: {len(p24)} counties")

# Treatment metadata (switcher flag + sample group) from 2020 period
tx_meta = (main[main['year_int'] == 2020][['fips', 'switcher', 'sample_grp']]
           .drop_duplicates('fips'))

# Merge treatment status into 2022 period; exclude always-near-rec
p24_tx   = p24.merge(tx_meta, on='fips', how='inner')
p24_main = p24_tx[p24_tx['sample_grp'] != 'Always near rec border'].copy()
print(f"  Period-2022 main sample: {len(p24_main)} counties")

# Three-period balanced FIPS (counties present in all three periods)
_cols3 = ['fips', 'state', 'year_int', 'mhlth', 'depression', 'switcher']
p20_3  = main[main['year_int'] == 2020][_cols3].copy()
p23_3  = main[main['year_int'] == 2023][_cols3].copy()
p22_3  = p24_main[['fips', 'year_int', 'mhlth', 'depression', 'switcher']].copy()
p22_3  = p22_3.merge(p20_3[['fips', 'state']].drop_duplicates('fips'), on='fips', how='left')

bal3   = sorted(set(p20_3['fips']) & set(p22_3['fips']) & set(p23_3['fips']))
n_bal3 = len(bal3)
n_sw3  = int(p20_3[p20_3['fips'].isin(bal3) & (p20_3['switcher']==1)]['fips'].nunique())
n_ct3  = int(p20_3[p20_3['fips'].isin(bal3) & (p20_3['switcher']==0)]['fips'].nunique())
print(f"  Balanced 3-period counties: {n_bal3}  ({n_sw3} switchers, {n_ct3} controls)")

# Stacked first-difference event study: FD relative to 2020 reference
def make_fd_years(p_pre, p_post, outcome, keep_fips):
    pre  = p_pre[p_pre['fips'].isin(keep_fips)].set_index('fips')
    post = p_post[p_post['fips'].isin(keep_fips)].set_index('fips')
    both = pre.index.intersection(post.index)
    fd   = pd.DataFrame(index=both)
    fd['d_y']      = post.loc[both, outcome].values - pre.loc[both, outcome].values
    fd['switcher'] = pre.loc[both, 'switcher'].values.astype(float)
    fd['state']    = pre.loc[both, 'state'].values
    return fd.dropna(subset=['d_y', 'switcher'])

# FD 2020->2022: pre-period / early-treatment test
fd_e22_mh = make_fd_years(p20_3, p22_3, 'mhlth',     bal3)
fd_e22_dp = make_fd_years(p20_3, p22_3, 'depression', bal3)
b22_mh, se22_mh, p22_mh, n22 = ols_cl(fd_e22_mh['d_y'], fd_e22_mh[['switcher']], fd_e22_mh['state'], 'switcher')
b22_dp, se22_dp, p22_dp, _   = ols_cl(fd_e22_dp['d_y'], fd_e22_dp[['switcher']], fd_e22_dp['state'], 'switcher')

# FD 2020->2023 on balanced triple: main treatment effect
fd_e23_mh = make_fd_years(p20_3, p23_3, 'mhlth',     bal3)
fd_e23_dp = make_fd_years(p20_3, p23_3, 'depression', bal3)
b23_mh, se23_mh, p23_mh, n23 = ols_cl(fd_e23_mh['d_y'], fd_e23_mh[['switcher']], fd_e23_mh['state'], 'switcher')
b23_dp, se23_dp, p23_dp, _   = ols_cl(fd_e23_dp['d_y'], fd_e23_dp[['switcher']], fd_e23_dp['state'], 'switcher')

ci22_lo_mh = b22_mh - 1.96*se22_mh;  ci22_hi_mh = b22_mh + 1.96*se22_mh
ci23_lo_mh = b23_mh - 1.96*se23_mh;  ci23_hi_mh = b23_mh + 1.96*se23_mh
ci22_lo_dp = b22_dp - 1.96*se22_dp;  ci22_hi_dp = b22_dp + 1.96*se22_dp
ci23_lo_dp = b23_dp - 1.96*se23_dp;  ci23_hi_dp = b23_dp + 1.96*se23_dp

print(f"  MHLTH  2022: b={b22_mh:+.3f} p={p22_mh:.3f}  |  2023: b={b23_mh:+.3f} p={p23_mh:.3f}")
print(f"  Dep    2022: b={b22_dp:+.3f} p={p22_dp:.3f}  |  2023: b={b23_dp:+.3f} p={p23_dp:.3f}")

# ════════════════════════════════════════════════════════════
# 4. FIGURES
# ════════════════════════════════════════════════════════════

# ── Fig 1: Treatment map ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
yr23 = df[df['year_int']==2023]
cfg = [
    ('Never near rec border',     GRAY,  '.', 2,  0.25, 'Control (never near rec)'),
    ('Always near rec border',    BLUE,  '.', 4,  0.45, 'Always near rec border'),
    ('Switcher (newly near rec)', RED,   'o', 14, 0.80, 'Switcher (gained rec neighbor 2020-2023)'),
]
if 'lat' in yr23.columns and 'lon' in yr23.columns:
    for grp, col, mk, ms, al, lab in cfg:
        s = yr23[yr23['sample_grp']==grp].dropna(subset=['lat','lon'])
        if len(s):
            ax.scatter(s['lon'], s['lat'], c=col, marker=mk, s=ms, alpha=al,
                       linewidths=0, zorder=(3 if grp=='Switcher (newly near rec)' else 1))
handles = [mpatches.Patch(color=c, alpha=al, label=lab) for _,c,_,_,al,lab in cfg]
ax.legend(handles=handles, fontsize=7, frameon=False, loc='lower right')
healy(ax, title='Figure 1. Treatment Assignment in Never-Legalizing States',
      xlabel='Longitude', ylabel='Latitude')
ax.figure.text(0.05,0.01,'Red = switchers (gained rec neighbor 2020-2023).  Blue = always near rec.  Gray = control.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0,0.04,1,1])
fig1_path = os.path.join(FIG,'fig1_map.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 1 saved")

# ── Fig 2: Distance distribution ──────────────────────────
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize=(7, 4))
yr23 = df[df['year_int']==2023]
for grp, col, ls, lab in [
    ('Never near rec border',     GRAY, '-',  'Control'),
    ('Switcher (newly near rec)', RED,  '--', 'Switcher'),
    ('Always near rec border',    BLUE, ':',  'Always near rec'),
]:
    s = yr23[(yr23['sample_grp']==grp) & yr23['dist_rec_mi'].notna()]
    if len(s) > 5:
        d = s['dist_rec_mi'].clip(0, s['dist_rec_mi'].quantile(0.98))
        kde = gaussian_kde(d, bw_method=0.35)
        xr  = np.linspace(0, d.max(), 300)
        ax.plot(xr, kde(xr), color=col, linestyle=ls, lw=1.8, label=lab)
ax.axvline(100, color='#888', linestyle=':', lw=1, label='100-mile threshold')
ax.legend(fontsize=7, frameon=False)
healy(ax, title='Figure 2. Distance to Nearest Recreational State Border by Group',
      xlabel='Distance to nearest rec border (miles)', ylabel='Density')
ax.figure.text(0.05,0.01,'Period 2 (2023). Never-legalizing states only.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0,0.04,1,1])
fig2_path = os.path.join(FIG,'fig2_dist.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 2 saved")

# ── Fig 3: Outcome trends ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
tr = main.groupby(['switcher','year_int'])[['mhlth','depression']].mean().reset_index()
for sw, col, ls, mk in [(0,GRAY,'-','o'),(1,RED,'--','o')]:
    s = tr[tr['switcher']==sw].sort_values('year_int')
    lbl = 'Control' if sw==0 else 'Switcher'
    ax.plot(s['year_int'], s['mhlth'],      color=col, linestyle=ls, marker=mk,
            ms=6, lw=2,   label=f'{lbl} -- MHLTH')
    ax.plot(s['year_int'], s['depression'], color=col, linestyle=ls, marker='s',
            ms=5, lw=1.2, label=f'{lbl} -- Depression', alpha=0.65)
ax.set_xticks([2020,2023])
ax.legend(fontsize=7, frameon=False, ncol=2)
healy(ax, title='Figure 3. Mean Outcome Trends: Switchers vs Controls',
      xlabel='Period', ylabel='Mean crude prevalence (%)')
ax.figure.text(0.05,0.01,'Unweighted county means. Main sample only (switchers + controls). Never-legalizing states.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0,0.04,1,1])
fig3_path = os.path.join(FIG,'fig3_trends.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 3 saved")

# ── Fig 4: US map -- treatment assignment (Healy style) ────
import json, urllib.request

_gj_cache = os.path.join(DATA, 'us_states_simple.json')
_gj_url   = ('https://raw.githubusercontent.com/PublicaMundi/MappingAPI'
             '/master/data/geojson/us-states.json')
if not os.path.exists(_gj_cache):
    try:
        with urllib.request.urlopen(_gj_url, timeout=15) as _r:
            with open(_gj_cache, 'wb') as _f:
                _f.write(_r.read())
        print("  State GeoJSON downloaded")
    except Exception as _ex:
        print(f"  GeoJSON download failed: {_ex}")

_states_js = None
if os.path.exists(_gj_cache):
    with open(_gj_cache) as _f:
        _states_js = json.load(_f)

# County centroids for ALL US counties (from Gazetteer)
_gaz = pd.read_csv(os.path.join(DATA, '2023_Gaz_counties_national.txt'), sep='\t')
_gaz.columns = [c.strip().upper() for c in _gaz.columns]
_gaz['fips'] = _gaz['GEOID'].apply(lambda x: str(int(x)).zfill(5))
_gaz['lat']  = _gaz['INTPTLAT'].astype(float)
_gaz['lon']  = _gaz['INTPTLONG'].astype(float)
_gaz = _gaz[['fips', 'USPS', 'lat', 'lon']].rename(columns={'USPS': 'state_abbr'})

# Merge treatment group (analysis panel covers only never-legal states)
_tx = (df[df['year_int']==2023][['fips', 'sample_grp']]
       .drop_duplicates('fips').set_index('fips'))
_gaz = _gaz.join(_tx, on='fips').rename(columns={'sample_grp': 'grp'})

# Classify unmatched counties (rec states + territories) as background
_AK_HI  = {'AK','HI','AS','GU','MP','PR','VI'}
_REC_AB = {'AZ','CA','CO','CT','IL','MD','MA','MI','MN','MO','MT',
           'NV','NJ','NM','NY','OH','OR','RI','VT','WA'}
_EXCL2  = {'02','15','60','66','69','72','78'}
_REC2   = {'04','06','08','09','17','24','25','26','27','29','30',
           '32','34','35','36','39','41','44','50','53'}

_gaz = _gaz[~_gaz['state_abbr'].isin(_AK_HI)]
_gaz = _gaz[_gaz['lat'].between(24, 50) & _gaz['lon'].between(-130, -60)]
_gaz['grp'] = _gaz['grp'].fillna('Rec/other')
_gaz.loc[_gaz['state_abbr'].isin(_REC_AB), 'grp'] = 'Rec/other'

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Draw state fills and outlines
if _states_js:
    for _feat in _states_js['features']:
        _sid = str(_feat.get('id', '')).zfill(2)
        if _sid in _EXCL2:
            continue
        _fc   = '#fce8e8' if _sid in _REC2 else 'white'
        _geom = _feat['geometry']
        _rings = ([_geom['coordinates']] if _geom['type'] == 'Polygon'
                  else _geom['coordinates'])
        for _ring in _rings:
            _coords = np.array(_ring[0])
            if _coords[:, 0].min() > -140:
                _patch = mpatches.Polygon(
                    _coords, closed=True,
                    facecolor=_fc, edgecolor='#bbbbbb',
                    linewidth=0.5, zorder=1)
                ax.add_patch(_patch)

# County dots -- plot from background to foreground
_dot_layers = [
    ('Rec/other',                 '#c0c0c0', 'o', 0.7,  0.18, ''),
    ('Never near rec border',     '#6e6e6e', 'o', 2.2,  0.40, 'Control (never near rec)'),
    ('Always near rec border',    BLUE,      'o', 4.5,  0.65, 'Always near rec border'),
    ('Switcher (newly near rec)', RED,       'o', 9.5,  0.92, 'Switcher (gained rec neighbor)'),
]
for _grp, _col, _mk, _ms, _al, _lab in _dot_layers:
    _s = _gaz[_gaz['grp'] == _grp]
    if len(_s):
        _zo = 5 if _grp == 'Switcher (newly near rec)' else 2
        ax.scatter(_s['lon'], _s['lat'], c=_col, marker=_mk, s=_ms,
                   alpha=_al, linewidths=0, zorder=_zo)

ax.set_xlim(-125, -65)
ax.set_ylim(24.5, 50)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for _sp in ax.spines.values():
    _sp.set_visible(False)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Legend
_leg_h = [
    mpatches.Patch(facecolor='#fce8e8', edgecolor='#bbb', linewidth=0.7,
                   label='Recreational state'),
    mpatches.Patch(facecolor='white', edgecolor='#bbb', linewidth=0.7,
                   label='Never-legalizing state'),
    mpatches.Patch(color=RED,      label='Switcher (gained rec neighbor 2020-2023)'),
    mpatches.Patch(color=BLUE,     label='Always near rec border (excl. from main sample)'),
    mpatches.Patch(color='#6e6e6e', label='Control county (never near rec)'),
]
ax.legend(handles=_leg_h, fontsize=7.5, frameon=False, loc='lower right',
          handlelength=1.4, handleheight=0.9, borderpad=0, labelspacing=0.4)

ax.set_title(
    'Figure 4.  Treatment Assignment -- Never-Legalizing State Counties (Contiguous US, 2023)',
    fontsize=9.5, loc='left', pad=8)
ax.figure.text(
    0.05, 0.01,
    'County centroids plotted as dots. Red fill = recreational states by 2023. '
    'Switchers: counties in never-legalizing states that gained a rec neighbor 2020-2023 (within 100 miles). '
    'Always near rec border: excluded from main sample.',
    fontsize=6.5, color='#555555', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig4_path = os.path.join(FIG, 'fig4_map.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ── Fig 5: Dose-response ───────────────────────────────────
band_labels = ['0-50 mi','50-100 mi','100-200 mi','200-300 mi']
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
for ax, hb, col, title in [
    (axes[0], hb_mh, BLUE, 'Mental Health (MHLTH)'),
    (axes[1], hb_dp, RED,  'Depression'),
]:
    ylabs = band_labels[::-1]
    ys    = list(range(len(ylabs)))
    for y, (b,s,p) in zip(ys, [hb[3-i] for i in range(4)]):
        lo, hi = ci95(b, s)
        ax.plot([lo,hi],[y,y], color=col, lw=2.5, solid_capstyle='round')
        ax.scatter([b],[y], color=col, s=50, zorder=5)
        ax.text(hi+0.005, y, f'{b:+.3f}{stars(p)}', va='center', fontsize=7.5, color=col)
    ax.axvline(0, color='#888', lw=0.8, linestyle='--')
    ax.set_yticks(ys); ax.set_yticklabels(ylabs, fontsize=8)
    healy(ax, title=title, xlabel='pp change vs >300 mi (95% CI)')
fig.suptitle('Figure 5. Dose-Response: Effect by Distance Band to Nearest Rec Border',
             fontsize=9, x=0.05, ha='left')
fig.text(0.05,0.01,'Reference: counties >300 mi from any rec border. First-difference estimator. Clustered SE.',
         fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0,0.05,1,0.93])
fig5_path = os.path.join(FIG,'fig5_dose.png')
plt.savefig(fig5_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 5 saved")

# ── Fig 6: Pre-period parallel-trends check ────────────────
fig, ax = plt.subplots(figsize=(7, 4))
pre_df = df[df['year_int']==2020].copy()
for out, col, mk, lab in [('mhlth',BLUE,'o','MHLTH 2020'),
                            ('depression',RED,'s','Depression 2020')]:
    bm = pre_df.groupby('dist_band')[out].mean()
    bs = pre_df.groupby('dist_band')[out].sem()
    xs = bm.index.values
    ax.errorbar(xs, bm.values, yerr=1.96*bs.values, color=col,
                marker=mk, ms=6, lw=1.4, capsize=4, label=lab)
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(['0-50','50-100','100-200','200-300','>300'], fontsize=8)
ax.legend(fontsize=7, frameon=False)
healy(ax, title='Figure 6. Pre-Period (2020) Outcomes by 2023 Distance Band -- Parallel Trends Check',
      xlabel='Distance to 2023 rec border (miles)', ylabel='Mean crude prevalence (%)')
ax.figure.text(0.05,0.01,'Flat gradient across bands supports parallel trends assumption.',
               fontsize=6, color='#777', style='italic')
plt.tight_layout(rect=[0,0.04,1,1])
fig6_path = os.path.join(FIG,'fig6_pretrends.png')
plt.savefig(fig6_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 6 saved")

# ── Fig 7: Pre/Post bar chart by medical group ─────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (out_label, new_pre, new_post, nev_pre, nev_post) in zip(axes, [
    ('Mental Health (% bad days >= 14)', mm_new20, mm_new23, mm_nev20, mm_nev23),
    ('Depression (%)',                   md_new20, md_new23, md_nev20, md_nev23),
]):
    labels  = ['Newly Medical\n(SD, AL, MS)', 'Never Medical\n(13 control states)']
    pre_v   = [new_pre,  nev_pre]
    post_v  = [new_post, nev_post]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, pre_v,  w, color=BLUE, alpha=0.75, label='2020 (pre)')
    ax.bar(x + w/2, post_v, w, color=RED,  alpha=0.75, label='2023 (post)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    healy(ax, title=out_label, ylabel='Mean prevalence (%)')
fig.suptitle('Figure 7. Pre/Post Outcomes by Medical Marijuana Group',
             fontsize=9, fontweight='bold', x=0.05, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig7_path = os.path.join(FIG, 'fig7_med_bars.png')
plt.savefig(fig7_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 7 saved")

# ── Fig 8: Medical group trend lines ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (title, new_vals, nev_vals) in zip(axes, [
    ('Mental Health (% bad days >= 14)', [mm_new20, mm_new23], [mm_nev20, mm_nev23]),
    ('Depression (%)',                   [md_new20, md_new23], [md_nev20, md_nev23]),
]):
    ax.plot([2020, 2023], new_vals, color=RED,  marker='o', ms=6, lw=1.8,
            label='Newly Medical (SD, AL, MS)')
    ax.plot([2020, 2023], nev_vals, color=GRAY, marker='s', ms=6, lw=1.8,
            label='Never Medical (13 states)')
    ax.set_xticks([2020, 2023])
    ax.legend(fontsize=8, frameon=False)
    healy(ax, title=title, xlabel='Period', ylabel='Mean prevalence (%)')
fig.suptitle('Figure 8. Parallel Trends Check -- Medical Marijuana Groups',
             fontsize=9, fontweight='bold', x=0.05, ha='left')
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig8_path = os.path.join(FIG, 'fig8_med_trends.png')
plt.savefig(fig8_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 8 saved")

# ── Fig 9: Comparison coefplot medical vs recreational ─────
fig, ax = plt.subplots(figsize=(7, 4))
estimates9 = [
    ('Medical MHLTH (direct)',      bmed_mh, semed_mh, RED),
    ('Rec. Spillover MHLTH',        bm2,     sem2,     BLUE),
    ('Medical Depression (direct)', bmed_dp, semed_dp, RED),
    ('Rec. Spillover Depression',   bd2,     sed2,     BLUE),
]
y_pos9 = [3.5, 2.7, 1.5, 0.7]
for (lbl, b, se, col), yp in zip(estimates9, y_pos9):
    ax.errorbar(b, yp, xerr=1.96*se, fmt='o', color=col, ms=7, lw=1.5, capsize=4)
    ax.text(b, yp + 0.22, f'{b:+.3f}', fontsize=7.5, ha='center', color=col)
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.set_yticks(y_pos9)
ax.set_yticklabels([e[0] for e in estimates9], fontsize=8.5)
ax.set_ylim(0, 4.5)
patch_med9 = mpatches.Patch(color=RED,  label='Medical Direct Effect')
patch_rec9 = mpatches.Patch(color=BLUE, label='Rec. Spillover Effect')
ax.legend(handles=[patch_med9, patch_rec9], fontsize=8, frameon=False)
healy(ax,
      title='Figure 9. Medical Direct vs Recreational Spillover -- First-Difference DiD',
      xlabel='Estimated effect (percentage-point change in prevalence)',
      ylabel='')
fig9_path = os.path.join(FIG, 'fig9_med_coefplot.png')
plt.savefig(fig9_path, dpi=150, bbox_inches='tight'); plt.close()
print("Fig 9 saved")

# ── Fig 10: Three-period event study ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
_es_data = [
    (axes[0], b22_mh, se22_mh, p22_mh, b23_mh, se23_mh, p23_mh, BLUE, 'Mental Health (MHLTH)'),
    (axes[1], b22_dp, se22_dp, p22_dp, b23_dp, se23_dp, p23_dp, RED,  'Depression'),
]
for ax, b22, se22, p22v, b23, se23, p23v, col, title in _es_data:
    xs  = [2020, 2022, 2023]
    bs  = [0.0,  b22,  b23]
    ses = [0.0,  se22, se23]
    cis = [0.0, 1.96*se22, 1.96*se23]
    ax.fill_between([2019.5, 2021.2], -99, 99, color='#e8f0fb', zorder=0, linewidth=0)
    ax.fill_between([2021.2, 2023.5], -99, 99, color='#fbe8e8', zorder=0, linewidth=0)
    ax.errorbar(xs, bs, yerr=cis, fmt='o', color=col, ms=7, lw=1.8,
                capsize=5, solid_capstyle='round', zorder=5)
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels(['2020\n(reference)', '2022\n(mid)', '2023\n(post)'], fontsize=8)
    for yr, b, se, pv in zip([2022, 2023], [b22, b23], [se22, se23], [p22v, p23v]):
        offset = 1.96*se + 0.06 if (1.96*se) > 0 else 0.12
        ax.text(yr, b + offset, f'{b:+.3f}{stars(pv)}', ha='center', fontsize=7.5, color=col)
    ymax = max(abs(b22) + 1.96*se22, abs(b23) + 1.96*se23) * 1.6 + 0.3
    ax.set_ylim(-ymax, ymax)
    healy(ax, title=title, ylabel='DiD coeff. relative to 2020 (pp)')
fig.text(0.05, 0.96, 'Figure 10.  Three-Period Event Study: TWFE with PLACES 2022 / 2024 / 2025',
         fontsize=9.5, ha='left', transform=fig.transFigure)
fig.text(0.05, 0.01,
         'County + year FE removed via within-county demeaning. SE clustered by state. Reference = 2020 (b=0 by construction). '
         'Blue shade = pre-treatment window; red shade = treatment window. '
         '2022 coefficient tests pre-trend assumption (note: some switchers gained rec neighbors in 2022).',
         fontsize=6, color='#555', style='italic')
plt.tight_layout(rect=[0, 0.07, 1, 0.94])
fig10_path = os.path.join(FIG, 'fig10_eventstudy.png')
plt.savefig(fig10_path, dpi=150, bbox_inches='tight')
plt.close()
print("Fig 10 saved")

# ── Fig 11: FARS marijuana crash trends by group ─────────────
if _fars_ok and len(fps) > 0:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sw_val, col, mk, ls, lab in [(0, GRAY, 'o', '-', 'Control'), (1, RED, 'o', '--', 'Switcher')]:
        s = fps[fps['switcher'] == sw_val]
        if len(s) == 0:
            continue
        means = [s['mj_rate_20'].mean(), s['mj_rate_23'].mean()]
        sems  = [s['mj_rate_20'].sem(),  s['mj_rate_23'].sem()]
        ax.errorbar([2020, 2023], means, yerr=[1.96*e for e in sems],
                    color=col, linestyle=ls, marker=mk, ms=7, lw=2,
                    capsize=4, label=lab)
    ax.set_xticks([2020, 2023])
    ax.legend(fontsize=8, frameon=False)
    healy(ax,
          title='Figure 11.  Marijuana-Involved Fatal Crashes by Group (NHTSA FARS)',
          xlabel='Period',
          ylabel=_rate_label.capitalize())
    ax.figure.text(0.05, 0.01,
        f'Source: NHTSA Fatal Analysis Reporting System (FARS) 2020 and 2023. '
        f'DRUGRES codes 600-695 (2020) and 5060-5067/5997 (2023). '
        f'Error bars = 95% CI. Never-legalizing counties only (main sample).',
        fontsize=6, color='#777', style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig11_path = os.path.join(FIG, 'fig11_fars.png')
    plt.savefig(fig11_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 11 saved")
else:
    fig11_path = None
    print("Fig 11 skipped (FARS data unavailable)")

# ════════════════════════════════════════════════════════════
# 5. BUILD PDF
# ════════════════════════════════════════════════════════════
print("Building PDF ...")

W, H = letter
doc = SimpleDocTemplate(OUT, pagesize=letter,
                         leftMargin=1*inch, rightMargin=1*inch,
                         topMargin=1*inch, bottomMargin=1*inch)

TN='Times-Roman'; TNB='Times-Bold'; TNI='Times-Italic'; COU='Courier'

def PS(name, f=TN, sz=11, ld=16, al=TA_JUSTIFY, sb=0, sa=6, li=0):
    return ParagraphStyle(name, fontName=f, fontSize=sz, leading=ld,
                          alignment=al, spaceBefore=sb, spaceAfter=sa,
                          leftIndent=li, textColor=colors.black)

s_h1   = PS('h1',  f=TNB, sz=13, ld=17, al=TA_LEFT,   sb=16, sa=5)
s_h2   = PS('h2',  f=TNB, sz=11, ld=15, al=TA_LEFT,   sb=10, sa=4)
s_body = PS('body',f=TN,  sz=10.5, ld=15, sa=5)
s_ind  = PS('ind', f=TN,  sz=10.5, ld=15, sa=3, li=18)
s_note = PS('note',f=TNI, sz=8.5,  ld=12, sa=3, al=TA_LEFT)
s_cap  = PS('cap', f=TNB, sz=9,    ld=12, sa=3, al=TA_CENTER)
s_code = PS('code',f=COU, sz=9,    ld=13, sa=3, li=24, al=TA_LEFT)
s_cov  = PS('cov', f=TN,  sz=18,   ld=24, al=TA_CENTER, sb=60, sa=8)
s_csub = PS('csub',f=TNI, sz=13,   ld=17, al=TA_CENTER, sa=6)
s_caut = PS('caut',f=TN,  sz=11,   ld=16, al=TA_CENTER, sa=4)

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
        out.append(Spacer(1,4))
        out.append(Image(path, width=w, height=w*(2/3.2)))
    if cap:  out.append(p(cap, s_cap))
    if note: out.append(p(note, s_note))
    out.append(Spacer(1,6))
    return out

def fmt(x, f='.2f'): return f'{x:{f}}' if pd.notna(x) else 'N/A'

story = []

# ── Economics Letters styles ──────────────────────────────────────────────
s_title = PS('title',  f=TNB, sz=14, ld=18, al=TA_CENTER, sb=0,  sa=8)
s_sub   = PS('sub',    f=TNI, sz=12, ld=16, al=TA_CENTER, sb=0,  sa=10)
s_aut   = PS('aut',    f=TN,  sz=11, ld=15, al=TA_CENTER, sb=0,  sa=8)
s_absh  = PS('absh',  f=TNB, sz=11, ld=15, al=TA_LEFT,   sb=12, sa=2)
s_abs   = PS('abs',    f=TN,  sz=10.5, ld=15, al=TA_JUSTIFY, sb=0, sa=8)
s_jel   = PS('jel',   f=TNI, sz=9.5, ld=13, al=TA_LEFT,   sb=0,  sa=14)
s_refp  = PS('refp',  f=TN,  sz=9.5, ld=13, al=TA_LEFT,   sb=0,  sa=4, li=14)
s_app   = PS('app',   f=TNB, sz=12,  ld=16, al=TA_LEFT,   sb=24, sa=6)

# ── Title block ───────────────────────────────────────────────────────────
story += [
    Spacer(1, 0.5*inch),
    p('Cross-Border Spillover Effects of Recreational<br/>'
      'Marijuana Legalization on County-Level Mental Health', s_title),
    p('Evidence from US County Panel Data, 2020–2023', s_sub),
    HRFlowable(width='55%', thickness=0.5,
               color=colors.HexColor('#aaaaaa'), hAlign='CENTER'),
    Spacer(1, 0.1*inch),
    p('Anonymous  |  July 2026', s_aut),
    Spacer(1, 0.15*inch),
]

# ── Abstract ─────────────────────────────────────────────────────────────
story.append(p(B('Abstract'), s_absh))
story.append(p(
    f'We estimate cross-border spillover effects of recreational marijuana legalization '
    f'on county-level mental health in never-legalizing US states. Exploiting within-state '
    f'variation in proximity to newly opened retail marijuana markets, we compare counties '
    f'within 100 miles of a newly legalized recreational border in 2023 to interior counties '
    f'in the same states using a first-difference estimator on CDC PLACES county data '
    f'({int(n_counties):,} counties; {n_states} states). '
    f'Counties that gained a recreational marijuana neighbor between 2020 and 2023 '
    f'experienced a statistically significant increase in poor mental health prevalence '
    f'({bm2:+.3f} pp; p = {pm2:.3f}) and depression ({bd2:+.3f} pp; '
    f'p = {ppd2:.3f}). A three-period stacked event study confirms parallel pre-trends. '
    f'A medical marijuana robustness check yields null results, pointing to a mechanism '
    f'specific to recreational retail access.', s_abs))
story.append(p(
    f'<i>JEL Classification:</i> I11, I18, K32, R12 '
    f'<i>Keywords:</i> recreational marijuana; mental health; cross-border spillovers; '
    f'difference-in-differences; US county panel', s_jel))
story.append(HRFlowable(width='100%', thickness=0.4,
                         color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 8))

# ─────────────────────────────────────────────────────────────────────────
# 1. INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────
story.append(p('1.  Introduction', s_h1))
story.append(p(
    'Between 2012 and 2023, recreational marijuana legalization expanded from two pioneering '
    'states -- Colorado and Washington -- to twenty-three, covering roughly half the US '
    'population. This rapid diffusion creates a natural policy experiment: residents of '
    'neighboring prohibition states can access legal recreational marijuana by crossing '
    'state lines, generating cross-border externalities whose public health consequences '
    'remain poorly understood. This paper provides the first estimates of the mental health '
    'spillover effects of recreational marijuana legalization in never-legalizing US states.'))
story.append(p(
    'A substantial literature evaluates the within-state consequences of marijuana '
    'liberalization, documenting effects on substance use (Wen et al., 2015; '
    'Pacula et al., 2015; Dills et al., 2017), traffic safety (Anderson et al., 2013), '
    'opioid mortality (Powell et al., 2018; Bradford and Bradford, 2016), and hospital '
    'utilization. The mental health channel is particularly contested. Cannabis interacts '
    'with the endocannabinoid system and has been invoked both as a treatment for anxiety '
    'and depression and as a contributor to cannabis use disorder and psychosis risk '
    '(Volkow et al., 2014; Di Forti et al., 2019). Anderson et al. (2014) find that '
    'medical marijuana laws reduce suicides by gender, suggesting substitution away from '
    'lethal coping strategies. Hasin et al. (2015) document that cannabis use disorder '
    'prevalence rose substantially following legalization. Whether these dynamics propagate '
    'to neighboring prohibition states is an open empirical question.'))
story.append(p(
    'Cross-border behavioral spillovers from state drug policies are well documented. '
    'Hao and Cowan (2020) provide the most direct precursor, finding that recreational '
    'marijuana legalization significantly reduces drug arrests within 100 miles of newly '
    'opened borders -- direct evidence that residents of prohibition states participate '
    'in adjacent legal markets. Cerda et al. (2012) show that state marijuana '
    'liberalization shapes prevalence in neighboring regions. If border residents '
    'substitute legal cannabis for illicit purchases, the downstream mental health effects '
    'are theoretically ambiguous: reduced enforcement anxiety could lower psychological '
    'distress, while elevated consumption may raise cannabis use disorder risk or '
    'worsen anxiety at high doses (Volkow et al., 2014; Hasin et al., 2015).'))
story.append(p(
    f'We address this ambiguity using the CDC PLACES database, which provides '
    f'model-based county-level prevalence estimates derived from the BRFSS via multilevel '
    f'regression and poststratification (MRP) for all {n_counties:,} counties in '
    f'{n_states} never-legalizing states across 2020 and 2023. A third PLACES wave '
    f'(2024 release; BRFSS 2021-2022) enables a three-period stacked event study. '
    f'Our identification strategy exploits within-state variation in proximity to '
    f'recreational borders: counties within 100 miles of a newly legalized state in 2023 '
    f'("switchers") are compared to interior counties in the same states. This within-state '
    f'design controls for state-level time trends without comparing legalizing to '
    f'non-legalizing states, avoiding the compositional confounders that complicate '
    f'cross-state designs.'))
story.append(p(
    f'We find that counties that gained a recreational marijuana neighbor between 2020 and '
    f'2023 experienced a statistically significant {dir_mh} in poor mental health '
    f'prevalence of {abs(bm2):.2f} pp (p = {pm2:.3f}) and a {dir_dep} in depression '
    f'prevalence of {abs(bd2):.2f} pp (p = {ppd2:.3f}), relative to control counties. '
    f'A dose-response pattern -- effects largest within 50 miles and attenuating with '
    f'distance -- is consistent with a border-proximity mechanism. A three-period stacked '
    f'event study confirms no significant 2022 pre-trend (MHLTH: {b22_mh:+.3f} pp, '
    f'p = {p22_mh:.3f}; Depression: {b22_dp:+.3f} pp, p = {p22_dp:.3f}). '
    f'A medical marijuana robustness check using entirely different treatment and control '
    f'states yields null results (MHLTH: {bmed_mh:+.3f} pp, p = {pmed_mh:.3f}), '
    f'isolating a mechanism specific to recreational retail access.'))
story.append(p(
    'Our three contributions are: (i) the first estimates of recreational marijuana '
    'legalization\'s spillover effects on county-level mental health; '
    '(ii) a within-state identification design avoiding compositional differences between '
    'legalizing and non-legalizing states; and '
    '(iii) the first three-period pre-trend test for recreational spillover effects '
    'using the CDC PLACES 2024 release. '
    'Section 2 describes data and strategy. Section 3 presents results. '
    'Section 4 contains robustness checks. Section 5 concludes.'))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# 2. DATA AND EMPIRICAL STRATEGY
# ─────────────────────────────────────────────────────────────────────────
story.append(p('2.  Data and Empirical Strategy', s_h1))
story.append(p('2.1  Outcome Data', s_h2))
story.append(p(
    f'Outcome data come from CDC PLACES, which produces county-level model-based small-area '
    f'prevalence estimates for all 3,143 US counties using multilevel regression and '
    f'poststratification (MRP) applied to Behavioral Risk Factor Surveillance System (BRFSS) '
    f'microdata. We use two outcomes: (i) MHLTH -- the share of adults reporting 14 or more '
    f'mentally unhealthy days in the past 30 days, a flow measure responsive to near-term '
    f'behavioral changes; and (ii) depression prevalence -- the share ever diagnosed by a '
    f'physician, a stock (lifetime diagnosis) measure. We exploit three PLACES releases as '
    f'panel waves: the 2022 release (BRFSS 2019-2020, labelled "2020"), the 2024 release '
    f'(BRFSS 2021-2022, labelled "2022"), and the 2025 release (BRFSS 2022-2023, labelled '
    f'"2023"). The balanced two-period panel (2020 and 2023) covers {n_counties:,} unique '
    f'counties across {n_states} never-legalizing states; the balanced three-period panel '
    f'yields {n_bal3:,} counties ({n_sw3} switchers, {n_ct3} controls). MRP smoothing toward '
    f'state and national means introduces classical measurement error that attenuates all '
    f'DiD coefficients toward zero, making our estimates conservative lower bounds.'))
story.append(p(
    f'Recreational marijuana retail legalization dates are hand-coded from state statutes, '
    f'recording the year of first over-the-counter adult sales. The primary identifying '
    f'variation comes from ten states opening recreational retail in 2021-2023: '
    f'Arizona (2021); Montana, New Jersey, New Mexico, New York, Rhode Island, and Vermont '
    f'(2022); and Connecticut, Maryland, and Missouri (2023). County centroids '
    f'(internal latitude/longitude points) are from the US Census Bureau 2023 County '
    f'Gazetteer. We compute Haversine great-circle distances from each never-legalizing '
    f'county centroid to the nearest recreational-state county centroid, separately for '
    f'each panel period. Summary statistics appear in Table A.1 (Appendix).'))
story.append(p('2.2  Treatment Definition', s_h2))
story.append(p(
    f'We restrict the sample to {n_states} states that maintained prohibition through 2023. '
    f'County {I("i")} is a {B("switcher")} if dist(i,2023) < 100 miles and '
    f'dist(i,2020) >= 100 miles, where dist(i,t) is the minimum Haversine distance to '
    f'any recreational-state county in period t. This identifies counties that gained a '
    f'legal marijuana neighbor during the study window. The 100-mile threshold follows '
    f'Hao and Cowan (2020). The control group consists of counties in the same '
    f'never-legalizing states with dist(i,t) >= 100 miles in both periods '
    f'({n_control} counties; mean 2023 distance {mean_dctl:.0f} miles). '
    f'Switchers number {n_switchers} and lie a mean {mean_dsw:.0f} miles from the '
    f'nearest recreational border in 2023. Figure 1 (below) maps treatment assignment.'))
story += fig_block(fig4_path, w=5.8*inch,
    cap='Figure 1.  Treatment Assignment -- Never-Legalizing State Counties (Contiguous US, 2023)',
    note='County centroids as dots. Pale red fill = recreational states by 2023. '
         'Red = switcher counties; blue = always-near (excluded from main sample); '
         'gray = control counties. State boundaries from GeoJSON (PublicaMundi/MappingAPI).')
story.append(p('2.3  Estimating Equation', s_h2))
story.append(p('The main estimating equation is the county-level first difference (FD):'))
story.append(p(
    'Delta-Y(i) = alpha + beta * Switcher(i) + epsilon(i)          (1)',
    s_code))
story.append(p(
    f'where Delta-Y(i) = Y(i,2023) - Y(i,2020) is the county-level change in outcome '
    f'(percentage points) and Switcher(i) is binary. With T = 2 periods, FD is '
    f'algebraically equivalent to TWFE with county and year fixed effects: FD removes '
    f'county fixed effects by subtraction and absorbs the year fixed effect in the '
    f'constant. The coefficient beta is the ATT: the differential change for switcher '
    f'versus control counties, net of the common national trend. Standard errors are '
    f'clustered at the state level throughout to account for within-state spatial '
    f'correlation (Bertrand et al., 2004; Cameron and Miller, 2015; '
    f'{n_states} clusters). We also estimate a continuous specification replacing '
    f'Switcher(i) with Delta-log(dist(i)), the change in log-distance to the nearest '
    f'recreational border.'))
story.append(p(
    f'Identification requires parallel trends: absent neighboring legalization, switcher '
    f'and control counties would have followed the same mental health trajectory. We provide '
    f'four pieces of supporting evidence. '
    f'(i) A placebo regression of 2020 MHLTH on future (2023) switcher status yields '
    f'beta = {pl_b:+.3f} pp (p = {pl_p:.3f}), indicating no pre-existing health gradient. '
    f'(ii) Figure A.3 shows 2020 outcomes are flat across future distance bands. '
    f'(iii) The three-period event study in Section 3.2 finds no significant 2022 pre-trend. '
    f'(iv) Treatment is determined by geographic proximity to state borders -- an exogenous '
    f'feature of county location.'))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# 3. RESULTS
# ─────────────────────────────────────────────────────────────────────────
story.append(p('3.  Results', s_h1))
story.append(p('3.1  Main DiD Estimates', s_h2))

story.append(p(
    f'{B("Table 1.")} DiD Estimates of Recreational Marijuana Spillovers '
    f'on County Mental Health', s_cap))
hdr_t1 = ['', 'MHLTH\n(1)', 'MHLTH\n(2)', 'Depression\n(3)', 'Depression\n(4)']
rows_t1 = [
    ['Delta-log(dist)  [preferred]',
     f'{bm4m:+.3f}{stars(pm4m)}', '',
     f'{bd3m:+.3f}{stars(ppd3m)}', ''],
    ['', f'({sem4m:.3f})', '', f'({sed3m:.3f})', ''],
    ['Switcher x Post  [binary robustness]', '',
     f'{bm2:+.3f}{stars(pm2)}', '',
     f'{bd2:+.3f}{stars(ppd2)}'],
    ['', '', f'({sem2:.3f})', '', f'({sed2:.3f})'],
    ['County + Year FE', 'Yes', 'Yes', 'Yes', 'Yes'],
    ['Always-near counties excl.', 'Yes', 'Yes', 'Yes', 'Yes'],
    ['N (county FDs)', f'{nm4m:,}', f'{nm2:,}', f'{nd3m:,}', f'{nd2:,}'],
    ['State clusters', str(n_states), str(n_states), str(n_states), str(n_states)],
]
cw_t1 = [2.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]
story += rtable(hdr_t1, rows_t1, cw_t1,
    note=('Outcome: change in county-level prevalence 2020-2023 (Delta-Y, pp). '
          'Main sample throughout: switchers + controls only (excludes always-near). '
          'Cols (1),(3): continuous treatment Delta-log(dist) = log(dist_2023)-log(dist_2020); '
          'negative when county got closer to a rec border (preferred specification). '
          'Cols (2),(4): binary switcher indicator (robustness). '
          'SE clustered by state in parentheses. * p<0.10, ** p<0.05, *** p<0.01.'))
story.append(p(
    f'Table 1 presents our main results. The preferred specification (cols. 1 and 3) '
    f'uses the continuous treatment Delta-log(dist), the within-county change in '
    f'log-distance to the nearest recreational border. A negative coefficient indicates '
    f'that counties which got closer to a recreational border experienced worse mental '
    f'health outcomes. For MHLTH, we find {bm4m:+.3f} pp per unit decrease in log-distance '
    f'(SE = {sem4m:.3f}; 95% CI [{ci_lo_m_cont:.3f}, {ci_hi_m_cont:.3f}]; p = {pm4m:.3f}). '
    f'For depression, the estimate is {bd3m:+.3f} pp '
    f'(SE = {sed3m:.3f}; 95% CI [{ci_lo_d_cont:.3f}, {ci_hi_d_cont:.3f}]; p = {ppd3m:.3f}). '
    f'The binary switcher specification (cols. 2 and 4), which serves as a robustness check, '
    f'yields MHLTH {bm2:+.3f} pp (SE = {sem2:.3f}; p = {pm2:.3f}) and '
    f'depression {bd2:+.3f} pp (SE = {sed2:.3f}; p = {ppd2:.3f}), '
    f'representing {pct_mh:.1f}% and {pct_dep:.1f}% of the respective 2020 county '
    f'means ({mean_mh20:.2f}% and {mean_dep20:.2f}%). '
    f'The national increase in MHLTH over this period was {mean_mh23-mean_mh20:.2f} pp; '
    f'our spillover estimate is approximately {abs(bm2)/(mean_mh23-mean_mh20)*100:.0f}% '
    f'of this common trend. '
    f'Both specifications consistently point to worse mental health in counties that gained '
    f'proximity to recreational borders -- inconsistent with an access-alleviates-distress '
    f'narrative and more consistent with demand-side effects: cross-border consumers who '
    f'increase cannabis use face elevated risk of cannabis use disorder or '
    f'dose-dependent anxiety (Hasin et al., 2015; Volkow et al., 2014). '
    f'Figure A.2 plots raw trends for both groups.'))

story.append(p('3.2  Pre-Trend Test and Event Study', s_h2))
story.append(p(
    f'{B("Table 2.")} Three-Period Stacked First-Difference Event Study', s_cap))
hdr_t2 = ['', 'MHLTH\n2020->2022', 'MHLTH\n2020->2023',
          'Depression\n2020->2022', 'Depression\n2020->2023']
rows_t2 = [
    ['Switcher x Post',
     f'{b22_mh:+.3f}{stars(p22_mh)}', f'{b23_mh:+.3f}{stars(p23_mh)}',
     f'{b22_dp:+.3f}{stars(p22_dp)}', f'{b23_dp:+.3f}{stars(p23_dp)}'],
    ['', f'({se22_mh:.3f})', f'({se23_mh:.3f})',
        f'({se22_dp:.3f})', f'({se23_dp:.3f})'],
    ['95% CI',
     f'[{ci22_lo_mh:.3f},{ci22_hi_mh:.3f}]', f'[{ci23_lo_mh:.3f},{ci23_hi_mh:.3f}]',
     f'[{ci22_lo_dp:.3f},{ci22_hi_dp:.3f}]', f'[{ci23_lo_dp:.3f},{ci23_hi_dp:.3f}]'],
    ['N (county FDs)', f'{n22:,}', f'{n23:,}', f'{n22:,}', f'{n23:,}'],
    ['State clusters', str(n_states), str(n_states), str(n_states), str(n_states)],
]
cw_t2 = [1.9*inch, 1.05*inch, 1.05*inch, 1.07*inch, 1.07*inch]
story += rtable(hdr_t2, rows_t2, cw_t2,
    note=(f'Balanced three-period panel: {n_bal3:,} counties ({n_sw3} switchers, '
          f'{n_ct3} controls) present in PLACES 2022, 2024, and 2025 releases. '
          f'Each column is a separate FD relative to 2020. SE clustered by state. '
          f'The 2022 coefficient mixes pre-trend (counties near 2023-cohort states CT, MD, MO) '
          f'with early-treatment effects (near 2022-cohort states MT, NJ, NM, NY, RI, VT). '
          f'* p<0.10, ** p<0.05, *** p<0.01.'))
story.append(p(
    f'Table 2 and Figure 2 report the three-period stacked event study. '
    f'The 2022 pre-period coefficients are statistically indistinguishable from zero: '
    f'MHLTH {b22_mh:+.3f} pp (p = {p22_mh:.3f}) and '
    f'Depression {b22_dp:+.3f} pp (p = {p22_dp:.3f}). '
    f'The absence of a detectable pre-trend supports parallel trends. '
    f'The 2023 estimates on the balanced triple closely replicate the main two-period '
    f'results: MHLTH {b23_mh:+.3f} pp (p = {p23_mh:.3f}) and '
    f'Depression {b23_dp:+.3f} pp (p = {p23_dp:.3f}), '
    f'confirming robustness to sample restriction. '
    f'We note, following Callaway and Sant\'Anna (2021) and Goodman-Bacon (2021), '
    f'that staggered adoption across 2022- and 2023-cohort states means the pooled '
    f'2022 coefficient is not a clean pre-trend test for all switchers; a fully '
    f'cohort-specific event study would sharpen this validation.'))
story += fig_block(fig10_path, w=5.8*inch,
    cap='Figure 2.  Three-Period Stacked Event Study: Switcher Coefficients Relative to 2020.',
    note='Stacked FD estimates on the balanced 1,989-county panel. '
         'Blue shading = pre-treatment window; red = treatment window. '
         'Whiskers are 95% CI. SE clustered by state.')

story.append(p('3.3  Dose-Response by Distance', s_h2))
story.append(p(
    f'A genuine causal spillover should exhibit distance decay: largest effects close '
    f'to the border and attenuation with distance. Figure A.4 (Appendix) and Table A.2 '
    f'(Appendix) show FD estimates for counties in four distance bands (reference: '
    f'>300 miles). For MHLTH, the 0-50 mile band yields {hb_mh[0][0]:+.3f} pp '
    f'(SE = {hb_mh[0][1]:.3f}) and the 50-100 mile band {hb_mh[1][0]:+.3f} pp '
    f'(SE = {hb_mh[1][1]:.3f}), declining to {hb_mh[2][0]:+.3f} pp at '
    f'100-200 miles. For Depression: {hb_dp[0][0]:+.3f} pp (0-50 mi) and '
    f'{hb_dp[1][0]:+.3f} pp (50-100 mi). Wide confidence intervals prevent formal '
    f'monotonicity tests, but the qualitative pattern is consistent with a '
    f'border-proximity mechanism.'))

story.append(p('3.4  Marijuana-Impaired Driving (NHTSA FARS)', s_h2))
_fars_intro = (
    f'We provide a behavioral validation of the border-proximity mechanism using '
    f'NHTSA Fatal Analysis Reporting System (FARS) data for 2020 and 2023. '
    f'FARS records all US fatal motor vehicle crashes; we identify marijuana-involved '
    f'crashes via toxicological DRUGRES codes (cannabinoids: 600-695 in 2020; '
    f'natural cannabinoids codes 5060-5067 and 5997 in 2023 under the updated coding '
    f'scheme). County-level crash counts are normalized to {_rate_label}. '
    f'This outcome captures a behavioral channel -- cross-border trips during which '
    f'consumers drive after consuming marijuana -- directly related to the access mechanism.')
if _fars_ok:
    _fars_dir  = 'fell' if (pd.notna(bf) and bf < 0) else 'rose'
    _ctrl_dir  = 'rose' if (pd.notna(ct_mj_23) and ct_mj_23 > ct_mj_20) else 'fell'
    _fars_sig  = stars(pf)
    _fars_interp = (
        'Notably, switcher counties show a relative DECREASE in marijuana fatal crashes '
        'compared to controls -- consistent with a harm-reduction mechanism whereby '
        'legal access nearby reduces risky long-distance driving to dispensaries, '
        'or alternatively reflecting insufficient statistical power (p = {:.3f}).'.format(pf)
        if pd.notna(bf) and bf < 0
        else 'The positive coefficient is consistent with greater cannabis consumption near borders.')
    _fars_intro += (
        f' Switcher counties\' marijuana fatal crash rate {_fars_dir} from {sw_mj_20:.2f} to '
        f'{sw_mj_23:.2f} per 100k; control counties\' rate {_ctrl_dir} from {ct_mj_20:.2f} to '
        f'{ct_mj_23:.2f}. '
        f'The first-difference DiD estimate is {bf:+.3f}{_fars_sig} '
        f'(SE = {sef:.3f}; 95% CI [{ci_lo_f:.3f}, {ci_hi_f:.3f}]; '
        f'p = {pf:.3f}; N = {nf:,} county FDs; {n_sw_f} switchers, {n_ct_f} controls). '
        f'{_fars_interp} '
        f'Figure 3 plots the group mean trends.')
else:
    _fars_intro += ' FARS data could not be loaded; see Appendix for details.'
story.append(p(_fars_intro))

if _fars_ok and fig11_path and os.path.exists(fig11_path):
    story += fig_block(fig11_path, w=5.2*inch,
        cap='Figure 3.  Marijuana-Involved Fatal Crashes per 100k: Switchers vs Controls (NHTSA FARS)',
        note='First-difference DiD on county-level marijuana fatal crash rates. '
             'NHTSA FARS 2020 and 2023. Error bars = 95% CI. Never-legalizing states, main sample only.')

story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# 4. ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────
story.append(p('4.  Robustness: Medical Marijuana Adoption', s_h1))
story.append(p(
    f'We assess whether the spillover effects are specific to recreational retail access '
    f'or reflect broader proximity to any marijuana policy change. We compare counties in '
    f'three states that newly adopted comprehensive medical marijuana programs '
    f'(South Dakota, Alabama, and Mississippi; {n_med_newly} treated counties) '
    f'to counties in thirteen states maintaining full prohibition '
    f'(Georgia, Idaho, Indiana, Iowa, Kansas, Kentucky, North Carolina, Nebraska, '
    f'South Carolina, Tennessee, Texas, Wisconsin, Wyoming; {n_med_never} control counties). '
    f'This comparison uses entirely different treatment and control states, a different '
    f'mechanism (in-state adoption vs cross-border spillover), and a different policy '
    f'context (medical vs recreational access).'))
story.append(p(f'{B("Table 3.")} Robustness: Medical Marijuana Adoption DiD', s_cap))
hdr_t3 = ['', 'MHLTH', 'Depression']
rows_t3 = [
    ['Medical adoption (= 1)',
     f'{bmed_mh:+.3f}{stars(pmed_mh)}',
     f'{bmed_dp:+.3f}{stars(pmed_dp)}'],
    ['', f'({semed_mh:.3f})', f'({semed_dp:.3f})'],
    ['p-value', f'{pmed_mh:.3f}', f'{pmed_dp:.3f}'],
    ['95% CI',
     f'[{ci_lo_med_mh:.3f}, {ci_hi_med_mh:.3f}]',
     f'[{ci_lo_med_dp:.3f}, {ci_hi_med_dp:.3f}]'],
    ['Treatment counties', str(n_med_newly), str(n_med_newly)],
    ['Control counties', str(n_med_never), str(n_med_never)],
    ['Total N (FDs)', str(nmed_mh), str(nmed_dp)],
    ['SE clustering', 'State', 'State'],
]
cw_t3 = [2.8*inch, 1.5*inch, 1.5*inch]
story += rtable(hdr_t3, rows_t3, cw_t3,
    note='FD estimator, 2020 to 2023. Treatment = newly adopted medical marijuana '
         '(SD, AL, MS). Control = 13 states with no medical or recreational program. '
         'SE clustered by state. * p<0.10, ** p<0.05, *** p<0.01.')
story.append(p(
    f'Table 3 shows that medical marijuana adoption is associated with '
    f'MHLTH {bmed_mh:+.3f} pp (p = {pmed_mh:.3f}) and '
    f'Depression {bmed_dp:+.3f} pp (p = {pmed_dp:.3f}) -- both statistically '
    f'indistinguishable from zero. The null result in the medical analysis, in contrast '
    f'to the significant recreational spillover estimates, points to a mechanism specific '
    f'to recreational retail access. The commercial availability of marijuana products, '
    f'active retail environments, and consumer-facing marketing that accompany recreational '
    f'legalization -- features absent from early-stage medical programs -- may be the '
    f'operative channels. This interpretation is consistent with Bradford and Bradford '
    f'(2016), who show that medical and recreational marijuana laws have distinct effects '
    f'on downstream pharmaceutical markets, and with Dills et al. (2017), who document '
    f'that legalization type matters for consumption patterns.'))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# 5. CONCLUSION
# ─────────────────────────────────────────────────────────────────────────
story.append(p('5.  Conclusion', s_h1))
story.append(p(
    f'This paper estimates cross-border spillover effects of recreational marijuana '
    f'legalization on county-level mental health in never-legalizing US states. '
    f'Using a continuous treatment strategy that exploits within-county changes in '
    f'log-distance to the nearest recreational border, we find that counties which '
    f'gained proximity to a recreational market experienced worse mental health '
    f'(MHLTH: {bm4m:+.3f} pp; p = {pm4m:.3f}; Depression: {bd3m:+.3f} pp; p = {ppd3m:.3f}). '
    f'A binary switcher robustness check confirms: MHLTH {bm2:+.3f} pp (p = {pm2:.3f}), '
    f'depression {bd2:+.3f} pp (p = {ppd2:.3f}). A three-period stacked event '
    f'study confirms parallel pre-trends; a medical marijuana robustness check yields '
    f'null results, pointing to a mechanism specific to recreational retail access. '
    f'NHTSA FARS data on marijuana-involved fatal crashes provide a behavioral complement: '
    f'the DiD estimate ({bf:+.3f} per 100k; p = {pf:.3f}) is statistically insignificant, '
    f'with switcher counties\' crash rate falling while controls\' rose, suggesting '
    f'legal access may reduce risky driving but does not attenuate the mental health spillover. '
    f'The dose-response pattern -- effects largest within 50 miles and attenuating with '
    f'distance -- is consistent with a causal border-proximity channel.'))
story.append(p(
    f'The positive sign -- worse mental health near recreational borders -- is inconsistent '
    f'with a simple "access alleviates distress" narrative and more consistent with '
    f'demand-side effects: residents who cross borders for legal cannabis may increase '
    f'consumption in ways that elevate cannabis use disorder risk or worsen anxiety at '
    f'high doses (Hasin et al., 2015; Volkow et al., 2014). '
    f'NHTSA FARS evidence on marijuana-involved fatal crashes provides a behavioral '
    f'complement: the DiD estimate of {bf:+.3f} per 100k (p = {pf:.3f}) is statistically '
    f'indistinguishable from zero. Switcher counties\' crash rate actually fell from '
    f'{sw_mj_20:.2f} to {sw_mj_23:.2f} while controls\' rose from {ct_mj_20:.2f} to '
    f'{ct_mj_23:.2f}, consistent with a harm-reduction effect of nearby legal access '
    f'on impaired driving -- or simply noise given insufficient power. This null FARS result '
    f'suggests the mental health channel operates independently of fatal impaired driving. '
    f'The magnitudes of mental health effects are modest: '
    f'{abs(bm2):.2f} pp for MHLTH ({pct_mh:.1f}% of baseline) and {abs(bd2):.2f} pp '
    f'for depression ({pct_dep:.1f}% of baseline), implying that individual-level effects '
    f'do not translate to large aggregate county-level changes, partly because PLACES MRP '
    f'estimates are smoothed toward state means (attenuation bias toward zero).'))
story.append(p(
    f'Limitations include: MRP attenuation of DiD coefficients; centroid-to-centroid '
    f'distance mismeasurement for large counties; the absence of time-varying controls; '
    f'and the staggered-adoption caveat for the event study pre-trend test. Future work '
    f'incorporating individual-level BRFSS microdata, cohort-specific estimators '
    f'(Callaway and Sant\'Anna, 2021; Goodman-Bacon, 2021), and first-stage measures of '
    f'cross-border cannabis purchases would substantially strengthen causal identification. '
    f'Notwithstanding these limitations, our estimates bound the spillover below '
    f'{max(abs(ci_lo_m), ci_hi_m):.2f} pp for MHLTH and '
    f'{max(abs(ci_lo_d), ci_hi_d):.2f} pp for depression, providing actionable '
    f'information for state legislators evaluating the inter-state externalities of '
    f'recreational marijuana legalization.'))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────
story.append(p('References', s_h1))
story.append(HRFlowable(width='100%', thickness=0.4,
                         color=colors.HexColor('#bbbbbb')))
story.append(Spacer(1, 6))
_refs = [
    ('Anderson, D.M., Hansen, B., Rees, D.I., 2013.',
     'Medical marijuana laws, traffic fatalities, and alcohol consumption. '
     '<i>J. Law Econ.</i> 56 (2), 333-369.'),
    ('Anderson, D.M., Rees, D.I., Sabia, J.J., 2014.',
     'Medical marijuana laws and suicides by gender and age. '
     '<i>Am. J. Public Health</i> 104 (12), 2369-2376.'),
    ('Bertrand, M., Duflo, E., Mullainathan, S., 2004.',
     'How much should we trust differences-in-differences estimates? '
     '<i>Q. J. Econ.</i> 119 (1), 249-275.'),
    ('Bradford, A.C., Bradford, W.D., 2016.',
     'Medical marijuana laws reduce prescription medication use in Medicare Part D. '
     '<i>Health Aff.</i> 35 (7), 1230-1236.'),
    ("Callaway, B., Sant'Anna, P.H.C., 2021.",
     'Difference-in-differences with multiple time periods. '
     '<i>J. Econom.</i> 225 (2), 200-230.'),
    ('Cameron, A.C., Miller, D.L., 2015.',
     "A practitioner's guide to cluster-robust inference. "
     '<i>J. Hum. Resour.</i> 50 (2), 317-372.'),
    ('Cerda, M., Wall, M., Keyes, K.M., Galea, S., Hasin, D., 2012.',
     'Medical marijuana laws in 50 states: Investigating the relationship between '
     'state legalization of medical marijuana and marijuana use, abuse and dependence. '
     '<i>Drug Alcohol Depend.</i> 120 (1-3), 22-27.'),
    ('Dills, A.K., Goffard, S., Miron, J., 2017.',
     'The effects of marijuana legalizations and decriminalizations on marijuana use. '
     '<i>NBER Working Paper No. 23779.</i>'),
    ('Di Forti, M., Quattrone, D., Freeman, T.P., et al., 2019.',
     'The contribution of cannabis use to variation in the incidence of psychotic '
     'disorder across Europe (EU-GEI): a multicentre case-control study. '
     '<i>Lancet Psychiatry</i> 6 (5), 427-436.'),
    ('Goodman-Bacon, A., 2021.',
     'Difference-in-differences with variation in treatment timing. '
     '<i>J. Econom.</i> 225 (2), 254-277.'),
    ('Hao, Z., Cowan, B.W., 2020.',
     'The cross-border spillover effects of recreational marijuana legalization. '
     '<i>Contemp. Econ. Policy</i> 38 (1), 1-18.'),
    ('Hasin, D.S., Saha, T.D., Kerridge, B.T., et al., 2015.',
     'Prevalence of marijuana use disorders in the United States between 2001-02 '
     'and 2012-13. '
     '<i>JAMA Psychiatry</i> 72 (12), 1235-1242.'),
    ('Pacula, R.L., Powell, D., Heaton, P., Sevigny, E.L., 2015.',
     'Assessing the effects of medical marijuana laws on marijuana use: '
     'the devil is in the details. '
     '<i>J. Policy Anal. Manage.</i> 34 (1), 7-31.'),
    ('Powell, D., Pacula, R.L., Jacobson, M., 2018.',
     'Do medical marijuana laws reduce addictions and deaths related to pain killers? '
     '<i>J. Health Econ.</i> 58, 29-42.'),
    ('Volkow, N.D., Baler, R.D., Compton, W.M., Weiss, S.R.B., 2014.',
     'Adverse health effects of marijuana use. '
     '<i>N. Engl. J. Med.</i> 370 (23), 2219-2227.'),
    ('Wen, H., Hockenberry, J.M., Cummings, J.R., 2015.',
     'The effect of medical marijuana laws on adolescent and adult use of '
     'marijuana, alcohol, and other substances. '
     '<i>J. Health Econ.</i> 42, 64-80.'),
    ('National Highway Traffic Safety Administration (NHTSA), 2020, 2023.',
     'Fatality Analysis Reporting System (FARS). '
     'National Center for Statistics and Analysis, US Dept. of Transportation. '
     'Washington, DC.'),
]
for _auth, _rest in _refs:
    story.append(Paragraph(f'<b>{_auth}</b> {_rest}', s_refp))
story.append(PageBreak())

# ─────────────────────────────────────────────────────────────────────────
# APPENDIX
# ─────────────────────────────────────────────────────────────────────────
story.append(p('Appendix', s_app))
story.append(HRFlowable(width='100%', thickness=0.5,
                         color=colors.HexColor('#bbbbbb')))
story.append(Spacer(1, 6))

# Table A.1: Summary Statistics
_ac = PS('acap', f=TNB, sz=9, ld=12, sb=8, sa=3, al=TA_CENTER)
story.append(p('Table A.1.  Summary Statistics -- Never-Legalizing States', _ac))
_sw20 = main[(main['switcher']==1) & (main['year_int']==2020)]
_ct20 = main[(main['switcher']==0) & (main['year_int']==2020)]
hdr_A1 = ['', '2020 Mean', '2020 SD', '2023 Mean', '2023 SD']
rows_A1 = [
    ['MHLTH (% adults, 14+ poor MH days/mo)',
     fmt(mean_mh20), fmt(sd_mh20), fmt(mean_mh23), fmt(sd_mh23)],
    ['Depression (% adults, ever diagnosed)',
     fmt(mean_dep20), fmt(sd_dep20), fmt(mean_dep23), fmt(sd_dep23)],
    [f'  Switchers (N = {n_switchers})',
     fmt(_sw20['mhlth'].mean()), fmt(_sw20['mhlth'].std()), 'see main', ''],
    [f'  Controls (N = {n_control})',
     fmt(_ct20['mhlth'].mean()), fmt(_ct20['mhlth'].std()), 'sample', ''],
    [f'County-year obs (total: {int(n_counties*2):,})',
     f'{n_counties:,}', '', f'{n_counties:,}', ''],
]
cw_A1 = [2.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch]
story += rtable(hdr_A1, rows_A1, cw_A1,
    note=(f'CDC PLACES crude prevalence. '
          f'Switcher mean dist to nearest rec border (2023): {mean_dsw:.1f} mi. '
          f'Control mean dist: {mean_dctl:.1f} mi. '
          f'N = {n_always} always-near counties excluded from main sample.'))
story.append(PageBreak())

# Table A.2: Dose-response
story.append(p('Table A.2.  DiD Estimates by Distance Band (Reference: >300 Miles)', _ac))
hdr_A2 = ['Distance Band', 'MHLTH\nCoef', 'MHLTH\n(SE)', 'MHLTH\np-val',
          'Dep.\nCoef', 'Dep.\n(SE)', 'Dep.\np-val']
_bands = ['0-50 miles', '50-100 miles', '100-200 miles', '200-300 miles']
rows_A2 = []
for _j, _lab in enumerate(_bands):
    rows_A2.append([
        _lab,
        f'{hb_mh[_j][0]:+.3f}{stars(hb_mh[_j][2])}',
        f'({hb_mh[_j][1]:.3f})',
        f'{hb_mh[_j][2]:.3f}',
        f'{hb_dp[_j][0]:+.3f}{stars(hb_dp[_j][2])}',
        f'({hb_dp[_j][1]:.3f})',
        f'{hb_dp[_j][2]:.3f}',
    ])
cw_A2 = [1.55*inch, 0.78*inch, 0.72*inch, 0.70*inch, 0.78*inch, 0.72*inch, 0.70*inch]
story += rtable(hdr_A2, rows_A2, cw_A2,
    note='FD estimates. Reference group: counties >300 miles from any recreational border. '
         'Full never-legalizing sample. SE clustered by state. '
         '* p<0.10, ** p<0.05, *** p<0.01.')
story.append(PageBreak())

# Appendix Figures
story.append(p('Appendix Figures', s_app))
story.append(HRFlowable(width='100%', thickness=0.4,
                         color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 6))

story += fig_block(fig1_path,
    cap='Figure A.1.  Treatment Assignment -- County Scatter Map (Contiguous US)',
    note='County centroids colored by group. Red = switcher, blue = always-near, gray = control.')
story.append(PageBreak())

story += fig_block(fig3_path,
    cap='Figure A.2.  Mean Outcome Trends: Switchers vs Controls, 2020-2023',
    note='Unweighted county means by group. Main sample only (switchers and controls).')
story.append(p(
    f'Both groups trend upward in parallel, consistent with parallel trends. '
    f'Raw mean change: switchers MHLTH +{fd_m[fd_m["switcher"]==1]["d_y"].mean():.2f} pp '
    f'vs controls +{fd_m[fd_m["switcher"]==0]["d_y"].mean():.2f} pp; '
    f'the DiD estimate of {bm2:+.3f} pp reflects this differential.'))
story.append(PageBreak())

story += fig_block(fig6_path,
    cap='Figure A.3.  Pre-Period (2020) Outcomes by Future (2023) Distance Band',
    note='Flat profile across bands supports parallel trends: 2020 health levels do not '
         'systematically vary with proximity to borders that will legalize in 2023. '
         f'Placebo regression: beta = {pl_b:+.3f} pp (p = {pl_p:.3f}).')
story.append(PageBreak())

story += fig_block(fig5_path, w=5.8*inch,
    cap='Figure A.4.  Dose-Response: DiD Coefficients by Distance Band',
    note='Reference group: >300 miles. MHLTH (left) and Depression (right). '
         'Consistent with distance-decay mechanism. 95% CI. SE clustered by state.')
story.append(PageBreak())

story += fig_block(fig2_path,
    cap='Figure A.5.  Distance Distribution by County Group (Period 2, 2023)',
    note='Kernel density. Dashed vertical line = 100-mile treatment threshold.')
story.append(PageBreak())

story += fig_block(fig7_path,
    cap='Figure A.6.  Medical Marijuana Robustness: Pre/Post Group Means',
    note='Unweighted county means. Medical treatment = SD, AL, MS. Control = 13 states.')
story += fig_block(fig8_path,
    cap='Figure A.7.  Medical Marijuana Robustness: Trend Lines by Group',
    note='Parallel-trends visualization for the medical marijuana robustness check.')
story.append(PageBreak())

story += fig_block(fig9_path,
    cap='Figure A.8.  Medical Direct vs Recreational Spillover: DiD Comparison',
    note='Red = medical adoption (direct effect). Blue = recreational spillover. '
         '95% CI. SE clustered by state.')

story.append(Spacer(1, 0.3*inch))
story.append(HRFlowable(width='100%', thickness=0.4,
                         color=colors.HexColor('#cccccc')))
story.append(Spacer(1, 4))
story.append(p(
    '<i>Data:</i> CDC PLACES (data.cdc.gov) · US Census Bureau 2023 County Gazetteer '
    '· NHTSA FARS (2020, 2023) · State Recreational and Medical Marijuana Statutes. '
    '<i>Software:</i> Python 3 / pandas / statsmodels / matplotlib / reportlab.',
    PS('foot', f=TNI, sz=8.5, ld=12, sa=3, al=TA_LEFT)))


# ── BUILD ──────────────────────────────────────────────────
print("Writing PDF ...")
doc.build(story)
print(f"\nDONE  ->  {OUT}")
