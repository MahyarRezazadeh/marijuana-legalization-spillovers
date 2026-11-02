# ============================================================
# Dose-response and marginal-effect figures for the continuous
# distance specification. Linear ME is a flat line by
# construction; quadratic ME is shown alongside it but the
# quadratic term is NOT statistically significant (p=0.099,
# joint F-test p=0.11), so it's presented as suggestive, not
# a replacement specification. Kieran Healy style.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import statsmodels.api as sm
    HAVE_LOWESS = True
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    HAVE_LOWESS = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'
MUTED = '#8a8a8a'
ACCENT = '#c1502e'
ACCENT2 = '#4c9a6a'
GRAY = '#8a8a8a'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=9.5, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

panel = pd.read_csv(os.path.join(DATA, 'doseresponse_panel.csv'))
with open(os.path.join(DATA, 'doseresponse_coefs.json')) as f:
    c = json.load(f)

x = panel['dist_change_mi'].to_numpy()
y = panel['d_od'].to_numpy()
xr = np.linspace(x.min(), x.max(), 200)

# ============================================================
# Fig A: dose-response -- binned scatter + LOWESS + linear + quadratic
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.scatter(x, y, s=10, alpha=0.18, color=GRAY, linewidths=0, label='_nolegend_')

# 20-mile bin averages
bins = pd.cut(x, bins=np.arange(-520, 21, 20))
binned = pd.DataFrame({'x': x, 'y': y, 'bin': bins}).groupby('bin', observed=True).agg(
    xm=('x', 'mean'), ym=('y', 'mean'), n=('y', 'size')).query('n >= 5')
ax.scatter(binned['xm'], binned['ym'], s=28, color=INK, alpha=0.8, zorder=4, label='20-mile bin means')

# Use a simple OLS refit here just to get the intercept for plotting (coeffs already validated in analyze_doseresponse.py)
X1 = sm.add_constant(x)
b0, b1 = sm.OLS(y, X1).fit().params
ax.plot(xr, b0 + b1*xr, color=ACCENT, lw=2.2, label=f'Linear fit (β={c["lin_b1"]:.4f}/mi, p=0.103)')

X2 = sm.add_constant(np.column_stack([x, x**2]))
b0q, b1q, b2q = sm.OLS(y, X2).fit().params
ax.plot(xr, b0q + b1q*xr + b2q*xr**2, color=ACCENT2, lw=2.2, linestyle='--',
        label='Quadratic fit (curvature n.s., p=0.099)')

if HAVE_LOWESS:
    sm_lowess = lowess(y, x, frac=0.5, return_sorted=True)
    ax.plot(sm_lowess[:, 0], sm_lowess[:, 1], color=INK, lw=1.4, linestyle=':', label='LOWESS (nonparametric)')

ax.axvline(0, color='#ddd', lw=0.8)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')
healy(ax, title='Figure A. Dose-Response: Overdose-Rate Change vs. Distance Change',
      xlabel='Change in distance to nearest legal border (miles; negative = became closer)',
      ylabel='Change in overdose rate per 100,000')
plt.tight_layout(rect=[0, 0.1, 1, 1])
caption_a = ('Gray points are counties; black points are 20-mile bin means (bins with n>=5). The quadratic\n'
             'curvature term is not statistically significant (p=0.099 individually, F-test p=0.11 jointly) --\n'
             'shown for transparency, not as a preferred specification.')
fig.text(0.02, 0.015, caption_a, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figA_dose_response.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figA_dose_response.png")

# ============================================================
# Fig B: marginal effect curve -- flat (linear) vs. varying (quadratic)
# ============================================================
me_lin = np.full_like(xr, c['lin_b1'])
me_lin_se = np.full_like(xr, c['lin_se1'])

me_quad = c['quad_b1'] + 2 * c['quad_b2'] * xr
var_quad = c['quad_se1']**2 + (2*xr)**2 * c['quad_se2']**2 + 2*(2*xr)*c['quad_cov12']
me_quad_se = np.sqrt(np.maximum(var_quad, 0))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
ax = axes[0]
ax.plot(xr, me_lin, color=ACCENT, lw=2.2)
ax.fill_between(xr, me_lin - 1.96*me_lin_se, me_lin + 1.96*me_lin_se, color=ACCENT, alpha=0.15, linewidth=0)
ax.axhline(0, color='#bbb', lw=1, linestyle='--')
healy(ax, title='(a) Linear specification: ME is constant by construction',
      xlabel='Change in distance to border (miles)', ylabel='Marginal effect on overdose rate / 100k / mile')

ax = axes[1]
ax.plot(xr, me_quad, color=ACCENT2, lw=2.2)
ax.fill_between(xr, me_quad - 1.96*me_quad_se, me_quad + 1.96*me_quad_se, color=ACCENT2, alpha=0.15, linewidth=0)
ax.axhline(0, color='#bbb', lw=1, linestyle='--')
healy(ax, title='(b) Quadratic specification: ME varies with distance (curvature n.s.)',
      xlabel='Change in distance to border (miles)')

plt.tight_layout(rect=[0, 0.16, 1, 0.92])
fig.suptitle('Figure B. Marginal Effect of a One-Mile Change in Distance, by Specification',
             fontsize=10.5, x=0.02, ha='left', y=0.99, color=INK)
caption_b = ('Shaded bands are 95% CIs (delta method for the quadratic case). Panel (a): the linear model imposes\n'
             'a constant marginal effect everywhere, so this is a flat line at the estimated slope, not a claim\n'
             'about the data. Panel (b) shows how the marginal effect would vary if the (statistically unconfirmed)\n'
             'quadratic curvature were taken at face value -- its CI comfortably includes the flat line in panel (a).')
fig.text(0.02, 0.02, caption_b, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figB_marginal_effect.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figB_marginal_effect.png")
