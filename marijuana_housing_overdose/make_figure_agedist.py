# ============================================================
# National age distribution of overdose deaths (descriptive
# context only -- no county/state geography exists publicly to
# use this in the regression design itself). Kieran Healy style.
# ============================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

INK = '#3f3f3f'
MUTED = '#8a8a8a'
ACCENT = '#c1502e'

def healy(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, colors=INK)
    if title:  ax.set_title(title, fontsize=10, loc='left', pad=6, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)

age_dist = pd.read_csv(os.path.join(DATA, 'national_age_distribution_overdose_2024.csv'))
with open(os.path.join(DATA, 'avg_age_at_death.json')) as f:
    age_info = json.load(f)
avg_age = age_info['avg_age_at_death']

order = ['Younger than 15 years', '15-24 years', '25-34 years', '35-44 years', '45-54 years',
         '55-64 years', '65-74 years', '75-84 years', '85 years and older']
age_dist['subgroup'] = pd.Categorical(age_dist['subgroup'], categories=order, ordered=True)
age_dist = age_dist.sort_values('subgroup')
labels = ['<15', '15-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
ax.bar(labels, age_dist['rate'], color=ACCENT, alpha=0.85, width=0.65)
healy(ax, title='(a) Overdose death rate by age group, 2024 (national)',
      xlabel='Age group', ylabel='Deaths per 100,000')

ax = axes[1]
ax.bar(labels, age_dist['share_of_deaths']*100, color=MUTED, alpha=0.85, width=0.65)
ax.axvline(np.interp(avg_age, [7.5,19.5,29.5,39.5,49.5,59.5,69.5,79.5,90], range(9)),
           color=ACCENT, lw=1.6, linestyle='--')
ax.annotate(f'avg. age at death\n= {avg_age:.1f} yrs', xy=(3.55, ax.get_ylim()[1]*0.85),
            fontsize=8, color=ACCENT, ha='left')
healy(ax, title='(b) Share of all overdose deaths by age group',
      xlabel='Age group', ylabel='Share of deaths (%)')

fig.suptitle('Figure C. National Age Distribution of Drug-Overdose Deaths, 2024',
             fontsize=11, x=0.02, ha='left', y=0.99, color=INK)
caption = ('CDC DQS national data, "All drug overdose deaths." No county- or state-level age breakdown is publicly\n'
           'available (CDC WONDER, which would provide one, is not reachable from this environment) -- shown here\n'
           'as descriptive context only, not usable in the county-level regression design.')
plt.tight_layout(rect=[0, 0.12, 1, 0.92])
fig.text(0.02, 0.02, caption, fontsize=7, color=MUTED, style='italic')
fig.savefig(os.path.join(FIG, 'figC_age_distribution.png'), dpi=170, facecolor='white')
plt.close()
print("Saved figC_age_distribution.png")
