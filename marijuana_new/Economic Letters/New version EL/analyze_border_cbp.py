# ============================================================
# B1 sector-specific results: same six-spec table as
# analyze_border_bizform.py, run once per ancillary NAICS sector.
# Reports both the raw full sample and a cleaned sample (drops WY/DE
# and counties <5,000 population, per the artifact found in the
# aggregate BFS check) so the two can be compared directly.
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

s = pd.read_csv(os.path.join(DATA, 'stacked_panel_border_cbp.csv'), dtype={'fips5': str})

DIVISION = {
    'NH': 'NewEngland',
    'PA': 'MidAtlantic',
    'IN': 'ENCentral', 'OH': 'ENCentral', 'WI': 'ENCentral',
    'IA': 'WNCentral', 'KS': 'WNCentral', 'MN': 'WNCentral', 'NE': 'WNCentral', 'ND': 'WNCentral', 'SD': 'WNCentral',
    'DE': 'SAtlantic', 'FL': 'SAtlantic', 'GA': 'SAtlantic', 'NC': 'SAtlantic', 'SC': 'SAtlantic', 'VA': 'SAtlantic', 'DC': 'SAtlantic', 'WV': 'SAtlantic',
    'AL': 'ESCentral', 'KY': 'ESCentral', 'MS': 'ESCentral', 'TN': 'ESCentral',
    'AR': 'WSCentral', 'LA': 'WSCentral', 'OK': 'WSCentral', 'TX': 'WSCentral',
    'ID': 'Mountain', 'UT': 'Mountain', 'WY': 'Mountain',
    'HI': 'Pacific',
}
s['division'] = s['state_abbr'].map(DIVISION)

specs = {
    'No controls': '{y} ~ {x} + C(transition)',
    '+ baseline rate': '{y} ~ {x} + {base} + C(transition)',
    '+ state FE': '{y} ~ {x} + {base} + C(transition) + C(state_abbr)',
    '+ division x transition FE': '{y} ~ {x} + {base} + C(transition):C(division)',
    '+ state x transition FE': '{y} ~ {x} + {base} + C(transition):C(state_abbr)',
    '+ county FE': '{y} ~ {x} + {base} + C(transition) + C(fips5)',
}

sectors = ['security_systems', 'hvac_contractors', 'commercial_printing', 'mgmt_sci_tech_consulting', 'testing_labs']

def run(sub, sector, tag):
    y = f'd_{sector}'
    base = f'baseline_{sector}_rate'
    sub = sub[sub[y].notna() & sub[base].notna() & sub['division'].notna() & (sub['population'] > 0)].copy()
    sub['g'] = pd.Categorical(sub['state_abbr']).codes
    sub['g_county'] = pd.Categorical(sub['fips5']).codes
    print(f"\n=== {sector} [{tag}] === n={len(sub)}, switch events={int(sub['switcher'].sum())}")
    print(sub.groupby('switcher')[y].agg(['mean', 'median', 'std', 'count']))
    out_rows = []
    for spec_name, tmpl in specs.items():
        formula = tmpl.format(y=y, x='switcher', base=base)
        try:
            m = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g']})
            mc = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['g_county']})
            print(f"  {spec_name:28s} b={m.params['switcher']:+.4f} se={m.bse['switcher']:.4f} "
                  f"p={m.pvalues['switcher']:.4f} | se_county={mc.bse['switcher']:.4f} p_county={mc.pvalues['switcher']:.4f} n={int(m.nobs)}")
            out_rows.append(dict(sector=sector, tag=tag, spec=spec_name, coef=m.params['switcher'], se=m.bse['switcher'],
                                  p=m.pvalues['switcher'], se_county=mc.bse['switcher'], p_county=mc.pvalues['switcher'], n=int(m.nobs)))
        except Exception as e:
            print(f"  {spec_name:28s} FAILED: {e}")
    return out_rows

all_results = []
for sector in sectors:
    all_results += run(s, sector, 'raw')
    clean = s[(~s['state_abbr'].isin({'WY', 'DE'})) & (s['population'] >= 5000)]
    all_results += run(clean, sector, 'cleaned')

res = pd.DataFrame(all_results)
res.to_csv(os.path.join(DATA, 'border_cbp_results.csv'), index=False)
print("\nSaved border_cbp_results.csv")
