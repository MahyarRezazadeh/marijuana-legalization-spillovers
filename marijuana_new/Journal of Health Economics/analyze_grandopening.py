# ============================================================
# Mechanism check #2 (grand-opening / promotional-pricing effect):
# tests whether cannabis excise tax revenue in the LEGAL-MARKET
# states (the 24-state list whose retail dates drive switcher
# timing elsewhere in this project) shows a spike-then-plateau
# pattern around each state's own retail-opening quarter -- which
# would support reading the mortality effect's one-year concentration
# as a demand-side "grand opening" artifact (promotional pricing
# temporarily pulling forward purchases/cross-border trips) rather
# than a lasting change in relative prices or availability.
#
# Tested directly at the legal-market source states (not traced
# through to specific downstream switcher counties), since the
# grand-opening mechanism is fundamentally about the market's own
# revenue dynamics, not about which neighboring county benefits.
#
# Source: Census Bureau's experimental Cannabis Excise Sales Tax
# Collections product (data/census_cannabis_tax.xlsm, quarterly,
# state-level, Q3-2021 through Q1-2026).
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

raw = pd.read_excel(os.path.join(DATA, 'census_cannabis_tax.xlsm'), sheet_name='T17 Experimental Data', header=None)

header_row = raw.iloc[4]
quarter_cols = {}
for col in range(1, raw.shape[1]):
    label = header_row[col]
    if isinstance(label, str) and label.startswith(('Q1-', 'Q2-', 'Q3-', 'Q4-')):
        quarter_cols[col] = label

data = raw.iloc[5:].copy()
data = data.rename(columns={0: 'state_name'})
data = data[data['state_name'].notna()].copy()

long_rows = []
for _, r in data.iterrows():
    state_name = str(r['state_name']).strip()
    for col, qlabel in quarter_cols.items():
        val = r[col]
        if pd.isna(val) or str(val).strip() in ('X', 'r/', 'NA', ''):
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        q, y = qlabel.split('-')
        long_rows.append({'state_name': state_name, 'year': int(y), 'quarter': int(q[1]), 'revenue_k': val})

tax = pd.DataFrame(long_rows)
print(f"Parsed {len(tax)} state-quarter observations, {tax['state_name'].nunique()} states, "
      f"{tax['year'].min()}Q{tax[tax.year==tax.year.min()].quarter.min()} to {tax['year'].max()}")

STATE_NAME_TO_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
}
tax['state_abbr'] = tax['state_name'].map(STATE_NAME_TO_ABBR)

legal = pd.read_csv(os.path.join(BASE, '..', 'Economic Letters', 'data', 'legal_dates_new.csv'), parse_dates=['retail_date'])
legal = legal.dropna(subset=['retail_date'])[['state_abbr', 'retail_date']]
legal['retail_q'] = legal['retail_date'].dt.quarter
legal['retail_y'] = legal['retail_date'].dt.year
print(f"\n{len(legal)} legal states with a retail date")

m = tax.merge(legal, on='state_abbr', how='inner')
m['q_index'] = m['year'] * 4 + m['quarter']
m['retail_q_index'] = m['retail_y'] * 4 + m['retail_q']
m['event_quarter'] = m['q_index'] - m['retail_q_index']

m = m[m['event_quarter'].between(-8, 8)].copy()
print(f"Matched to {m['state_abbr'].nunique()} legal states with tax data in the +/-8 quarter window")
print(sorted(m['state_abbr'].unique()))

# index each state's own revenue to its pre-opening (event_quarter<0) mean = 1.0,
# so states with very different revenue scales can be pooled
pre_mean = m[m['event_quarter'] < 0].groupby('state_abbr')['revenue_k'].mean().rename('pre_mean')
m = m.merge(pre_mean, on='state_abbr', how='left')
m = m.dropna(subset=['pre_mean'])
m = m[m['pre_mean'] > 0]
m['revenue_index'] = m['revenue_k'] / m['pre_mean']

summary = m.groupby('event_quarter')['revenue_index'].agg(['mean', 'std', 'count']).reset_index()
summary['se'] = summary['std'] / np.sqrt(summary['count'])
print("\n=== Revenue index by event-quarter (pooled across legal states, pre-opening mean = 1.0) ===")
print(summary.to_string(index=False))

m.to_csv(os.path.join(DATA, 'grandopening_panel.csv'), index=False)
summary.to_csv(os.path.join(DATA, 'grandopening_summary.csv'), index=False)
print("\nSaved grandopening_panel.csv, grandopening_summary.csv")
