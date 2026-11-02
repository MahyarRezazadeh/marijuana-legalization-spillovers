# ============================================================
# Mechanism check #4 (attention/awareness effect), analysis stage:
# indexes each state's "naloxone" search-interest series to its own
# pre-switch (event-time -3 to -1) mean = 1.0, bins to event-time
# -3..+3 relative to each state's own reference switch year, and
# pools (mean + SE across states, descriptive -- not a clustered
# regression, given only 21 states).
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

trends = pd.read_csv(os.path.join(DATA, 'google_trends_naloxone_raw.csv'))
trends['event_time'] = trends['year'] - trends['ref_year']
trends['event_time'] = trends['event_time'].clip(-3, 3)

pre = trends[trends['event_time'] < 0].groupby('state_abbr')['search_interest'].mean().rename('pre_mean')
trends = trends.merge(pre, on='state_abbr', how='left')
trends = trends.dropna(subset=['pre_mean'])
trends = trends[trends['pre_mean'] > 0]
trends['index'] = trends['search_interest'] / trends['pre_mean']

summary = trends.groupby('event_time')['index'].agg(['mean', 'std', 'count']).reset_index()
summary['se'] = summary['std'] / np.sqrt(summary['count'])
print("=== Naloxone search-interest index by event-time (pooled across 21 states, pre-switch mean = 1.0) ===")
print(summary.to_string(index=False))

trends.to_csv(os.path.join(DATA, 'google_trends_panel.csv'), index=False)
summary.to_csv(os.path.join(DATA, 'google_trends_summary.csv'), index=False)
print("\nSaved google_trends_panel.csv, google_trends_summary.csv")
