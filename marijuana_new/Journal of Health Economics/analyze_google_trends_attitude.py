# ============================================================
# Attitude-proxy check, analysis stage. IMPORTANT CORRECTION
# (2026-08-06): a raw version of this check (indexing each state's
# series to its own pre-switch mean only, mirroring
# analyze_google_trends.py's naloxone check exactly) showed a sharp
# post-switch decline -- but a direct national-level pull of the
# same term ("legalize marijuana", geo=US, 2019-2025) revealed a
# steady ~15%/year secular decline in national search volume for
# this exact phrase over the whole period, unrelated to any single
# state. Since post-periods mechanically fall in later calendar
# years than pre-periods, the raw pooled index was very likely just
# recreating that national trend, not a real state-specific effect.
# This version DETRENDS first: divides each state-year's raw search
# value by the national US value for that same year, THEN indexes
# each state's detrended series to its own pre-switch mean = 1.0,
# before pooling to event-time. This isolates state-specific
# deviations from the common national trend.
# ============================================================
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')

trends = pd.read_csv(os.path.join(DATA, 'google_trends_attitude_raw.csv'))
national = pd.read_csv(os.path.join(DATA, 'google_trends_attitude_national.csv'))
national.columns = ['year', 'national_interest']
trends = trends.merge(national, on='year', how='left')
trends['detrended'] = trends['search_interest'] / trends['national_interest']

trends['event_time'] = trends['year'] - trends['ref_year']
trends['event_time'] = trends['event_time'].clip(-3, 3)

# --- Raw (not detrended) version, for comparison ---
pre_raw = trends[trends['event_time'] < 0].groupby('state_abbr')['search_interest'].mean().rename('pre_mean_raw')
trends = trends.merge(pre_raw, on='state_abbr', how='left')
trends = trends[trends['pre_mean_raw'] > 0].copy()
trends['index_raw'] = trends['search_interest'] / trends['pre_mean_raw']

summary_raw = trends.groupby('event_time')['index_raw'].agg(['mean', 'std', 'count']).reset_index()
summary_raw['se'] = summary_raw['std'] / np.sqrt(summary_raw['count'])
print(f"=== RAW (not detrended) index by event-time (pooled across {trends['state_abbr'].nunique()} states) ===")
print(summary_raw.to_string(index=False))

# --- Detrended version (preferred) ---
pre_dt = trends[trends['event_time'] < 0].groupby('state_abbr')['detrended'].mean().rename('pre_mean_dt')
trends = trends.merge(pre_dt, on='state_abbr', how='left')
trends = trends[trends['pre_mean_dt'] > 0].copy()
trends['index'] = trends['detrended'] / trends['pre_mean_dt']

summary = trends.groupby('event_time')['index'].agg(['mean', 'std', 'count']).reset_index()
summary['se'] = summary['std'] / np.sqrt(summary['count'])
print(f"\n=== DETRENDED (net of national trend) index by event-time (pooled across {trends['state_abbr'].nunique()} states, pre-switch mean = 1.0) ===")
print(summary.to_string(index=False))

trends.to_csv(os.path.join(DATA, 'google_trends_attitude_panel.csv'), index=False)
summary.to_csv(os.path.join(DATA, 'google_trends_attitude_summary.csv'), index=False)
summary_raw.to_csv(os.path.join(DATA, 'google_trends_attitude_summary_raw.csv'), index=False)
print("\nSaved google_trends_attitude_panel.csv, google_trends_attitude_summary.csv (detrended, preferred), google_trends_attitude_summary_raw.csv (raw, for comparison)")
