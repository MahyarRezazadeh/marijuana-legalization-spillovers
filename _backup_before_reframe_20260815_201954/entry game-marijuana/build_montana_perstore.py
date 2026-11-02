# ============================================================
# Montana revenue-per-store: current-snapshot cross-section (same
# limitation/approach as Oregon and the CO placebo per-retailer
# check -- MT's licensed dispensary list has no open-date history,
# only a current city). City->county crosswalk built from Wikipedia's
# "List of municipalities in Montana" (129 incorporated places, the
# authoritative source for city-county assignment) plus a small,
# high-confidence supplement for well-known unincorporated
# communities. Cities not confidently mapped are dropped (reduces n,
# does not risk mislabeling a county's border status).
# ============================================================
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
TABLES = os.path.join(BASE, 'tables')

# Wikipedia "List of municipalities in Montana" (incorporated places)
CITY_COUNTY = {
    'ALBERTON': 'Mineral', 'ANACONDA': 'Deer Lodge', 'BAINVILLE': 'Roosevelt', 'BAKER': 'Fallon',
    'BEARCREEK': 'Carbon', 'BELGRADE': 'Gallatin', 'BELT': 'Cascade', 'BIG SANDY': 'Chouteau',
    'BIG TIMBER': 'Sweet Grass', 'BILLINGS': 'Yellowstone', 'BOULDER': 'Jefferson', 'BOZEMAN': 'Gallatin',
    'BRIDGER': 'Carbon', 'BROADUS': 'Powder River', 'BROADVIEW': 'Yellowstone', 'BROWNING': 'Glacier',
    'BUTTE': 'Silver Bow', 'CASCADE': 'Cascade', 'CHESTER': 'Liberty', 'CHINOOK': 'Blaine',
    'CHOTEAU': 'Teton', 'CIRCLE': 'McCone', 'CLYDE PARK': 'Park', 'COLSTRIP': 'Rosebud',
    'COLUMBIA FALLS': 'Flathead', 'COLUMBUS': 'Stillwater', 'CONRAD': 'Pondera', 'CULBERTSON': 'Roosevelt',
    'CUT BANK': 'Glacier', 'DARBY': 'Ravalli', 'DEER LODGE': 'Powell', 'DENTON': 'Fergus',
    'DILLON': 'Beaverhead', 'DODSON': 'Phillips', 'DRUMMOND': 'Granite', 'DUTTON': 'Teton',
    'EAST HELENA': 'Lewis and Clark', 'EKALAKA': 'Carter', 'ENNIS': 'Madison', 'EUREKA': 'Lincoln',
    'FAIRFIELD': 'Teton', 'FAIRVIEW': 'Richland', 'FLAXVILLE': 'Daniels', 'FORSYTH': 'Rosebud',
    'FORT BENTON': 'Chouteau', 'FORT PECK': 'Valley', 'FROID': 'Roosevelt', 'FROMBERG': 'Carbon',
    'GERALDINE': 'Chouteau', 'GLASGOW': 'Valley', 'GLENDIVE': 'Dawson', 'GRASS RANGE': 'Fergus',
    'GREAT FALLS': 'Cascade', 'HAMILTON': 'Ravalli', 'HARDIN': 'Big Horn', 'HARLEM': 'Blaine',
    'HARLOWTON': 'Wheatland', 'HAVRE': 'Hill', 'HELENA': 'Lewis and Clark', 'HINGHAM': 'Hill',
    'HOBSON': 'Judith Basin', 'HOT SPRINGS': 'Sanders', 'HYSHAM': 'Treasure', 'ISMAY': 'Custer',
    'JOLIET': 'Carbon', 'JORDAN': 'Garfield', 'JUDITH GAP': 'Wheatland', 'KALISPELL': 'Flathead',
    'KEVIN': 'Toole', 'LAUREL': 'Yellowstone', 'LAVINA': 'Golden Valley', 'LEWISTOWN': 'Fergus',
    'LIBBY': 'Lincoln', 'LIMA': 'Beaverhead', 'LIVINGSTON': 'Park', 'LODGE GRASS': 'Big Horn',
    'MALTA': 'Phillips', 'MANHATTAN': 'Gallatin', 'MEDICINE LAKE': 'Sheridan', 'MELSTONE': 'Musselshell',
    'MILES CITY': 'Custer', 'MISSOULA': 'Missoula', 'MOORE': 'Fergus', 'NASHUA': 'Valley',
    'NEIHART': 'Cascade', 'OPHEIM': 'Valley', 'OUTLOOK': 'Sheridan', 'PHILIPSBURG': 'Granite',
    'PINESDALE': 'Ravalli', 'PLAINS': 'Sanders', 'PLENTYWOOD': 'Sheridan', 'PLENTYWOO D': 'Sheridan',
    'PLEVNA': 'Fallon', 'POLSON': 'Lake', 'POPLAR': 'Roosevelt', 'RED LODGE': 'Carbon',
    'REXFORD': 'Lincoln', 'RICHEY': 'Dawson', 'RONAN': 'Lake', 'ROUNDUP': 'Musselshell',
    'RYEGATE': 'Golden Valley', 'SACO': 'Phillips', 'SCOBEY': 'Daniels', 'SHELBY': 'Toole',
    'SHERIDAN': 'Madison', 'SIDNEY': 'Richland', 'ST IGNATIUS': 'Lake', 'SAINT IGNATIUS': 'Lake',
    'STANFORD': 'Judith Basin', 'STEVENSVILLE': 'Ravalli', 'STEVENSVILL E': 'Ravalli',
    'SUNBURST': 'Toole', 'SUPERIOR': 'Mineral', 'TERRY': 'Prairie', 'THOMPSON FALLS': 'Sanders',
    'THREE FORKS': 'Gallatin', 'TOWNSEND': 'Broadwater', 'TROY': 'Lincoln', 'TWIN BRIDGES': 'Madison',
    'VALIER': 'Pondera', 'VIRGINIA CITY': 'Madison', 'WALKERVILLE': 'Silver Bow',
    'WEST YELLOWSTONE': 'Gallatin', 'WESTBY': 'Sheridan', 'WHITE SULPHUR SPRINGS': 'Meagher',
    'WHITEFISH': 'Flathead', 'WHITEHALL': 'Jefferson', 'WIBAUX': 'Wibaux', 'WINIFRED': 'Fergus',
    'WINNETT': 'Petroleum', 'WOLF POINT': 'Roosevelt',
    # high-confidence unincorporated-community supplement
    'BIG SKY': 'Gallatin', 'GALLATIN GTWY': 'Gallatin', 'EMIGRANT': 'Park', 'GARDINER': 'Park',
    'BIGFORK': 'Flathead', 'LAKESIDE': 'Flathead', 'SOMERS': 'Flathead', 'MARION': 'Flathead',
    'OLNEY': 'Flathead', 'HUNGRY HORSE': 'Flathead', 'ROLLINS': 'Flathead', 'ELMO': 'Lake',
    'CHARLO': 'Lake', 'ARLEE': 'Lake', 'ST REGIS': 'Mineral', 'SAINT REGIS': 'Mineral',
    'SALTESE': 'Mineral', 'CLINTON': 'Missoula', 'SEELEY LAKE': 'Missoula', 'LOLO': 'Missoula',
    'FLORENCE': 'Ravalli', 'VICTOR': 'Ravalli', 'CORVALLIS': 'Ravalli', 'TROUT': 'Sanders',
    'NOXON': 'Sanders', 'HERON': 'Sanders', 'BUSBY': 'Big Horn', 'DECKER': 'Big Horn',
    'ASHLAND': 'Rosebud', 'SHEPHERD': 'Yellowstone', 'ACTON': 'Yellowstone', 'ROBERTS': 'Carbon',
    'BOX ELDER': 'Hill', 'BLACK EAGLE': 'Cascade', 'EAST GLACIER PARK': 'Glacier',
    'MC ALLISTER': 'Madison', 'NORRIS': 'Madison', 'HATHAWAY': 'Rosebud', 'SAVAGE': 'Richland',
    'BAINVILLE': 'Roosevelt', 'BASIN': 'Jefferson', 'BOX ELDER': 'Hill',
}

disp = pd.read_csv(os.path.join(DATA, 'raw', 'mt_dispensary_locations.csv'))
disp['sales_type'] = disp['sales_type'].astype(str)
retail = disp[disp['sales_type'].str.contains('Adult', case=False, na=False)].copy()
retail['city_key'] = retail['city'].astype(str).str.strip().str.upper()
retail['county_key'] = retail['city_key'].map(CITY_COUNTY)
matched = retail['county_key'].notna().sum()
print(f"Adult-use dispensaries: {len(retail)}, matched to a county: {matched} ({matched/len(retail):.0%})")
unmatched_cities = sorted(retail[retail['county_key'].isna()]['city_key'].unique())
print(f"Unmatched cities ({len(unmatched_cities)}): {unmatched_cities}")

counts = retail.dropna(subset=['county_key']).groupby('county_key').size().rename('dispensary_count').reset_index()
counts['county_key'] = counts['county_key'].str.strip().str.upper()

panel = pd.read_csv(os.path.join(TABLES, 'montana_revenue_percapita_panel.csv'), dtype={'fips5': str})
latest = panel[panel['fy_year'] == panel['fy_year'].max()].copy()
latest['county_key2'] = latest['county_name'].str.replace(' County', '', regex=False).str.strip().str.upper()

cross = latest.merge(counts, left_on='county_key2', right_on='county_key', how='left')
cross['dispensary_count'] = cross['dispensary_count'].fillna(0)
cross = cross[cross['dispensary_count'] > 0].copy()
cross['revenue_per_store'] = cross['total_sales'] / cross['dispensary_count']
cross['log_revenue_per_store'] = np.log(cross['revenue_per_store'])

print(f"\nCross-section year: {latest['fy_year'].max()}, {len(cross)} counties with >=1 matched retailer, "
      f"border={cross['border_70mi'].sum()}")
print(cross[['county_name', 'border_70mi', 'total_sales', 'dispensary_count', 'revenue_per_store']]
      .sort_values('border_70mi', ascending=False).to_string(index=False))

m = smf.ols('log_revenue_per_store ~ border_70mi + log_population', data=cross).fit(cov_type='HC1')
b, se, p = m.params['border_70mi'], m.bse['border_70mi'], m.pvalues['border_70mi']
print(f"\n+ log(pop), single cross-section  b={b:+.4f} se(HC1)={se:.4f} p={p:.4f} n={int(m.nobs)}")

cross.to_csv(os.path.join(TABLES, 'montana_revenue_perstore_cross.csv'), index=False)
print(f"\nSaved to {TABLES}")
