/* ================================================================
   MARIJUANA LEGALIZATION CROSS-BORDER SPILLOVER: DATA PREPARATION
   01_data_prep.do

   Data sources:
     - CDC PLACES 2022 release (GIS, county, BRFSS 2019-2020) → period 2020
     - CDC PLACES 2025 release (standard, county, BRFSS 2022-2023) → period 2023
     - US Census 2023 Gazetteer (county centroids)
     - Recreational marijuana retail sales dates (hand-coded)
   ================================================================ */

clear all
set more off
set linesize 120

global DIR `"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect"'
global DATA "$DIR\data"
global FIG  "$DIR\figures"
global TAB  "$DIR\tables"

cd "$DIR"
cap mkdir "$DATA"
cap mkdir "$FIG"
cap mkdir "$TAB"

// ── Install packages ────────────────────────────────────────────────
foreach pkg in reghdfe ftools csdid drdid event_plot coefplot estout outreg2 {
    cap which `pkg'
    if _rc ssc install `pkg', replace
}

// ================================================================
// PART 1: LEGALIZATION DATES
// ================================================================
// Source: state statutes and news records
// Date: year retail recreational marijuana sales began
// DC excluded (possession-only, no retail market)

preserve
clear
input int state_fips str2 abbr int rec_year
   2  "AK"  2016
   4  "AZ"  2021
   6  "CA"  2018
   8  "CO"  2014
   9  "CT"  2023
  17  "IL"  2020
  24  "MD"  2023
  25  "MA"  2018
  26  "MI"  2019
  27  "MN"  2024
  29  "MO"  2023
  30  "MT"  2022
  32  "NV"  2017
  34  "NJ"  2022
  35  "NM"  2022
  36  "NY"  2022
  39  "OH"  2024
  41  "OR"  2015
  44  "RI"  2022
  50  "VT"  2022
  53  "WA"  2014
end
label var state_fips "State FIPS (2-digit)"
label var abbr        "State abbreviation"
label var rec_year    "Year retail recreational marijuana sales began"
save "$DATA\legalization_dates.dta", replace
restore

// ================================================================
// PART 2: COUNTY CENTROIDS (Census Gazetteer 2023)
// Pre-downloaded via PowerShell to $DATA\2023_Gaz_counties_national.txt
// ================================================================
import delimited "$DATA\2023_Gaz_counties_national.txt", varnames(1) delimiters(tab) clear

rename geoid    county_fips_num
rename usps     state_abbr
rename intptlat lat
rename intptlong lon
rename name     county_name

// 5-digit string FIPS (preserve leading zeros)
gen fips5 = string(county_fips_num, "%05.0f")
gen state_fips = floor(county_fips_num / 1000)

keep fips5 county_fips_num state_fips state_abbr county_name lat lon
save "$DATA\county_centroids.dta", replace
di "County centroids saved: `c(N)' counties"

// ================================================================
// PART 3A: CDC PLACES 2022 RELEASE (GIS wide format, period 2020)
// Pre-downloaded to $DATA\places_2022_gis.csv
// BRFSS data years: 2019-2020  → label as period 2020
// Key variables: mhlth_crudeprev, depression_crudeprev
// ================================================================
import delimited "$DATA\places_2022_gis.csv", varnames(1) clear

keep stateabbr statedesc countyfips totalpopulation mhlth_crudeprev depression_crudeprev

rename stateabbr    state_abbr
rename statedesc    state_name
rename mhlth_crudeprev  mhlth
rename depression_crudeprev depression
rename totalpopulation population

destring mhlth depression population countyfips, replace force
// Ensure 5-digit zero-padded string FIPS for merging
gen fips5 = string(countyfips, "%05.0f")
drop countyfips

gen year = 2020
gen release = "PLACES-2022 (BRFSS 2019-2020)"
save "$DATA\places_period1.dta", replace
di "Period 1 (2020): `c(N)' county observations"

// ================================================================
// PART 3B: CDC PLACES 2025 RELEASE (standard long format, period 2023)
// Pre-downloaded to $DATA\places_2025_mhlth.csv and places_2025_dep.csv
// Filtered to crude prevalence only (2958 counties each)
// BRFSS data years: 2022-2023  → label as period 2023
// ================================================================
import delimited "$DATA\places_2025_mhlth.csv", varnames(1) clear
// locationid may arrive numeric (leading zeros stripped by CSV parser)
// tostring with format pads zeros; if already string, cap suppresses error
cap tostring locationid, replace format(%05.0f)
rename locationid   fips5
rename stateabbr    state_abbr
rename statedesc    state_name
rename locationname county_name_p2
rename data_value   mhlth
keep fips5 state_abbr state_name county_name_p2 mhlth
save "$DATA\places_p2_mhlth.dta", replace

import delimited "$DATA\places_2025_dep.csv", varnames(1) clear
cap tostring locationid, replace format(%05.0f)
rename locationid   fips5
rename data_value   depression
keep fips5 depression
save "$DATA\places_p2_dep.dta", replace

// Merge the two measures for period 2
use "$DATA\places_p2_mhlth.dta", clear
merge 1:1 fips5 using "$DATA\places_p2_dep.dta", nogen
gen year = 2023
gen release = "PLACES-2025 (BRFSS 2022-2023)"
save "$DATA\places_period2.dta", replace
di "Period 2 (2023): `c(N)' county observations"

// ================================================================
// PART 4: STACK PERIODS, MERGE CENTROIDS & LEGALIZATION DATES
// ================================================================
use "$DATA\places_period1.dta", clear
append using "$DATA\places_period2.dta"

// Drop territories
drop if inlist(state_abbr, "PR", "VI", "GU", "MP", "AS", "")

// Merge county centroids
merge m:1 fips5 using "$DATA\county_centroids.dta", keepusing(lat lon state_fips county_fips_num)
drop if _merge != 3
drop _merge

// Merge legalization dates (state-level)
merge m:1 state_fips using "$DATA\legalization_dates.dta", keepusing(rec_year)
// Unmatched: states without recreational marijuana → rec_year stays missing
drop _merge

// Flags for which period each state is "rec"
// (using rec_year <= 2019 for period 1; rec_year <= 2023 for period 2)
gen is_rec_2020 = (!missing(rec_year) & rec_year <= 2019)
gen is_rec_2023 = (!missing(rec_year) & rec_year <= 2023)

// Never-legalizing through our panel: rec_year missing or > 2023
gen never_legal = missing(rec_year) | rec_year > 2023

label var mhlth        "Mental health not good >=14 days (%)"
label var depression   "Depression prevalence (%)"
label var is_rec_2020  "State had rec marijuana retail in 2020 period"
label var is_rec_2023  "State had rec marijuana retail in 2023 period"
label var never_legal  "State never legalized rec through 2023"

save "$DATA\places_merged.dta", replace

// ================================================================
// PART 5: DISTANCE COMPUTATION (Mata – Haversine)
// For each county in a never-legalizing state:
//   dist_rec_2020_km = min distance to any county centroid in rec states (rec_year <= 2019)
//   dist_rec_2023_km = min distance to any county centroid in rec states (rec_year <= 2023)
// ================================================================

// Build unique county-centroid reference (one row per county)
use "$DATA\places_merged.dta", clear
bysort fips5: keep if _n == 1
keep fips5 county_fips_num state_abbr state_fips lat lon is_rec_2020 is_rec_2023 never_legal rec_year

// ── Mata distance function ────────────────────────────────────────
mata:
real colvector min_dist_haversine(real colvector lat, real colvector lon,
                                   real colvector is_rec) {
    real scalar R, pi, i
    real rowvector dlat, dlon, a, c
    real colvector lat_rec, lon_rec, min_d

    R   = 6371           // Earth radius (km)
    pi  = pi()

    // Rec county centroids
    lat_rec = select(lat, is_rec :== 1)
    lon_rec = select(lon, is_rec :== 1)

    if (rows(lat_rec) == 0) return(J(rows(lat), 1, .))

    min_d = J(rows(lat), 1, .)

    for (i = 1; i <= rows(lat); i++) {
        if (is_rec[i] == 0) {
            dlat = (lat_rec :- lat[i]) :* pi / 180
            dlon = (lon_rec :- lon[i]) :* pi / 180
            a    = sin(dlat:/2):^2 :+
                   cos(lat[i]*pi/180) :* cos(lat_rec:*pi/180) :* sin(dlon:/2):^2
            // Stata Mata atan2(x,y)=arctan(y/x) — swap vs standard atan2(y,x)
            c    = 2 :* atan2(sqrt(1:-a), sqrt(a))
            min_d[i] = min(R :* c)
        }
    }
    return(min_d)
}
end

di "Computing distances to 2020 rec borders..."
mata:
lat    = st_data(., "lat")
lon    = st_data(., "lon")
is_rec = st_data(., "is_rec_2020")
d = min_dist_haversine(lat, lon, is_rec)
(void) st_addvar("double", "dist_rec_2020_km")
st_store(., "dist_rec_2020_km", d)
end

di "Computing distances to 2023 rec borders..."
mata:
lat    = st_data(., "lat")
lon    = st_data(., "lon")
is_rec = st_data(., "is_rec_2023")
d = min_dist_haversine(lat, lon, is_rec)
(void) st_addvar("double", "dist_rec_2023_km")
st_store(., "dist_rec_2023_km", d)
end

// Convert to miles and create treatment variables
gen dist_rec_2020_mi = dist_rec_2020_km * 0.621371
gen dist_rec_2023_mi = dist_rec_2023_km * 0.621371

// Binary treatment at 100-mile threshold
gen treat_2020 = (dist_rec_2020_mi < 100) if !missing(dist_rec_2020_mi)
gen treat_2023 = (dist_rec_2023_mi < 100) if !missing(dist_rec_2023_mi)

// Switcher: county gained a rec marijuana neighbor (i.e., distance DECREASED)
gen switcher = (treat_2020 == 0 & treat_2023 == 1) if never_legal

// Distance change
gen dist_change_mi = dist_rec_2023_mi - dist_rec_2020_mi

// Log distances (for continuous treatment specification)
gen log_dist_2020 = log(dist_rec_2020_mi + 1)
gen log_dist_2023 = log(dist_rec_2023_mi + 1)

// C&S gvar: year county first got a rec neighbor within 100 mi
// Approximate: use median rec_year of nearby newly-legalized state
// Simplified: gvar = 2023 for switchers (new rec neighbor appeared 2020-2023)
//             gvar = 0  for never-treated (dist_rec_2023_mi >= 100)
gen gvar = 0
replace gvar = 2020 if treat_2020 == 1 & never_legal  // always near rec
replace gvar = 2023 if switcher == 1

label var dist_rec_2020_mi "Dist to nearest rec border in 2020 (miles)"
label var dist_rec_2023_mi "Dist to nearest rec border in 2023 (miles)"
label var treat_2020       "Within 100 mi of rec border in 2020"
label var treat_2023       "Within 100 mi of rec border in 2023"
label var switcher         "Gained rec neighbor 2020-2023 (switcher)"
label var gvar             "C&S first-treated year (0=never near rec)"

save "$DATA\county_distances.dta", replace
di "Distance file saved: `c(N)' counties"

// ================================================================
// PART 6: FINAL ANALYSIS PANEL
// ================================================================
use "$DATA\places_merged.dta", clear

// Merge distances
merge m:1 fips5 using "$DATA\county_distances.dta", ///
    keepusing(dist_rec_2020_mi dist_rec_2023_mi log_dist_2020 log_dist_2023 ///
              treat_2020 treat_2023 switcher dist_change_mi gvar)
drop if _merge != 3
drop _merge

// Year-specific distance and treatment
gen dist_rec_mi = cond(year == 2020, dist_rec_2020_mi, dist_rec_2023_mi)
gen log_dist    = cond(year == 2020, log_dist_2020,   log_dist_2023)
gen treat_bin   = cond(year == 2020, treat_2020,       treat_2023)

// Panel DiD variables
gen post              = (year == 2023)
gen switcher_post     = switcher * post           // main DiD interaction
gen log_dist_post     = log_dist * post           // continuous version

// Distance bands (period 2 distance)
gen dist_band = .
replace dist_band = 1 if dist_rec_2023_mi <= 50
replace dist_band = 2 if dist_rec_2023_mi > 50  & dist_rec_2023_mi <= 100
replace dist_band = 3 if dist_rec_2023_mi > 100 & dist_rec_2023_mi <= 200
replace dist_band = 4 if dist_rec_2023_mi > 200 & dist_rec_2023_mi <= 300
replace dist_band = 5 if dist_rec_2023_mi > 300

label define dband 1 "0-50 mi" 2 "50-100 mi" 3 "100-200 mi" 4 "200-300 mi" 5 ">300 mi"
label values dist_band dband

// Sample groups for reporting
gen sample_grp = ""
replace sample_grp = "Always near rec border"  if treat_2020 == 1 & never_legal
replace sample_grp = "Switcher (newly near rec)" if switcher == 1
replace sample_grp = "Never near rec border"   if treat_2020 == 0 & treat_2023 == 0 & never_legal

// Restrict to never-legalizing states
keep if never_legal == 1

// Panel identifiers
encode fips5, gen(county_id)
encode state_abbr, gen(state_id)
xtset county_id year

label var switcher_post  "Switcher × Post (main DiD)"
label var log_dist_post  "Log(dist) × Post (continuous DiD)"
label var post           "Post (2023 period)"
label var treat_bin      "Within 100 mi of rec border"

save "$DATA\analysis_panel.dta", replace

// Quick check
di "Analysis panel:"
tab year
tab sample_grp
sum mhlth depression dist_rec_mi if year == 2023

di "01_data_prep.do COMPLETE"
