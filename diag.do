clear all
global DATA `"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect\data"'

// Check county_meta contents
use "$DATA\places_merged.dta", clear
bysort fips5: keep if _n == 1
keep fips5 state_abbr state_fips lat lon is_rec_2020 is_rec_2023 never_legal rec_year

di "=== is_rec_2020 distribution ==="
tab is_rec_2020
di "=== is_rec_2023 distribution ==="
tab is_rec_2023
di "=== Sample lat/lon (first 5 rows) ==="
list fips5 state_abbr lat lon is_rec_2020 is_rec_2023 in 1/10
di "=== California counties (FIPS starts with 06) ==="
list fips5 state_abbr lat lon is_rec_2020 is_rec_2023 if state_fips == 6 in 1/3
di "=== rec_year summary ==="
sum rec_year

// Quick Mata test: distance from Indiana county to California county
mata:
lat1 =  40.0; lon1 = -86.0  // Indiana
lat2 =  37.5; lon2 = -119.5 // California
R = 6371; pi = pi()
dlat = (lat2 - lat1) * pi / 180
dlon = (lon2 - lon1) * pi / 180
a    = sin(dlat/2)^2 + cos(lat1*pi/180) * cos(lat2*pi/180) * sin(dlon/2)^2
c    = 2 * atan2(sqrt(a), sqrt(1-a))
dist_km = R * c
dist_mi = dist_km * 0.621371
printf("Mata test: Indiana to California = %f km = %f miles\n", dist_km, dist_mi)
end
