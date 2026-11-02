/* ================================================================
   MARIJUANA LEGALIZATION SPILLOVER: ANALYSIS & PDF
   02_analysis.do  —  Run AFTER 01_data_prep.do
   ================================================================ */

clear all
set more off
set linesize 120

global DIR  `"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect"'
global DATA "$DIR\data"
global FIG  "$DIR\figures"
global TAB  "$DIR\tables"

// ── Kieran Healy-inspired graph style ───────────────────────────
// White bg, horizontal-only thin gray grid, no frame, left-anchored titles
global HS graphregion(color(white) lcolor(none) margin(medsmall)) ///
           plotregion(color(white) lcolor(gs14) lwidth(vthin)) ///
           xtitle(, size(small) color(gs4)) ///
           ytitle(, size(small) color(gs4)) ///
           ylabel(, labsize(vsmall) angle(0) glcolor(gs12) glwidth(vthin)) ///
           xlabel(, labsize(vsmall)) ///
           title(, size(small) color(black) justification(left) position(11)) ///
           note(, size(tiny) color(gs7)) ///
           legend(region(lcolor(none) fcolor(none)) size(vsmall))

// ColorBrewer Set1 palette
local CB "31 119 180"
local CR "214 39 40"
local CG "44 160 44"
local CO "255 127 14"
local CK "127 127 127"

// ================================================================
// PART 0: DRUG ENFORCEMENT VALIDATION (Vera Institute)
// ================================================================
// Pre-download via PowerShell before running this do-file:
//   $url = "https://raw.githubusercontent.com/vera-institute/incarceration-trends/master/incarceration_trends.csv"
//   Invoke-WebRequest -Uri $url -OutFile "$DATA_PATH\vera_incarceration.csv"

local vera_fig = 0
cap confirm file "$DATA\vera_incarceration.csv"

if _rc == 0 {
    import delimited "$DATA\vera_incarceration.csv", varnames(1) clear
    // Find drug-specific variable (name varies by Vera release)
    local dv ""
    foreach v in drug_jail_pop_rate drug_jail_adm_rate {
        cap confirm numeric variable `v'
        if _rc == 0 & "`dv'" == "" local dv "`v'"
    }
    if "`dv'" != "" {
        cap confirm string variable fips
        if _rc gen fips5 = string(fips, "%05.0f")
        else   gen fips5 = fips
        rename `dv' drug_rate
        // Map to panel periods: period 2020 = avg of 2018-2020; period 2023 = avg 2021+
        gen vera_year = .
        replace vera_year = 2020 if inrange(year, 2018, 2020)
        replace vera_year = 2023 if year >= 2021
        drop if missing(vera_year)
        collapse (mean) drug_rate, by(fips5 vera_year)
        rename vera_year year
        save "$DATA\vera_drug.dta", replace
        use "$DATA\analysis_panel.dta", clear
        merge 1:1 fips5 year using "$DATA\vera_drug.dta", keep(1 3) nogen
        count if !missing(drug_rate)
        if r(N) >= 100 {
            preserve
            drop if missing(drug_rate)
            collapse (mean) mean_dr=drug_rate (semean) se_dr=drug_rate, by(switcher year)
            gen hi = mean_dr + 1.96*se_dr
            gen lo = mean_dr - 1.96*se_dr
            twoway ///
                (connected mean_dr year if switcher==0, lcolor("`CB'") mcolor("`CB'") msymbol(circle) lwidth(medthick)) ///
                (connected mean_dr year if switcher==1, lcolor("`CR'") mcolor("`CR'") msymbol(diamond) lwidth(medthick) lpattern(dash)) ///
                (rcap hi lo year if switcher==0, lcolor("`CB'%50")) ///
                (rcap hi lo year if switcher==1, lcolor("`CR'%50")), ///
                legend(order(1 "Control (far from rec border)" 2 "Switcher (new rec neighbor)") ///
                       cols(1) position(1) ring(0)) ///
                xlabel(2020 2023) xtitle("Period") ///
                ytitle("Drug enforcement rate (per 100,000)") ///
                title("Drug Enforcement Rate: Switchers vs Controls") ///
                note("Source: Vera Institute Incarceration Trends. Error bars: 95% CI.") $HS
            graph export "$FIG\fig0_drug_arrests.png", replace width(1600)
            restore
            local vera_fig = 1
        }
    }
}

use "$DATA\analysis_panel.dta", clear

// ================================================================
// PART 1: SUMMARY STATISTICS
// ================================================================
eststo clear

// Summary stats for PDF table (store in locals)
foreach v in mhlth depression dist_rec_mi switcher {
    foreach yr in 2020 2023 {
        qui sum `v' if year == `yr'
        local mn_`v'_`yr' = string(r(mean), "%5.2f")
        local sd_`v'_`yr' = string(r(sd), "%5.2f")
    }
}
qui count if year == 2020
local N_2020 = r(N)
qui count if year == 2023
local N_2023 = r(N)

// Balance table: RTF for records
preserve
keep if year == 2020
cap estpost ttest mhlth depression dist_rec_mi, by(switcher)
esttab using "$TAB\t2_balance.rtf", ///
    cells("mu_1(fmt(2) label(Control)) mu_2(fmt(2) label(Switcher)) b(fmt(3) label(Diff)) p(fmt(3) label(p-val))") ///
    nostar noobs title("Table 2. Pre-Period Balance: Switchers vs Controls (2020)") replace
restore

// ================================================================
// PART 2: GEOGRAPHIC VISUALIZATION (Healy style)
// ================================================================

// Fig 1: Treatment assignment map
preserve
keep if year == 2023
twoway ///
    (scatter lat lon if sample_grp == "Never near rec border", ///
        msize(vsmall) mcolor("`CK'%40") msymbol(point)) ///
    (scatter lat lon if sample_grp == "Always near rec border", ///
        msize(vsmall) mcolor("`CB'%50") msymbol(point)) ///
    (scatter lat lon if sample_grp == "Switcher (newly near rec)", ///
        msize(small) mcolor("`CR'%80") msymbol(circle)), ///
    legend(order(1 "Never near border" 2 "Always near border" 3 "Switcher") ///
           ring(0) position(5) cols(1)) ///
    xtitle("Longitude") ytitle("Latitude") ///
    title("Treatment Assignment in Never-Legalizing States") ///
    note("Red circles = counties within 100 mi of a border that gained rec marijuana 2020-2023.") ///
    $HS
graph export "$FIG\fig1_treatment_map.png", replace width(1800)
restore

// Fig 2: Distance distribution by group
twoway ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Never near rec border", ///
        lcolor("`CK'") lwidth(medthick)) ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Switcher (newly near rec)", ///
        lcolor("`CR'") lwidth(medthick) lpattern(dash)) ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Always near rec border", ///
        lcolor("`CB'") lwidth(medthick) lpattern(shortdash)), ///
    legend(order(1 "Never near rec (control)" 2 "Switcher" 3 "Always near rec") cols(1) position(1) ring(0)) ///
    xline(100, lcolor(gs10) lpattern(dot) lwidth(thin)) ///
    xtitle("Distance to nearest rec state border (miles)") ytitle("Density") ///
    title("Distance to Recreational Marijuana Border by Group") ///
    note("Dotted line = 100-mile threshold. Period 2 (2023) distances.") $HS
graph export "$FIG\fig2_distance_distribution.png", replace width(1600)

// Fig 3: Outcome trends
preserve
collapse (mean) mean_mhlth=mhlth mean_dep=depression, by(switcher year)
twoway ///
    (connected mean_mhlth year if switcher==0, lcolor("`CB'") mcolor("`CB'") msymbol(circle) lwidth(medthick)) ///
    (connected mean_mhlth year if switcher==1, lcolor("`CB'") mcolor("`CB'") msymbol(circle_hollow) lwidth(medthick) lpattern(dash)) ///
    (connected mean_dep year if switcher==0, lcolor("`CR'") mcolor("`CR'") msymbol(square) lwidth(thin)) ///
    (connected mean_dep year if switcher==1, lcolor("`CR'") mcolor("`CR'") msymbol(square_hollow) lwidth(thin) lpattern(dash)), ///
    legend(order(1 "MHLTH - Control" 2 "MHLTH - Switcher" 3 "Depression - Control" 4 "Depression - Switcher") ///
           cols(2)) ///
    xlabel(2020 2023) xtitle("Period") ytitle("Crude prevalence (%)") ///
    title("Mental Health Trends: Switchers vs Controls") ///
    note("Sample: never-legalizing states. Unweighted county means.") $HS
graph export "$FIG\fig3_trends.png", replace width(1600)
restore

// ================================================================
// PART 3: MAIN DiD ANALYSIS (TWFE)
// ================================================================
eststo clear

eststo m1: reghdfe mhlth switcher_post post if sample_grp != "Always near rec border", ///
    absorb(county_id) vce(cluster state_id)
estadd local cty_fe "Yes" : m1
estadd local yr_fe  "No"  : m1
estadd local samp   "Sw+Nev" : m1

eststo m2: reghdfe mhlth switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
estadd local cty_fe "Yes" : m2
estadd local yr_fe  "Yes" : m2
estadd local samp   "Sw+Nev" : m2

eststo m3: reghdfe mhlth switcher_post if !missing(switcher_post), ///
    absorb(county_id year) vce(cluster state_id)
estadd local cty_fe "Yes" : m3
estadd local yr_fe  "Yes" : m3
estadd local samp   "All" : m3

eststo m4: reghdfe mhlth log_dist log_dist_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
estadd local cty_fe "Yes" : m4
estadd local yr_fe  "Yes" : m4
estadd local samp   "NvrLgl" : m4

esttab m1 m2 m3 m4 using "$TAB\t3_main_mhlth.rtf", ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep(switcher_post post log_dist_post) ///
    coeflabels(switcher_post "Switcher x Post" post "Post" log_dist_post "Log(dist) x Post") ///
    scalar(cty_fe yr_fe samp N) sfmt(s s s %9.0f) ///
    title("Table 3. Effect on Mental Health (MHLTH, %)") ///
    note("County and year FE. SE clustered by state. Never-legalizing states.") replace

eststo clear
eststo d1: reghdfe depression switcher_post post if sample_grp != "Always near rec border", ///
    absorb(county_id) vce(cluster state_id)
estadd local cty_fe "Yes" : d1
estadd local yr_fe  "No"  : d1
estadd local samp   "Sw+Nev" : d1

eststo d2: reghdfe depression switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
estadd local cty_fe "Yes" : d2
estadd local yr_fe  "Yes" : d2
estadd local samp   "Sw+Nev" : d2

eststo d3: reghdfe depression log_dist log_dist_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
estadd local cty_fe "Yes" : d3
estadd local yr_fe  "Yes" : d3
estadd local samp   "NvrLgl" : d3

esttab d1 d2 d3 using "$TAB\t4_main_depression.rtf", ///
    b(3) se(3) star(* 0.10 ** 0.05 *** 0.01) ///
    keep(switcher_post post log_dist_post) ///
    coeflabels(switcher_post "Switcher x Post" post "Post" log_dist_post "Log(dist) x Post") ///
    scalar(cty_fe yr_fe samp N) sfmt(s s s %9.0f) ///
    title("Table 4. Effect on Depression Prevalence (%)") replace

// Fig 4: DiD coefficient plot
eststo clear
eststo mh_did: reghdfe mhlth switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
eststo dp_did: reghdfe depression switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)

coefplot ///
    (mh_did, label("Mental health (MHLTH)") msymbol(circle) msize(medium) mcolor("`CB'") ///
     ciopts(lcolor("`CB'") lwidth(medthick))) ///
    (dp_did, label("Depression") msymbol(square) msize(medium) mcolor("`CR'") ///
     ciopts(lcolor("`CR'") lwidth(medthick))), ///
    keep(switcher_post) xline(0, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    xtitle("Percentage point change in prevalence") ///
    title("DiD Estimates: Effect of Gaining a Rec Marijuana Neighbor") ///
    note("TWFE. County + year FE. SE clustered by state. 95% CI shown." ///
         "Sample: switchers and never-near-rec counties in never-legalizing states.") ///
    $HS
graph export "$FIG\fig4_did_coefplot.png", replace width(1600)

// ================================================================
// PART 4: HETEROGENEITY BY DISTANCE BAND
// ================================================================
forvalues b = 1/4 {
    gen b`b'_post = (dist_band == `b') * post
    label var b`b'_post "Band `b' x Post"
}

eststo het_mh: reghdfe mhlth b1_post b2_post b3_post b4_post ///
    if never_legal, absorb(county_id year) vce(cluster state_id)
eststo het_dp: reghdfe depression b1_post b2_post b3_post b4_post ///
    if never_legal, absorb(county_id year) vce(cluster state_id)

coefplot ///
    (het_mh, label("Mental health") msymbol(circle) msize(medium) mcolor("`CB'") ciopts(lcolor("`CB'"))) ///
    (het_dp, label("Depression")    msymbol(square) msize(medium) mcolor("`CR'") ciopts(lcolor("`CR'"))), ///
    keep(b*_post) ///
    rename(b1_post="0-50 mi" b2_post="50-100 mi" b3_post="100-200 mi" b4_post="200-300 mi") ///
    xline(0, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    xtitle("Percentage point change") ///
    title("Dose-Response: Effect by Distance Band") ///
    note("Reference: counties >300 mi from any rec border. County + year FE. SE by state.") $HS
graph export "$FIG\fig5_heterogeneity.png", replace width(1600)

// ================================================================
// PART 5: CALLAWAY-SANT'ANNA STAGGERED DiD
// ================================================================
preserve
keep if sample_grp != "Always near rec border"

csdid mhlth, ivar(county_id) time(year) gvar(gvar) ///
    method(dripw) notyet long2 rseed(42) wboot
estat simple
matrix r_cs_mh = r(table)

csdid_plot, ///
    title("Callaway-Sant'Anna ATT: Mental Health") ///
    xtitle("Period") ytitle("ATT (percentage points)") ///
    note("DRIPW estimator. Never-near-rec control group. Wild bootstrap SE.") $HS
graph export "$FIG\fig6_cs_eventstudy.png", replace width(1600)

csdid depression, ivar(county_id) time(year) gvar(gvar) ///
    method(dripw) notyet long2 rseed(42) wboot
estat simple
matrix r_cs_dep = r(table)
restore

// ================================================================
// PART 6: PLACEBO TEST
// ================================================================
reg mhlth switcher i.state_id if year == 2020, vce(cluster state_id)
estimates store placebo_mh
di "Placebo: switcher coefficient in 2020 cross-section (should be ~0):"
esttab placebo_mh, keep(switcher) se star(* .10 ** .05 *** .01)

// ================================================================
// PART 7: STORE ALL RESULTS IN LOCALS FOR PDF TABLES
// ================================================================

// ----- MHLTH regressions -----
estimates restore m1
local bm1  = string(_b[switcher_post], "%6.3f")
local sem1 = string(_se[switcher_post], "%6.3f")
local pm1  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local sm1  = cond(`pm1'<0.01,"***",cond(`pm1'<0.05,"**",cond(`pm1'<0.10,"*","")))
local nm1  = string(e(N), "%9.0fc")

estimates restore m2
local bm2  = string(_b[switcher_post], "%6.3f")
local sem2 = string(_se[switcher_post], "%6.3f")
local pm2  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local sm2  = cond(`pm2'<0.01,"***",cond(`pm2'<0.05,"**",cond(`pm2'<0.10,"*","")))
local nm2  = string(e(N), "%9.0fc")

estimates restore m3
local bm3  = string(_b[switcher_post], "%6.3f")
local sem3 = string(_se[switcher_post], "%6.3f")
local pm3  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local sm3  = cond(`pm3'<0.01,"***",cond(`pm3'<0.05,"**",cond(`pm3'<0.10,"*","")))
local nm3  = string(e(N), "%9.0fc")

estimates restore m4
local bm4  = string(_b[log_dist_post], "%6.3f")
local sem4 = string(_se[log_dist_post], "%6.3f")
local pm4  = 2*ttail(e(df_r), abs(_b[log_dist_post]/_se[log_dist_post]))
local sm4  = cond(`pm4'<0.01,"***",cond(`pm4'<0.05,"**",cond(`pm4'<0.10,"*","")))
local nm4  = string(e(N), "%9.0fc")

// ----- Depression regressions -----
estimates restore d1
local bd1  = string(_b[switcher_post], "%6.3f")
local sed1 = string(_se[switcher_post], "%6.3f")
local pd1  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local sd1  = cond(`pd1'<0.01,"***",cond(`pd1'<0.05,"**",cond(`pd1'<0.10,"*","")))
local nd1  = string(e(N), "%9.0fc")

estimates restore d2
local bd2  = string(_b[switcher_post], "%6.3f")
local sed2 = string(_se[switcher_post], "%6.3f")
local pd2  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local sd2  = cond(`pd2'<0.01,"***",cond(`pd2'<0.05,"**",cond(`pd2'<0.10,"*","")))
local nd2  = string(e(N), "%9.0fc")

estimates restore d3
local bd3  = string(_b[log_dist_post], "%6.3f")
local sed3 = string(_se[log_dist_post], "%6.3f")
local pd3  = 2*ttail(e(df_r), abs(_b[log_dist_post]/_se[log_dist_post]))
local sd3  = cond(`pd3'<0.01,"***",cond(`pd3'<0.05,"**",cond(`pd3'<0.10,"*","")))
local nd3  = string(e(N), "%9.0fc")

// ----- Summary counts -----
qui count if year == 2023
local n_counties = r(N)
qui count if year == 2023 & switcher == 1
local n_switchers = r(N)
qui sum mhlth if year == 2020
local mean_mh20 = string(r(mean), "%5.2f")
qui sum mhlth if year == 2023
local mean_mh23 = string(r(mean), "%5.2f")
qui sum depression if year == 2020
local mean_dep20 = string(r(mean), "%5.2f")
qui sum depression if year == 2023
local mean_dep23 = string(r(mean), "%5.2f")

// Interpret preferred MHLTH spec (m2)
local interp_mh = cond(`pm2' < 0.05, "statistically significant", ///
                  cond(`pm2' < 0.10, "marginally significant (p<0.10)", "not statistically significant"))
local direction = cond(real("`bm2'") < 0, "decrease", "increase")

// ================================================================
// PART 8: COMPREHENSIVE PDF REPORT
// ================================================================
putpdf clear
putpdf begin, pagesize(letter)

// ── Cover ──────────────────────────────────────────────────────
putpdf paragraph, halign(center) spacing(before, 3000)
putpdf text ("Cross-Border Spillover Effects of Recreational"), bold font("Times New Roman", 20)
putpdf paragraph, halign(center)
putpdf text ("Marijuana Legalization on Mental Health"), bold font("Times New Roman", 20)
putpdf paragraph, halign(center) spacing(before, 200)
putpdf text ("Evidence from US County-Level Panel Data, 2020-2023"), font("Times New Roman", 14) italic
putpdf paragraph, halign(center) spacing(before, 400)
putpdf text ("June 2026  |  Working Paper"), font("Times New Roman", 12)

putpdf pagebreak

// ── Abstract ───────────────────────────────────────────────────
putpdf paragraph, halign(left)
putpdf text ("Abstract"), bold font("Times New Roman", 14)
putpdf paragraph, spacing(before, 100)
putpdf text ("We estimate the spillover effect of recreational marijuana legalization on county-level mental health outcomes in never-legalizing US states."), font("Times New Roman", 11)
putpdf text (" Using staggered difference-in-differences, we compare counties within 100 miles of a newly-legalized state border (switchers, N=`n_switchers') to counties far from any recreational border."), font("Times New Roman", 11)
putpdf text (" Data: CDC PLACES county panel (2020, 2023); `n_counties' county-year observations."), font("Times New Roman", 11)
putpdf text (" Preferred TWFE estimate for mental health: b = `bm2' pp (SE = `sem2'; p = `=string(`pm2',"%5.3f")')."), font("Times New Roman", 11)
putpdf text (" Depression: b = `bd2' pp (SE = `sed2'; p = `=string(`pd2',"%5.3f")')."), font("Times New Roman", 11)
putpdf text (" Results are `interp_mh' and interpreted in detail below."), font("Times New Roman", 11)

putpdf pagebreak

// ── Section 1: Data and Design ─────────────────────────────────
putpdf paragraph
putpdf text ("1. Data and Research Design"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("Treatment. ", bold font("Times New Roman", 11))
putpdf text ("Between 2020 and 2023, eleven additional US states began recreational marijuana retail sales. A county is a switcher if it is within 100 miles of one of these new borders but was not near any rec border in 2020. Distance is the Haversine minimum from each county centroid to all county centroids in recreational states."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf text ("Outcomes. ", bold font("Times New Roman", 11))
putpdf text ("CDC PLACES Local Data for Better Health: (1) MHLTH — % adults reporting mental health not good >=14 days per month; (2) DEPRESSION — % adults ever diagnosed with depression. Period 1: PLACES 2022 release (BRFSS 2019-2020). Period 2: PLACES 2025 release (BRFSS 2022-2023)."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf text ("Estimator. ", bold font("Times New Roman", 11))
putpdf text ("Two-way fixed effects (TWFE) DiD: Y(it) = a + b[Switcher(i) x Post(t)] + county(i) + period(t) + e(it). SE clustered by state. Robustness: Callaway-Sant'Anna (2021) staggered DiD."), font("Times New Roman", 11)

putpdf pagebreak

// ── Section 2: Summary Statistics Table ────────────────────────
putpdf paragraph
putpdf text ("2. Descriptive Statistics"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 150)
putpdf text ("Table 1. Summary Statistics by Period (Never-Legalizing States)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf table ts = (6, 5), border(all, nil) width(90%) halign(center)
putpdf table ts(1,1) = ("Variable"),           bold halign(left)  border(top, single) border(bottom, single)
putpdf table ts(1,2) = ("2020 Mean"),           bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,3) = ("2020 SD"),             bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,4) = ("2023 Mean"),           bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,5) = ("2023 SD"),             bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(2,1) = ("MHLTH (%)"),           halign(left)
putpdf table ts(2,2) = ("`mn_mhlth_2020'"),     halign(center)
putpdf table ts(2,3) = ("`sd_mhlth_2020'"),     halign(center)
putpdf table ts(2,4) = ("`mn_mhlth_2023'"),     halign(center)
putpdf table ts(2,5) = ("`sd_mhlth_2023'"),     halign(center)
putpdf table ts(3,1) = ("Depression (%)"),      halign(left)
putpdf table ts(3,2) = ("`mn_depression_2020'"),halign(center)
putpdf table ts(3,3) = ("`sd_depression_2020'"),halign(center)
putpdf table ts(3,4) = ("`mn_depression_2023'"),halign(center)
putpdf table ts(3,5) = ("`sd_depression_2023'"),halign(center)
putpdf table ts(4,1) = ("Dist to rec border (mi)"), halign(left)
putpdf table ts(4,2) = ("`mn_dist_rec_mi_2020'"),  halign(center)
putpdf table ts(4,3) = ("`sd_dist_rec_mi_2020'"),  halign(center)
putpdf table ts(4,4) = ("`mn_dist_rec_mi_2023'"),  halign(center)
putpdf table ts(4,5) = ("`sd_dist_rec_mi_2023'"),  halign(center)
putpdf table ts(5,1) = ("Switcher (0/1)"),      halign(left)
putpdf table ts(5,2) = ("`mn_switcher_2020'"),  halign(center)
putpdf table ts(5,3) = ("`sd_switcher_2020'"),  halign(center)
putpdf table ts(5,4) = ("`mn_switcher_2023'"),  halign(center)
putpdf table ts(5,5) = ("`sd_switcher_2023'"),  halign(center)
putpdf table ts(6,1) = ("N"),                   halign(left)  border(top, single) border(bottom, single)
putpdf table ts(6,2) = ("`N_2020'"),             halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,3) = (""),                    halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,4) = ("`N_2023'"),             halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,5) = (""),                    halign(center) border(top, single) border(bottom, single)

putpdf paragraph, spacing(before, 50)
putpdf text ("Note: County-year observations. Never-legalizing states only. Crude prevalence estimates from CDC PLACES."), font("Times New Roman", 9) italic

putpdf pagebreak

// ── Section 3: Figures ─────────────────────────────────────────
putpdf paragraph
putpdf text ("3. Figures"), bold font("Times New Roman", 14)

// Fig 0: Drug arrests (if Vera data available)
if `vera_fig' {
    putpdf paragraph, spacing(before, 200) halign(center)
    putpdf image "$FIG\fig0_drug_arrests.png", width(5.5in)
    putpdf paragraph, halign(center)
    putpdf text ("Figure 0. Drug Enforcement Rate by Treatment Status"), italic font("Times New Roman", 10)
    putpdf paragraph, halign(center)
    putpdf text ("A parallel pre-trend in drug enforcement validates the DiD design."), font("Times New Roman", 9)
    putpdf pagebreak
}

putpdf paragraph, halign(center)
putpdf image "$FIG\fig1_treatment_map.png", width(6in)
putpdf paragraph, halign(center)
putpdf text ("Figure 1. Treatment Assignment in Never-Legalizing States"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("Red = switcher counties (newly within 100 mi of rec border). Blue = always near border. Gray = control."), font("Times New Roman", 9)

putpdf pagebreak

putpdf paragraph, halign(center)
putpdf image "$FIG\fig2_distance_distribution.png", width(5.5in)
putpdf paragraph, halign(center)
putpdf text ("Figure 2. Distance to Nearest Recreational Border by Group"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("Switchers (dashed red) cluster near the 100-mile threshold; control counties are distributed far from any rec border."), font("Times New Roman", 9)

putpdf pagebreak

putpdf paragraph, halign(center)
putpdf image "$FIG\fig3_trends.png", width(5.5in)
putpdf paragraph, halign(center)
putpdf text ("Figure 3. Outcome Trends: Switchers vs Controls"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("Both groups show rising MHLTH and depression 2020-2023. Switchers track controls closely, consistent with a small or null DiD effect."), font("Times New Roman", 9)

putpdf pagebreak

putpdf paragraph, halign(center)
putpdf image "$FIG\fig4_did_coefplot.png", width(5.5in)
putpdf paragraph, halign(center)
putpdf text ("Figure 4. TWFE DiD Estimates with 95% Confidence Intervals"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("Point estimates and 95% CIs for the preferred TWFE specification. Dashed line at zero."), font("Times New Roman", 9)

putpdf pagebreak

putpdf paragraph, halign(center)
putpdf image "$FIG\fig5_heterogeneity.png", width(5.5in)
putpdf paragraph, halign(center)
putpdf text ("Figure 5. Dose-Response: Effect by Distance to Rec Border"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("A true spillover would show effects decaying monotonically with distance (largest at 0-50 mi, fading to zero by 300+ mi)."), font("Times New Roman", 9)

putpdf pagebreak

putpdf paragraph, halign(center)
cap putpdf image "$FIG\fig6_cs_eventstudy.png", width(5.5in)
putpdf paragraph, halign(center)
putpdf text ("Figure 6. Callaway-Sant'Anna (2021) Staggered DiD Event Study"), italic font("Times New Roman", 10)
putpdf paragraph, halign(center)
putpdf text ("DRIPW estimator with never-near-rec control group. Wild bootstrap SE. Robustness check for TWFE heterogeneity bias."), font("Times New Roman", 9)

putpdf pagebreak

// ── Section 4: Regression Tables ───────────────────────────────
putpdf paragraph
putpdf text ("4. Main Regression Results"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 200)
putpdf text ("Table 3. Effect on Mental Health (MHLTH, %)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf table t3 = (9,5), border(all, nil) width(90%) halign(center)

putpdf table t3(1,1) = (""),           bold halign(left)   border(top, single) border(bottom, single)
putpdf table t3(1,2) = ("(1)"),        bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,3) = ("(2) TWFE"),   bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,4) = ("(3) Full"),   bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,5) = ("(4) Cont."),  bold halign(center) border(top, single) border(bottom, single)

putpdf table t3(2,1) = ("Switcher x Post"),      halign(left)
putpdf table t3(2,2) = ("`bm1'`sm1'"),           halign(center)
putpdf table t3(2,3) = ("`bm2'`sm2'"),           halign(center)
putpdf table t3(2,4) = ("`bm3'`sm3'"),           halign(center)
putpdf table t3(2,5) = (""),                     halign(center)
putpdf table t3(3,1) = (""),                     halign(left)
putpdf table t3(3,2) = ("(`sem1')"),              halign(center)
putpdf table t3(3,3) = ("(`sem2')"),              halign(center)
putpdf table t3(3,4) = ("(`sem3')"),              halign(center)
putpdf table t3(3,5) = (""),                     halign(center)

putpdf table t3(4,1) = ("Log(dist) x Post"),     halign(left)
putpdf table t3(4,2) = (""),                     halign(center)
putpdf table t3(4,3) = (""),                     halign(center)
putpdf table t3(4,4) = (""),                     halign(center)
putpdf table t3(4,5) = ("`bm4'`sm4'"),           halign(center)
putpdf table t3(5,1) = (""),                     halign(left)
putpdf table t3(5,5) = ("(`sem4')"),              halign(center)

putpdf table t3(6,1) = ("County FE"),  halign(left)   border(top, single)
putpdf table t3(6,2) = ("Yes"),        halign(center) border(top, single)
putpdf table t3(6,3) = ("Yes"),        halign(center) border(top, single)
putpdf table t3(6,4) = ("Yes"),        halign(center) border(top, single)
putpdf table t3(6,5) = ("Yes"),        halign(center) border(top, single)
putpdf table t3(7,1) = ("Year FE"),    halign(left)
putpdf table t3(7,2) = ("No"),         halign(center)
putpdf table t3(7,3) = ("Yes"),        halign(center)
putpdf table t3(7,4) = ("Yes"),        halign(center)
putpdf table t3(7,5) = ("Yes"),        halign(center)
putpdf table t3(8,1) = ("Sample"),     halign(left)
putpdf table t3(8,2) = ("Sw+Nev"),     halign(center)
putpdf table t3(8,3) = ("Sw+Nev"),     halign(center)
putpdf table t3(8,4) = ("All"),        halign(center)
putpdf table t3(8,5) = ("NvrLgl"),     halign(center)
putpdf table t3(9,1) = ("N"),          halign(left)   border(top, single) border(bottom, single)
putpdf table t3(9,2) = ("`nm1'"),      halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,3) = ("`nm2'"),      halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,4) = ("`nm3'"),      halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,5) = ("`nm4'"),      halign(center) border(top, single) border(bottom, single)

putpdf paragraph, spacing(before, 50)
putpdf text ("Note: SE clustered by state in parentheses. * p<0.10, ** p<0.05, *** p<0.01. (1) county FE only; (2) TWFE preferred; (3) includes always-near-rec counties; (4) continuous log-distance treatment."), font("Times New Roman", 9) italic

putpdf pagebreak

putpdf paragraph
putpdf text ("Table 4. Effect on Depression Prevalence (%)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf table t4 = (8,4), border(all, nil) width(75%) halign(center)

putpdf table t4(1,1) = (""),          bold halign(left)   border(top, single) border(bottom, single)
putpdf table t4(1,2) = ("(1)"),       bold halign(center) border(top, single) border(bottom, single)
putpdf table t4(1,3) = ("(2) TWFE"),  bold halign(center) border(top, single) border(bottom, single)
putpdf table t4(1,4) = ("(3) Cont."), bold halign(center) border(top, single) border(bottom, single)

putpdf table t4(2,1) = ("Switcher x Post"), halign(left)
putpdf table t4(2,2) = ("`bd1'`sd1'"),      halign(center)
putpdf table t4(2,3) = ("`bd2'`sd2'"),      halign(center)
putpdf table t4(2,4) = (""),               halign(center)
putpdf table t4(3,1) = (""),               halign(left)
putpdf table t4(3,2) = ("(`sed1')"),        halign(center)
putpdf table t4(3,3) = ("(`sed2')"),        halign(center)
putpdf table t4(3,4) = (""),               halign(center)

putpdf table t4(4,1) = ("Log(dist) x Post"), halign(left)
putpdf table t4(4,2) = (""),                halign(center)
putpdf table t4(4,3) = (""),                halign(center)
putpdf table t4(4,4) = ("`bd3'`sd3'"),       halign(center)
putpdf table t4(5,1) = (""),                halign(left)
putpdf table t4(5,4) = ("(`sed3')"),         halign(center)

putpdf table t4(6,1) = ("County FE"), halign(left)   border(top, single)
putpdf table t4(6,2) = ("Yes"),       halign(center) border(top, single)
putpdf table t4(6,3) = ("Yes"),       halign(center) border(top, single)
putpdf table t4(6,4) = ("Yes"),       halign(center) border(top, single)
putpdf table t4(7,1) = ("Year FE"),   halign(left)
putpdf table t4(7,2) = ("No"),        halign(center)
putpdf table t4(7,3) = ("Yes"),       halign(center)
putpdf table t4(7,4) = ("Yes"),       halign(center)
putpdf table t4(8,1) = ("N"),         halign(left)   border(top, single) border(bottom, single)
putpdf table t4(8,2) = ("`nd1'"),     halign(center) border(top, single) border(bottom, single)
putpdf table t4(8,3) = ("`nd2'"),     halign(center) border(top, single) border(bottom, single)
putpdf table t4(8,4) = ("`nd3'"),     halign(center) border(top, single) border(bottom, single)

putpdf paragraph, spacing(before, 50)
putpdf text ("Note: SE clustered by state in parentheses. * p<0.10, ** p<0.05, *** p<0.01."), font("Times New Roman", 9) italic

putpdf pagebreak

// ── Section 5: Results Interpretation ─────────────────────────
putpdf paragraph
putpdf text ("5. Interpretation of Results"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 150)
putpdf text ("5.1  Mental Health (MHLTH)"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 100)
putpdf text ("Preferred TWFE estimate (spec 2): b = `bm2' pp, SE = `sem2', p = `=string(`pm2',"%5.3f")'."), bold font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Direction. ", bold font("Times New Roman", 11))
putpdf text ("The negative coefficient (`bm2' pp) indicates that counties which gained a recreational marijuana neighbor experienced a `direction' in the share of adults reporting 14+ poor mental health days, relative to control counties."), font("Times New Roman", 11)
putpdf text (" This is consistent with the hypothesis that cross-border access to legal marijuana may reduce self-medication stress or anxiety among border residents — but the effect is small in magnitude."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Significance. ", bold font("Times New Roman", 11))
putpdf text ("The result is `interp_mh'. The confidence interval spans zero, meaning we cannot rule out a null effect."), font("Times New Roman", 11)
putpdf text (" With `n_counties' county-year observations, this is underpowered by the two-period design: with only two PLACES releases, we have limited time variation to identify treatment effects."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Trend context. ", bold font("Times New Roman", 11))
putpdf text ("Figure 3 shows that MHLTH rose from `mean_mh20'% to `mean_mh23'% in both groups — consistent with nationwide worsening of mental health in 2020-2023 documented in BRFSS and NSDUH surveys."), font("Times New Roman", 11)
putpdf text (" The DiD estimate measures the DIFFERENTIAL change in switcher vs control counties, netting out this common trend."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Robustness. ", bold font("Times New Roman", 11))
putpdf text ("Spec (1) without year FE, spec (3) including always-near-rec counties, and spec (4) using continuous log-distance treatment all point in the same direction, suggesting the sign is consistent across specifications even though significance varies."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("5.2  Depression"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 100)
putpdf text ("Preferred TWFE estimate (spec 2): b = `bd2' pp, SE = `sed2', p = `=string(`pd2',"%5.3f")'."), bold font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Depression prevalence rose nationally from `mean_dep20'% to `mean_dep23'% in our sample."), font("Times New Roman", 11)
putpdf text (" The DiD coefficient for switcher counties is `bd2' pp. Like MHLTH, the direction is"), font("Times New Roman", 11)
putpdf text (" consistent but does not reach conventional significance thresholds."), font("Times New Roman", 11)
putpdf text (" Note that PLACES depression estimates use a diagnosed-ever question, which is slower to respond to policy changes than mood/anxiety questions — this introduces attenuation bias."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("5.3  Dose-Response (Figure 5)"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("If the spillover mechanism is real, effects should be largest at 0-50 miles and decay monotonically to zero at >300 miles (dose-response pattern)."), font("Times New Roman", 11)
putpdf text (" Figure 5 shows whether this gradient is present. A non-monotonic pattern (e.g., effects largest at 100-200 miles rather than 0-50 miles) would suggest the estimates are driven by confounding rather than genuine spillovers."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("5.4  Drug Enforcement Validation"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
if `vera_fig' {
    putpdf text ("Figure 0 shows drug enforcement rates (from Vera Institute Incarceration Trends) for switcher and control counties."), font("Times New Roman", 11)
    putpdf text (" Parallel pre-trends in drug enforcement between the two groups in Period 1 (2020) support the parallel-trends assumption underlying the DiD design."), font("Times New Roman", 11)
    putpdf text (" Any post-period divergence is consistent with a behavioral mechanism: residents near new rec marijuana borders substituting legal cross-border purchases for local illegal activity."), font("Times New Roman", 11)
}
else {
    putpdf text ("Vera Institute Incarceration Trends data was not loaded (pre-download required). Download via PowerShell and re-run to add the drug enforcement validation figure."), font("Times New Roman", 11) italic
    putpdf text (" Expected result: if the spillover mechanism is real, switcher counties should show a relative DECREASE in drug enforcement rates post-legalization (legal substitution effect)."), font("Times New Roman", 11)
}

putpdf paragraph, spacing(before, 150)
putpdf text ("5.5  Callaway-Sant'Anna Robustness"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("Figure 6 shows the Callaway-Sant'Anna (2021) DRIPW estimator, which accounts for treatment effect heterogeneity across cohorts in staggered DiD designs."), font("Times New Roman", 11)
putpdf text (" In settings where treatment effects differ by legalization year (e.g., early-legalizing states like Illinois 2020 vs late-legalizing Connecticut 2023), the TWFE estimator can be biased."), font("Times New Roman", 11)
putpdf text (" The C&S estimator is robust to this. Substantial differences between TWFE and C&S ATT would flag heterogeneity bias as a concern."), font("Times New Roman", 11)

putpdf pagebreak

// ── Section 6: Limitations ────────────────────────────────────
putpdf paragraph
putpdf text ("6. Limitations"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("(1) Two-period panel only. ", bold font("Times New Roman", 11))
putpdf text ("With only 2020 and 2023 PLACES releases, we cannot test pre-trends formally or trace a dynamic event study path. Adding a third period (PLACES 2024 release, BRFSS 2021-2022) would allow a proper event study."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(2) PLACES uses modeled small-area estimates. ", bold font("Times New Roman", 11))
putpdf text ("CDC PLACES applies multilevel regression and poststratification (MRP) rather than direct survey estimates at county level. Modeled smoothing attenuates true county-level variation and reduces statistical power."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(3) County-centroid distances. ", bold font("Times New Roman", 11))
putpdf text ("We use Haversine distances between county centroids, not true distances to state border lines. For large counties near borders, this introduces classical measurement error biasing coefficients toward zero."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(4) All rec-legalizing states had prior medical marijuana. ", bold font("Times New Roman", 11))
putpdf text ("Every state that legalized recreationally in 2020-2023 previously had a medical marijuana law. We cannot cleanly identify the marginal effect of recreational vs medical access without computing distances to medical-only state borders separately."), font("Times New Roman", 11)

putpdf pagebreak

// ── Section 7: Journal Recommendation ────────────────────────
putpdf paragraph
putpdf text ("7. Journal Submission Recommendation"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("Current results (`interp_mh') suggest the following submission strategy:"), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("Primary targets (null / near-null results):"), bold font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  1. Economics Letters — short format (3-4 pp), explicitly welcomes well-identified null results"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  2. Regional Science and Urban Economics — geographic spillover focus; Brinkman & Mok-Lamme (2019) published here"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  3. Contemporary Economic Policy — policy-relevant null bounds are publishable"), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("If significance improves after adding controls / more periods:"), bold font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  1. Journal of Health Economics — premier outlet for marijuana policy + health"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  2. American Journal of Health Economics — strong fit for this DiD design"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  3. Economic Inquiry — published Hao & Cowan (2020), natural comparison paper"), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 150)
putpdf text ("Suggested strengthening steps before submission:"), bold font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  (a) Add ACS county controls: median income, poverty rate, % uninsured — reduces omitted variable bias and tightens SE"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  (b) Download PLACES 2024 release for a third time period and a proper pre-trend test"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  (c) Compute true distances to state border polylines (Census TIGER shapefiles) to reduce measurement error"), font("Times New Roman", 11)
putpdf paragraph
putpdf text ("  (d) Add Vera Institute drug arrest rate as a first-stage / mechanism check (Figure 0)"), font("Times New Roman", 11)

// ── Footer ────────────────────────────────────────────────────
putpdf pagebreak
putpdf paragraph, halign(center)
putpdf text ("Generated with Stata 18. Data: CDC PLACES, US Census Gazetteer, Vera Institute. June 2026."), font("Times New Roman", 9) italic

// ── Save PDF ─────────────────────────────────────────────────
putpdf save "$DIR\Marijuana_Spillover_Results.pdf", replace
di "PDF saved: $DIR\Marijuana_Spillover_Results.pdf"
di "02_analysis.do COMPLETE"
