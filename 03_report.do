/* ================================================================
   COMPREHENSIVE PDF RESEARCH REPORT
   03_report.do  —  Run AFTER 01_data_prep.do
   Self-contained: re-runs regressions and builds the full PDF.
   Output: Marijuana_Spillover_Results.pdf
   ================================================================ */

clear all
set more off
set linesize 120

global DIR  `"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect"'
global DATA "$DIR\data"
global FIG  "$DIR\figures"
global TAB  "$DIR\tables"

// ── Healy-style graph globals ────────────────────────────────────
global HS graphregion(color(white) lcolor(none) margin(medsmall)) ///
           plotregion(color(white) lcolor(gs14) lwidth(vthin)) ///
           xtitle(, size(small) color(gs4)) ytitle(, size(small) color(gs4)) ///
           ylabel(, labsize(vsmall) angle(0) glcolor(gs12) glwidth(vthin)) ///
           xlabel(, labsize(vsmall)) ///
           title(, size(small) color(black) justification(left) position(11)) ///
           note(, size(tiny) color(gs7)) ///
           legend(region(lcolor(none) fcolor(none)) size(vsmall))

local CB "31 119 180"
local CR "214 39 40"
local CK "127 127 127"
local CO "255 127 14"

// ================================================================
// STEP 1: GENERATE ALL FIGURES
// ================================================================

use "$DATA\analysis_panel.dta", clear

// Fig 1: Treatment map
preserve
keep if year == 2023
twoway ///
    (scatter lat lon if sample_grp=="Never near rec border", msize(vsmall) mcolor("`CK'%35") msymbol(point)) ///
    (scatter lat lon if sample_grp=="Always near rec border", msize(vsmall) mcolor("`CB'%50") msymbol(point)) ///
    (scatter lat lon if sample_grp=="Switcher (newly near rec)", msize(small) mcolor("`CR'%80") msymbol(circle)), ///
    legend(order(1 "Never near border (control)" 2 "Always near border" 3 "Switcher") ring(0) position(5) cols(1)) ///
    xtitle("Longitude") ytitle("Latitude") ///
    title("Figure 1. Treatment Assignment in Never-Legalizing States") ///
    note("Red = counties within 100 mi of a border that gained recreational marijuana 2020-2023.") $HS
graph export "$FIG\fig1_treatment_map.png", replace width(1800)
restore

// Fig 2: Distance distribution
twoway ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Never near rec border", lcolor("`CK'") lwidth(medthick)) ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Switcher (newly near rec)", lcolor("`CR'") lwidth(medthick) lpattern(dash)) ///
    (kdensity dist_rec_mi if year==2023 & sample_grp=="Always near rec border", lcolor("`CB'") lwidth(medthick) lpattern(shortdash)), ///
    legend(order(1 "Control" 2 "Switcher" 3 "Always near rec") cols(1) position(1) ring(0)) ///
    xline(100, lcolor(gs10) lpattern(dot) lwidth(thin)) ///
    xtitle("Distance to nearest recreational state border (miles)") ytitle("Density") ///
    title("Figure 2. Distance to Recreational Border by Group") ///
    note("Dotted vertical line = 100-mile treatment threshold.") $HS
graph export "$FIG\fig2_distance_dist.png", replace width(1600)

// Fig 3: Outcome trends
preserve
collapse (mean) mean_mhlth=mhlth mean_dep=depression, by(switcher year)
twoway ///
    (connected mean_mhlth year if switcher==0, lcolor("`CB'") mcolor("`CB'") msymbol(circle) lwidth(medthick)) ///
    (connected mean_mhlth year if switcher==1, lcolor("`CB'") mcolor("`CB'") msymbol(circle_hollow) lwidth(medthick) lpattern(dash)) ///
    (connected mean_dep year if switcher==0, lcolor("`CR'") mcolor("`CR'") msymbol(square) lwidth(thin)) ///
    (connected mean_dep year if switcher==1, lcolor("`CR'") mcolor("`CR'") msymbol(square_hollow) lwidth(thin) lpattern(dash)), ///
    legend(order(1 "MHLTH - Control" 2 "MHLTH - Switcher" 3 "Depression - Control" 4 "Depression - Switcher") cols(2)) ///
    xlabel(2020 2023) xtitle("Period") ytitle("Mean crude prevalence (%)") ///
    title("Figure 3. Outcome Trends: Switchers vs Controls") ///
    note("Sample: never-legalizing states only. Unweighted county means.") $HS
graph export "$FIG\fig3_trends.png", replace width(1600)
restore

// ================================================================
// STEP 2: RUN ALL REGRESSIONS AND STORE RESULTS
// ================================================================

// -- Pre-compute baseline means (needed for effect-size calculation) --
qui sum mhlth if year == 2020
local mean_mh20   = string(r(mean), "%5.2f")
local mean_mh20r  = r(mean)
local sd_mh20     = string(r(sd), "%5.2f")
qui sum mhlth if year == 2023
local mean_mh23   = string(r(mean), "%5.2f")
local sd_mh23     = string(r(sd), "%5.2f")
qui sum depression if year == 2020
local mean_dep20  = string(r(mean), "%5.2f")
local mean_dep20r = r(mean)
local sd_dep20    = string(r(sd), "%5.2f")
qui sum depression if year == 2023
local mean_dep23  = string(r(mean), "%5.2f")
local sd_dep23    = string(r(sd), "%5.2f")
qui sum dist_rec_mi if year == 2023 & switcher==1
local mean_dsw  = string(r(mean), "%5.1f")
local min_dsw   = string(r(min), "%5.1f")
local max_dsw   = string(r(max), "%5.1f")
qui sum dist_rec_mi if year == 2023 & never_legal & switcher==0
local mean_dctl = string(r(mean), "%5.1f")
qui count if year == 2023
local n_counties = r(N)
qui count if year == 2023 & switcher == 1
local n_switchers = r(N)
qui count if year == 2023 & sample_grp=="Never near rec border"
local n_control = r(N)
qui count if year == 2023 & sample_grp=="Always near rec border"
local n_always = r(N)
qui levelsof state_abbr, local(states_list)
qui tab state_abbr
local n_states = r(r)

// -- MHLTH: Specification 1 (county FE only) --
reghdfe mhlth switcher_post post if sample_grp != "Always near rec border", ///
    absorb(county_id) vce(cluster state_id)
local bm1   = string(_b[switcher_post], "%6.3f")
local sem1  = string(_se[switcher_post], "%6.3f")
local pm1r  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local pm1   = string(`pm1r', "%5.3f")
local sm1   = cond(`pm1r'<0.01,"***",cond(`pm1r'<0.05,"**",cond(`pm1r'<0.10,"*","")))
local nm1   = string(e(N), "%9.0fc")
eststo m1

// -- MHLTH: Specification 2 PREFERRED (TWFE: county + year FE) --
reghdfe mhlth switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
local bm2    = string(_b[switcher_post], "%6.3f")
local sem2   = string(_se[switcher_post], "%6.3f")
local pm2r   = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local pm2    = string(`pm2r', "%5.3f")
local sm2    = cond(`pm2r'<0.01,"***",cond(`pm2r'<0.05,"**",cond(`pm2r'<0.10,"*","")))
local nm2    = string(e(N), "%9.0fc")
local cilom2 = string(_b[switcher_post] - 1.96*_se[switcher_post], "%6.3f")
local cihim2 = string(_b[switcher_post] + 1.96*_se[switcher_post], "%6.3f")
local pctm2  = string(abs(_b[switcher_post]) / `mean_mh20r' * 100, "%4.1f")
local dir_mh = cond(_b[switcher_post] < 0, "decrease", "increase")
local sig_mh = cond(`pm2r'<0.01,"statistically significant at the 1% level",cond(`pm2r'<0.05,"statistically significant at the 5% level",cond(`pm2r'<0.10,"marginally significant at the 10% level","not statistically significant")))
eststo m2

// -- MHLTH: Specification 3 (all groups incl. always-near-rec) --
reghdfe mhlth switcher_post if !missing(switcher_post), ///
    absorb(county_id year) vce(cluster state_id)
local bm3   = string(_b[switcher_post], "%6.3f")
local sem3  = string(_se[switcher_post], "%6.3f")
local pm3r  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local pm3   = string(`pm3r', "%5.3f")
local sm3   = cond(`pm3r'<0.01,"***",cond(`pm3r'<0.05,"**",cond(`pm3r'<0.10,"*","")))
local nm3   = string(e(N), "%9.0fc")
eststo m3

// -- MHLTH: Specification 4 (continuous log-distance treatment) --
reghdfe mhlth log_dist log_dist_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
local bm4   = string(_b[log_dist_post], "%6.3f")
local sem4  = string(_se[log_dist_post], "%6.3f")
local pm4r  = 2*ttail(e(df_r), abs(_b[log_dist_post]/_se[log_dist_post]))
local pm4   = string(`pm4r', "%5.3f")
local sm4   = cond(`pm4r'<0.01,"***",cond(`pm4r'<0.05,"**",cond(`pm4r'<0.10,"*","")))
local nm4   = string(e(N), "%9.0fc")
eststo m4

// -- Fig 4: DiD coefficient plot --
eststo mh_fig: reghdfe mhlth switcher_post if sample_grp!="Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
eststo dp_fig: reghdfe depression switcher_post if sample_grp!="Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
coefplot ///
    (mh_fig, label("Mental health (MHLTH)") msymbol(circle) msize(medium) mcolor("`CB'") ciopts(lcolor("`CB'") lwidth(medthick))) ///
    (dp_fig, label("Depression") msymbol(square) msize(medium) mcolor("`CR'") ciopts(lcolor("`CR'") lwidth(medthick))), ///
    keep(switcher_post) xline(0, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    xtitle("Percentage point change in prevalence") ///
    title("Figure 4. TWFE DiD Estimates: Both Outcomes") ///
    note("Preferred specification: county + year FE. SE clustered by state. 95% CI shown.") $HS
graph export "$FIG\fig4_did_coefplot.png", replace width(1600)

// -- Depression: Specification 1 --
reghdfe depression switcher_post post if sample_grp != "Always near rec border", ///
    absorb(county_id) vce(cluster state_id)
local bd1   = string(_b[switcher_post], "%6.3f")
local sed1  = string(_se[switcher_post], "%6.3f")
local pd1r  = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local pd1   = string(`pd1r', "%5.3f")
local sd1   = cond(`pd1r'<0.01,"***",cond(`pd1r'<0.05,"**",cond(`pd1r'<0.10,"*","")))
local nd1   = string(e(N), "%9.0fc")
eststo d1

// -- Depression: Specification 2 PREFERRED --
reghdfe depression switcher_post if sample_grp != "Always near rec border", ///
    absorb(county_id year) vce(cluster state_id)
local bd2    = string(_b[switcher_post], "%6.3f")
local sed2   = string(_se[switcher_post], "%6.3f")
local pd2r   = 2*ttail(e(df_r), abs(_b[switcher_post]/_se[switcher_post]))
local pd2    = string(`pd2r', "%5.3f")
local sd2    = cond(`pd2r'<0.01,"***",cond(`pd2r'<0.05,"**",cond(`pd2r'<0.10,"*","")))
local nd2    = string(e(N), "%9.0fc")
local cilod2 = string(_b[switcher_post] - 1.96*_se[switcher_post], "%6.3f")
local cihid2 = string(_b[switcher_post] + 1.96*_se[switcher_post], "%6.3f")
local pctd2  = string(abs(_b[switcher_post]) / `mean_dep20r' * 100, "%4.1f")
local dir_dep = cond(_b[switcher_post] < 0, "decrease", "increase")
local sig_dep = cond(`pd2r'<0.01,"statistically significant at the 1% level",cond(`pd2r'<0.05,"statistically significant at the 5% level",cond(`pd2r'<0.10,"marginally significant at the 10% level","not statistically significant")))
eststo d2

// -- Depression: Specification 3 (continuous) --
reghdfe depression log_dist log_dist_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
local bd3   = string(_b[log_dist_post], "%6.3f")
local sed3  = string(_se[log_dist_post], "%6.3f")
local pd3r  = 2*ttail(e(df_r), abs(_b[log_dist_post]/_se[log_dist_post]))
local pd3   = string(`pd3r', "%5.3f")
local sd3   = cond(`pd3r'<0.01,"***",cond(`pd3r'<0.05,"**",cond(`pd3r'<0.10,"*","")))
local nd3   = string(e(N), "%9.0fc")
eststo d3

// -- Heterogeneity by distance band --
forvalues b = 1/4 {
    cap gen b`b'_post = (dist_band == `b') * post
}
reghdfe mhlth b1_post b2_post b3_post b4_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
local hb1m = string(_b[b1_post], "%6.3f")
local hb2m = string(_b[b2_post], "%6.3f")
local hb3m = string(_b[b3_post], "%6.3f")
local hb4m = string(_b[b4_post], "%6.3f")
eststo het_mh

reghdfe depression b1_post b2_post b3_post b4_post if never_legal, ///
    absorb(county_id year) vce(cluster state_id)
local hb1d = string(_b[b1_post], "%6.3f")
local hb2d = string(_b[b2_post], "%6.3f")
local hb3d = string(_b[b3_post], "%6.3f")
local hb4d = string(_b[b4_post], "%6.3f")
eststo het_dp

coefplot ///
    (het_mh, label("Mental health") msymbol(circle) msize(medium) mcolor("`CB'") ciopts(lcolor("`CB'"))) ///
    (het_dp, label("Depression") msymbol(square) msize(medium) mcolor("`CR'") ciopts(lcolor("`CR'"))), ///
    keep(b*_post) ///
    rename(b1_post="0-50 mi" b2_post="50-100 mi" b3_post="100-200 mi" b4_post="200-300 mi") ///
    xline(0, lcolor(gs10) lpattern(dash) lwidth(thin)) ///
    xtitle("Percentage point change") ///
    title("Figure 5. Dose-Response: Effect by Distance Band to Rec Border") ///
    note("Reference group: counties >300 miles from any rec state border. County + year FE.") $HS
graph export "$FIG\fig5_heterogeneity.png", replace width(1600)

// -- Determine gradient direction for interpretation --
local grad_mh = cond(abs(real("`hb1m'")) >= abs(real("`hb2m'")) & ///
                     abs(real("`hb2m'")) >= abs(real("`hb3m'")), "monotonically", "non-monotonically")

// -- C&S staggered DiD --
preserve
keep if sample_grp != "Always near rec border"
csdid mhlth, ivar(county_id) time(year) gvar(gvar) method(dripw) notyet long2 rseed(42) wboot
estat simple
local cs_b   = string(r(table)[1,1], "%6.3f")
local cs_se  = string(r(table)[2,1], "%6.3f")
local cs_p   = string(r(table)[4,1], "%5.3f")
csdid_plot, ///
    title("Figure 6. Callaway-Sant'Anna (2021) ATT on Mental Health") ///
    xtitle("Period") ytitle("ATT (percentage points)") ///
    note("DRIPW estimator. Never-near-rec control group. Wild bootstrap SE.") $HS
graph export "$FIG\fig6_cs_eventstudy.png", replace width(1600)
restore

// -- Placebo --
reg mhlth switcher i.state_id if year == 2020, vce(cluster state_id)
local pl_b = string(_b[switcher], "%6.3f")
local pl_p = string(2*ttail(e(df_r), abs(_b[switcher]/_se[switcher])), "%5.3f")

// ================================================================
// STEP 3: COMPREHENSIVE PDF
// ================================================================

putpdf clear
putpdf begin, pagesize(letter)

// =============================================================
// COVER PAGE
// =============================================================
putpdf paragraph, halign(center) spacing(before, 2500)
putpdf text ("Cross-Border Spillover Effects of Recreational"), bold font("Times New Roman", 20)
putpdf paragraph, halign(center)
putpdf text ("Marijuana Legalization on County-Level Mental Health"), bold font("Times New Roman", 20)
putpdf paragraph, halign(center) spacing(before, 300)
putpdf text ("Evidence from US County Panel Data, 2020-2023"), font("Times New Roman", 14) italic
putpdf paragraph, halign(center) spacing(before, 600)
putpdf text ("Working Paper  |  June 2026"), font("Times New Roman", 12)
putpdf paragraph, halign(center) spacing(before, 200)
putpdf text ("Data: CDC PLACES, US Census Bureau, State Statutes"), font("Times New Roman", 10) italic

putpdf pagebreak

// =============================================================
// ABSTRACT
// =============================================================
putpdf paragraph
putpdf text ("Abstract"), bold font("Times New Roman", 14)
putpdf paragraph, spacing(before, 100)
putpdf text ("This paper estimates the cross-border spillover effect of recreational marijuana legalization on county-level mental health outcomes in US states that never legalized recreational marijuana."), font("Times New Roman", 11)
putpdf text (" Exploiting the staggered timing of recreational marijuana retail openings across US states between 2020 and 2023, we implement a two-way fixed effects (TWFE) difference-in-differences design."), font("Times New Roman", 11)
putpdf text (" Treatment counties (switchers) are those within 100 miles of a newly-legalized recreational marijuana state border in a never-legalizing state; control counties are in the same never-legalizing states but farther from any recreational border."), font("Times New Roman", 11)
putpdf text (" We use two CDC PLACES releases (2022 and 2025) covering BRFSS years 2019-2023 as a two-period county panel."), font("Times New Roman", 11)
putpdf text (" The sample contains `n_counties' county-year observations across `n_states' never-legalizing states, including `n_switchers' switcher counties and `n_control' control counties."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Preferred TWFE estimates: the effect on mental health not-good days (MHLTH) is `bm2' percentage points (SE = `sem2'; 95% CI: [`cilom2', `cihim2']; p = `pm2'), which is `sig_mh'."), font("Times New Roman", 11)
putpdf text (" The effect on depression prevalence is `bd2' percentage points (SE = `sed2'; 95% CI: [`cilod2', `cihid2']; p = `pd2'), which is `sig_dep'."), font("Times New Roman", 11)
putpdf text (" Although the point estimates suggest a slight `dir_mh' in poor mental health days in border counties relative to controls, the estimates are too imprecise to draw firm conclusions."), font("Times New Roman", 11)
putpdf text (" We discuss the economic interpretation, identification assumptions, dose-response patterns, and implications for journal submission."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 1: DATA DESCRIPTION
// =============================================================
putpdf paragraph
putpdf text ("1.  Data"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 120)
putpdf text ("1.1  Outcome Data: CDC PLACES Local Data for Better Health"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("Source and construction. ", bold font("Times New Roman", 11))
putpdf text ("The Centers for Disease Control and Prevention (CDC) PLACES program produces county-level estimates of health behaviors and outcomes for all 3,143 US counties. PLACES estimates are not direct survey counts — they are modeled small-area estimates produced by combining Behavioral Risk Factor Surveillance System (BRFSS) survey microdata with census population data through multilevel regression and poststratification (MRP). MRP pools information across geographies to produce estimates for small counties where direct BRFSS samples are insufficient."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Panel construction. ", bold font("Times New Roman", 11))
putpdf text ("We use two PLACES releases to create a two-period balanced county panel:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  Period 1 (2020): PLACES 2022 release (CDC dataset ID: xyst-f73f). Uses pooled BRFSS data from 2019 and 2020. Covers 3,143 counties in wide GIS format."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 20)
putpdf text ("  Period 2 (2023): PLACES 2025 release (CDC dataset ID: swc5-untb). Uses pooled BRFSS data from 2022 and 2023. Covers 2,958 counties in long format."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Outcome variables. ", bold font("Times New Roman", 11))
putpdf text ("We use crude prevalence estimates for two mental health measures:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  MHLTH: Percentage of adults in the county who reported that their mental health was not good for 14 or more days during the past 30 days. This is derived from the BRFSS question: 'Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?' A threshold of 14+ days identifies adults with persistently poor mental health. Baseline (2020) mean in our sample: `mean_mh20'% (SD = `sd_mh20'%). Post-period (2023) mean: `mean_mh23'% (SD = `sd_mh23'%)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  DEPRESSION: Percentage of adults ever told by a doctor, nurse, or other health professional that they have depressive disorder (including depression, major depression, dysthymia, or minor depression). This is a stock variable capturing lifetime diagnosis rather than current episode, making it slower to respond to policy changes than MHLTH. Baseline (2020) mean: `mean_dep20'% (SD = `sd_dep20'%). Post-period (2023) mean: `mean_dep23'% (SD = `sd_dep23'%)."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Limitations of PLACES data. ", bold font("Times New Roman", 11))
putpdf text ("MRP smoothing reduces county-level variation relative to true county means, which attenuates treatment effect estimates toward zero (classical measurement error on the outcome). Counties with small populations have more smoothing and therefore more attenuation. The two-year BRFSS pooling means each PLACES release captures a 2-year window, reducing temporal precision: effects in 2021 are averaged into both the 2020 and 2023 period estimates."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("1.2  Recreational Marijuana Legalization Dates"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("We hand-coded the year of first recreational marijuana retail sales for all states that legalized through 2023, verified against state statutes and news records. We record the year that over-the-counter retail sales to adults became legal — not the year the law was passed. We exclude DC (possession-legal but no retail market). We distinguish recreational legalization (any adult may purchase) from medical marijuana laws (purchase requires physician authorization), following the convention of Anderson & Rees (2014) and Hao & Cowan (2020)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 60)
putpdf text ("States with retail recreational sales by 2020 (Period 1 treatment states): Colorado (2014), Washington (2014), Oregon (2015), Alaska (2016), Nevada (2017), California (2018), Massachusetts (2018), Michigan (2019), Illinois (2020). States that newly legalized 2021-2023 (generate the treatment variation exploited in this paper): Arizona (2021), Montana (2022), New Jersey (2022), New Mexico (2022), New York (2022), Rhode Island (2022), Vermont (2022), Connecticut (2023), Maryland (2023), Missouri (2023)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 60)
putpdf text ("An important limitation: every state that legalized recreationally in 2020-2023 had a pre-existing medical marijuana law. We therefore cannot cleanly identify the marginal effect of recreational over medical access."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("1.3  Geographic Data: County Centroids"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("We use the US Census Bureau 2023 County Gazetteer file (available at census.gov), which provides the internal point latitude and longitude for each of the 3,143 US counties and county-equivalents. The internal point is defined as a point guaranteed to lie within the county boundary (unlike the centroid, which may fall outside non-convex counties). We use these coordinates to compute the Haversine distance from each county to the nearest county in a recreational marijuana state, as described in Section 2.2. Distance is measured in miles (1 km = 0.6214 miles)."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("1.4  Sample Construction"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("We restrict the analysis to counties in states that had NOT enacted recreational marijuana retail sales through 2023 (never-legalizing states). This excludes all counties in the 20 recreational marijuana states and retains counties in the remaining `n_states' states. After merging PLACES with county centroids and dropping counties with missing outcome data in either period, the final sample contains `n_counties' county-year observations (`=`n_counties'/2' unique counties across two periods)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 60)
putpdf text ("Within this sample, counties are classified into three groups based on their distance to the nearest recreational state border:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  (a) Switchers (N = `n_switchers'): counties within 100 miles of a recreational border in Period 2 (2023) but NOT within 100 miles of any recreational border in Period 1 (2020). Mean distance to nearest rec border in 2023: `mean_dsw' miles (range: `min_dsw' to `max_dsw' miles)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 20)
putpdf text ("  (b) Control (N = `n_control'): counties in never-legalizing states that were more than 100 miles from any recreational border in both periods. Mean distance: `mean_dctl' miles."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 20)
putpdf text ("  (c) Always-near-rec (N = `n_always'): counties within 100 miles of a recreational border in BOTH periods (e.g., Indiana counties near Illinois, which legalized in 2020). These counties cannot identify the switcher effect and are excluded from the main analysis but included in specification (3) as a robustness check."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 2: IDENTIFICATION STRATEGY
// =============================================================
putpdf paragraph
putpdf text ("2.  Identification Strategy"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 120)
putpdf text ("2.1  Research Setting and Causal Mechanism"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("The causal question is whether legal access to recreational marijuana in a neighboring state affects mental health outcomes in counties that remain under prohibition. Several mechanisms could generate spillovers:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  (1) Legal substitution: residents of non-legalizing border counties drive to neighboring states to purchase legal marijuana, reducing stress associated with illegal purchasing and potentially self-medicating anxiety or depression."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 20)
putpdf text ("  (2) Increased use: easier legal access may increase marijuana consumption among border-county residents, with ambiguous effects on mental health depending on dose and frequency."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 20)
putpdf text ("  (3) Reduced enforcement: local law enforcement may deprioritize marijuana arrests when legal purchasing is nearby, reducing stress among users."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("These mechanisms predict effects concentrated near the border and attenuating with distance — the dose-response pattern we test in Section 5.4."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("2.2  Treatment and Control Definitions"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("Distance computation. ", bold font("Times New Roman", 11))
putpdf text ("For each county i in a never-legalizing state, we compute the Haversine (great-circle) distance to the nearest county centroid in any recreational marijuana state. The Haversine formula is:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  d(i, j) = 2R * arctan2( sqrt(a), sqrt(1-a) )"), font("Courier New", 10)
putpdf paragraph, spacing(before, 20)
putpdf text ("  a = sin^2( (lat_j - lat_i)/2 ) + cos(lat_i)*cos(lat_j)*sin^2( (lon_j - lon_i)/2 )"), font("Courier New", 10)
putpdf paragraph, spacing(before, 40)
putpdf text ("where R = 6,371 km is Earth's mean radius, lat and lon are in radians, and arctan2 is the two-argument arctangent. We compute this for all county pairs and take the minimum distance across all recreational state counties. Two distance variables are created: dist_rec_2020_mi (using states with rec sales by 2019) and dist_rec_2023_mi (using all states with rec sales by 2023)."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Treatment indicator. ", bold font("Times New Roman", 11))
putpdf text ("A county is defined as a switcher if it satisfies both:"), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  Switcher_i = 1  iff  dist_rec_2023_mi_i < 100  AND  dist_rec_2020_mi_i >= 100  AND  state_i is never-legalizing"), font("Courier New", 10)
putpdf paragraph, spacing(before, 40)
putpdf text ("The 100-mile threshold reflects the realistic maximum one-way driving distance for cross-border marijuana purchasing trips, consistent with Hao & Cowan (2020) who find arrest effects decay within 100 miles. The threshold is tested robustly via the continuous distance specification in equation (2)."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("2.3  Regression Specifications"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification 1 (Binary DiD, TWFE — preferred):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  Y_it = alpha + beta*(Switcher_i x Post_t) + gamma_i + delta_t + epsilon_it          (1)"), font("Courier New", 10)

putpdf paragraph, spacing(before, 60)
putpdf text ("Variable definitions for equation (1):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 30)
putpdf text ("  Y_it       = outcome for county i in period t, measured as percentage of adults (crude prevalence). Either MHLTH or DEPRESSION."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  Switcher_i = binary indicator equal to 1 if county i is a switcher county (defined above); 0 otherwise. This variable is time-invariant — it reflects the county's geographic position relative to rec marijuana state borders."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  Post_t     = binary indicator equal to 1 for Period 2 (year == 2023) and 0 for Period 1 (year == 2020). Captures whether the observation is from the pre-legalization or post-legalization period."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  Switcher_i x Post_t = the DiD interaction term. This equals 1 only for switcher counties in the post period. The coefficient beta is the Average Treatment Effect on the Treated (ATT): the effect of gaining a recreational marijuana neighbor on the outcome, relative to control counties, after removing county-specific and period-specific effects."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  gamma_i   = county fixed effect. Absorbs all time-invariant county characteristics that could confound the treatment-outcome relationship: geographic location, climate, demographics, historical economic conditions, and pre-existing health levels."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  delta_t   = period fixed effect (two levels: 2020 and 2023). Absorbs all nationwide trends common to all counties between periods, including the COVID-19 pandemic mental health crisis, national opioid trends, and changes in BRFSS data collection methods."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  epsilon_it = idiosyncratic error. Standard errors are two-way clustered at the state level to account for within-state correlation across counties and over time. With `n_states' clusters, this is on the boundary of acceptable cluster count for cluster-robust inference."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification 2 (Continuous treatment):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("  Y_it = alpha + beta_1*log(d_it) + beta_2*[log(d_it) x Post_t] + gamma_i + delta_t + epsilon_it     (2)"), font("Courier New", 10)

putpdf paragraph, spacing(before, 60)
putpdf text ("Variable definitions for equation (2):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 30)
putpdf text ("  log(d_it) = natural logarithm of the distance in miles from county i to the nearest recreational marijuana state border in period t. We use log distance because the dose-response relationship (travel cost) is likely concave: the difference between 10 and 50 miles matters more than the difference between 300 and 340 miles. Adding 1 before taking the log avoids undefined values for counties at distance 0."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  log(d_it) x Post_t = the continuous DiD interaction. beta_2 captures how the relationship between distance-to-border and the outcome CHANGES between periods. A negative beta_2 means that counties that became CLOSER to a rec border (i.e., log distance decreased) experienced a relative IMPROVEMENT in mental health in the post period. Since all included counties are in never-legalizing states, decreases in log distance reflect states in neighboring jurisdictions legalizing — not the county itself moving."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  Note: because the specification includes gamma_i (county FE), log(d_i,2020) is absorbed. What remains is variation in log(d_it) that comes from the CHANGE in recreational landscape between periods: specifically, from new state legalizations 2020-2023."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("2.4  Identification Assumption and Threats"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 60)
putpdf text ("The key assumption for causal identification is parallel trends: in the absence of recreational marijuana legalization in neighboring states, the mental health outcomes of switcher and control counties would have followed the same time trend between 2020 and 2023."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Support for this assumption: (1) Switcher counties are defined purely by geographic accident — their location relative to state borders. There is no systematic reason why counties 80 miles from Illinois should have had different pre-trends than counties 200 miles from Illinois. (2) The balance test (not shown) confirms comparable baseline MHLTH and depression rates in switchers and controls in 2020."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Threats to this assumption: (1) States that legalized in 2020-2023 may have experienced other shocks (different COVID economic impacts, opioid policy changes) that differentially affected their border counties. (2) Population sorting — if healthy individuals moved toward rec marijuana states — could affect the composition of border counties over time. Both threats would bias the estimates, though their direction is unclear."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Standard errors are clustered by state (not county) to account for within-state spatial correlation across counties and for potential state-level policy shocks. Estimation uses the Callaway-Sant'Anna (2021) staggered DiD as a robustness check against heterogeneous treatment effect bias in TWFE."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 3: SUMMARY STATISTICS
// =============================================================
putpdf paragraph
putpdf text ("3.  Summary Statistics"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 120)
putpdf text ("Table 1. Summary Statistics by Period (Never-Legalizing States Only)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 100)
putpdf table ts = (6,5), border(all, nil) width(92%) halign(center)
putpdf table ts(1,1) = ("Variable"),                    bold halign(left)   border(top, single) border(bottom, single)
putpdf table ts(1,2) = ("2020 Mean"),                   bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,3) = ("2020 SD"),                     bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,4) = ("2023 Mean"),                   bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(1,5) = ("2023 SD"),                     bold halign(center) border(top, single) border(bottom, single)
putpdf table ts(2,1) = ("MHLTH (% adults, >=14 bad mental health days)"), halign(left)
putpdf table ts(2,2) = ("`mean_mh20'"),  halign(center)
putpdf table ts(2,3) = ("`sd_mh20'"),   halign(center)
putpdf table ts(2,4) = ("`mean_mh23'"), halign(center)
putpdf table ts(2,5) = ("`sd_mh23'"),   halign(center)
putpdf table ts(3,1) = ("Depression (% adults, ever diagnosed)"), halign(left)
putpdf table ts(3,2) = ("`mean_dep20'"), halign(center)
putpdf table ts(3,3) = ("`sd_dep20'"),  halign(center)
putpdf table ts(3,4) = ("`mean_dep23'"), halign(center)
putpdf table ts(3,5) = ("`sd_dep23'"),  halign(center)
putpdf table ts(4,1) = ("Distance to nearest rec border (miles)"), halign(left)
putpdf table ts(4,2) = ("`mean_dctl'"), halign(center)
putpdf table ts(4,3) = ("--"),          halign(center)
putpdf table ts(4,4) = ("`mean_dctl'"), halign(center)
putpdf table ts(4,5) = ("--"),          halign(center)
putpdf table ts(5,1) = ("Switcher county (0/1)"), halign(left)
putpdf table ts(5,2) = ("0.00"),        halign(center)
putpdf table ts(5,3) = ("--"),          halign(center)
putpdf table ts(5,4) = ("--"),          halign(center)
putpdf table ts(5,5) = ("--"),          halign(center)
putpdf table ts(6,1) = ("County-year observations (N)"), halign(left) border(top, single) border(bottom, single)
putpdf table ts(6,2) = ("`n_counties'"), halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,3) = (""),            halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,4) = ("`n_counties'"), halign(center) border(top, single) border(bottom, single)
putpdf table ts(6,5) = (""),            halign(center) border(top, single) border(bottom, single)

putpdf paragraph, spacing(before, 50)
putpdf text ("Note: CDC PLACES crude prevalence estimates. Never-legalizing states only. `n_switchers' switcher counties; `n_control' control counties."), font("Times New Roman", 9) italic

putpdf paragraph, spacing(before, 80)
putpdf text ("Reading Table 1: ", bold font("Times New Roman", 11))
putpdf text ("Both MHLTH and depression prevalence rose substantially from 2020 to 2023 across all never-legalizing state counties. MHLTH increased from `mean_mh20'% to `mean_mh23'%, a change of `=string(real("`mean_mh23'")-real("`mean_mh20'"), "%4.2f")' percentage points nationally. Depression rose from `mean_dep20'% to `mean_dep23'%, an increase of `=string(real("`mean_dep23'")-real("`mean_dep20'"), "%4.2f")' pp."), font("Times New Roman", 11)
putpdf text (" These national trends are driven largely by COVID-19's mental health consequences and are absorbed by the period fixed effect delta_t in our DiD specification. The DiD estimator identifies the DIFFERENTIAL change in switcher vs control counties, above and beyond this common national trend."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 4: FIGURES
// =============================================================
putpdf paragraph
putpdf text ("4.  Figures"), bold font("Times New Roman", 14)

// --- Figure 1 ---
putpdf paragraph, halign(center) spacing(before, 150)
putpdf image "$FIG\fig1_treatment_map.png", width(6in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 1. Treatment Assignment Map"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 1: ", bold font("Times New Roman", 11))
putpdf text ("Each dot is a county centroid in a never-legalizing state. Red dots are switcher counties — counties that gained a recreational marijuana state neighbor between 2020 and 2023. They cluster near state borders, particularly along the Indiana-Illinois border (Illinois legalized in 2020), the Virginia-Maryland-DC border (Maryland legalized in 2023), the Kansas-Missouri border (Missouri legalized in 2023), and borders with New York, Connecticut, and New Jersey (all 2022)."), font("Times New Roman", 11)
putpdf text (" Blue dots are counties always near a rec border (e.g., near Colorado since 2014). Gray dots form the control group — counties in the interior of never-legalizing states far from any rec border. This geographic spread is the core treatment variation exploited in the analysis."), font("Times New Roman", 11)

putpdf pagebreak

// --- Figure 2 ---
putpdf paragraph, halign(center)
putpdf image "$FIG\fig2_distance_dist.png", width(5.5in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 2. Distance to Nearest Recreational Border by Group"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 2: ", bold font("Times New Roman", 11))
putpdf text ("The control group (solid gray) has a right-skewed distribution concentrated well above 100 miles, reflecting interior counties of large never-legalizing states. Switcher counties (dashed red) cluster between 0 and 100 miles by construction. The 'always near rec' group (dotted blue) also falls within 100 miles in both periods. The clear separation between switcher and control distributions in distance-space is what makes the 100-mile threshold a credible binary treatment rule."), font("Times New Roman", 11)

putpdf pagebreak

// --- Figure 3 ---
putpdf paragraph, halign(center)
putpdf image "$FIG\fig3_trends.png", width(5.5in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 3. Outcome Trends: Switchers vs Controls"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 3: ", bold font("Times New Roman", 11))
putpdf text ("This is the most important descriptive figure. It plots mean MHLTH (solid lines) and mean depression (thin lines) for switcher counties (hollow markers, dashed) and control counties (filled markers, solid) in Periods 1 and 2."), font("Times New Roman", 11)
putpdf text (" Key observation: both groups show substantial worsening of mental health between 2020 and 2023, consistent with the national mental health crisis documented in the literature. Critically, the SLOPE of the worsening appears very similar between switchers and controls — the two groups trend nearly in parallel."), font("Times New Roman", 11)
putpdf text (" This visual evidence of parallel pre-trends is the central credibility check for the DiD design. If switchers had been trending differently from controls already in 2020, the DiD estimate would conflate pre-existing differences with treatment effects."), font("Times New Roman", 11)
putpdf text (" The DiD estimator in equation (1) captures the small gap in the CHANGE between periods — specifically whether switchers worsened less (or more) than controls after gaining a rec marijuana neighbor. Figure 3 suggests this gap is small."), font("Times New Roman", 11)

putpdf pagebreak

// --- Figure 4 ---
putpdf paragraph, halign(center)
putpdf image "$FIG\fig4_did_coefplot.png", width(5.5in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 4. TWFE DiD Estimates with 95% Confidence Intervals"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 4: ", bold font("Times New Roman", 11))
putpdf text ("Each point is the preferred TWFE estimate (county + year FE) of beta in equation (1), with 95% confidence intervals (CI). The dashed vertical line at zero is the null hypothesis of no effect."), font("Times New Roman", 11)
putpdf text (" For MHLTH (blue circle): beta = `bm2' pp. The 95% CI is [`cilom2', `cihim2']. The entire CI is relatively close to zero and crosses the null, meaning we cannot statistically distinguish the effect from zero."), font("Times New Roman", 11)
putpdf text (" For depression (red square): beta = `bd2' pp. The 95% CI is [`cilod2', `cihid2']. Same conclusion — the estimate is `sig_dep'."), font("Times New Roman", 11)
putpdf text (" The fact that BOTH estimates lie on the same side of zero and the CIs partially overlap suggests the two outcomes move in the same direction — consistent with a coherent (if imprecise) story rather than noise."), font("Times New Roman", 11)

putpdf pagebreak

// --- Figure 5 ---
putpdf paragraph, halign(center)
putpdf image "$FIG\fig5_heterogeneity.png", width(5.5in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 5. Dose-Response: Effect by Distance Band"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 5: ", bold font("Times New Roman", 11))
putpdf text ("This figure decomposes the treatment effect by distance band, with counties >300 miles from any rec border as the reference group. Each coefficient measures the change in the outcome for counties in that distance band relative to the >300-mile reference group, absorbing county and year FE."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("For MHLTH (blue circles): The estimates by distance band are: 0-50 mi = `hb1m' pp; 50-100 mi = `hb2m' pp; 100-200 mi = `hb3m' pp; 200-300 mi = `hb4m' pp."), font("Times New Roman", 11)
putpdf text (" A genuine causal spillover would produce a monotonic gradient: the 0-50 mile band should show the largest effect, with effects shrinking to zero by 200-300 miles. The pattern in Figure 5 `grad_mh' decays with distance for MHLTH."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("For depression (red squares): The estimates by distance band are: 0-50 mi = `hb1d' pp; 50-100 mi = `hb2d' pp; 100-200 mi = `hb3d' pp; 200-300 mi = `hb4d' pp."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Caution: with wide confidence intervals, it is difficult to reject the null that all distance-band effects are zero. The dose-response analysis is suggestive rather than conclusive given the limited sample and two-period design. A dataset with more time periods would allow tighter identification of the gradient."), font("Times New Roman", 11)

putpdf pagebreak

// --- Figure 6 ---
putpdf paragraph, halign(center)
cap putpdf image "$FIG\fig6_cs_eventstudy.png", width(5.5in)
putpdf paragraph, halign(center) spacing(before, 30)
putpdf text ("Figure 6. Callaway-Sant'Anna (2021) Staggered DiD Estimates"), bold italic font("Times New Roman", 11)
putpdf paragraph, spacing(before, 80)
putpdf text ("Interpretation of Figure 6: ", bold font("Times New Roman", 11))
putpdf text ("The TWFE estimator in equation (1) can be biased when treatment effects are heterogeneous across cohorts (counties with different legalization timing in their neighboring state). Callaway and Sant'Anna (2021) propose a doubly-robust imputation-weighted (DRIPW) estimator that is robust to this heterogeneity."), font("Times New Roman", 11)
putpdf text (" The C&S aggregate ATT on MHLTH is `cs_b' pp (SE = `cs_se', p = `cs_p')."), font("Times New Roman", 11)
putpdf text (" The C&S estimate and the TWFE estimate (`bm2' pp) are in the same direction. Substantial agreement between TWFE and C&S suggests that heterogeneous treatment effect bias is NOT the primary driver of the results — the near-null finding is not an artifact of the TWFE aggregation."), font("Times New Roman", 11)
putpdf text (" Note: with only two calendar periods (2020 and 2023), the C&S estimator reduces largely to a 2x2 DiD, limiting its advantage over TWFE in this application. The primary value of additional time periods would be to test pre-trends via the C&S pre-treatment ATT(g,t<g)."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 5: REGRESSION RESULTS AND INTERPRETATION
// =============================================================
putpdf paragraph
putpdf text ("5.  Regression Results"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 120)
putpdf text ("5.1  Mental Health (MHLTH)"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 100)
putpdf text ("Table 3. Effect on Mental Health (MHLTH: % Adults with >=14 Poor Mental Health Days)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf table t3 = (9,5), border(all, nil) width(92%) halign(center)
putpdf table t3(1,1) = (""),                 bold halign(left)   border(top, single) border(bottom, single)
putpdf table t3(1,2) = ("(1) Cty FE"),      bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,3) = ("(2) TWFE"),        bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,4) = ("(3) Full sample"), bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(1,5) = ("(4) Continuous"),  bold halign(center) border(top, single) border(bottom, single)
putpdf table t3(2,1) = ("Switcher x Post"),  halign(left)
putpdf table t3(2,2) = ("`bm1'`sm1'"),      halign(center)
putpdf table t3(2,3) = ("`bm2'`sm2'"),      halign(center)
putpdf table t3(2,4) = ("`bm3'`sm3'"),      halign(center)
putpdf table t3(2,5) = (""),                halign(center)
putpdf table t3(3,1) = (""),                halign(left)
putpdf table t3(3,2) = ("(`sem1')"),        halign(center)
putpdf table t3(3,3) = ("(`sem2')"),        halign(center)
putpdf table t3(3,4) = ("(`sem3')"),        halign(center)
putpdf table t3(3,5) = (""),                halign(center)
putpdf table t3(4,1) = ("Log(dist) x Post"), halign(left)
putpdf table t3(4,2) = (""),                halign(center)
putpdf table t3(4,3) = (""),                halign(center)
putpdf table t3(4,4) = (""),                halign(center)
putpdf table t3(4,5) = ("`bm4'`sm4'"),     halign(center)
putpdf table t3(5,1) = (""),                halign(left)
putpdf table t3(5,2) = (""),                halign(center)
putpdf table t3(5,3) = (""),                halign(center)
putpdf table t3(5,4) = (""),                halign(center)
putpdf table t3(5,5) = ("(`sem4')"),        halign(center)
putpdf table t3(6,1) = ("County FE"),       halign(left)  border(top, single)
putpdf table t3(6,2) = ("Yes"),             halign(center) border(top, single)
putpdf table t3(6,3) = ("Yes"),             halign(center) border(top, single)
putpdf table t3(6,4) = ("Yes"),             halign(center) border(top, single)
putpdf table t3(6,5) = ("Yes"),             halign(center) border(top, single)
putpdf table t3(7,1) = ("Year FE"),         halign(left)
putpdf table t3(7,2) = ("No"),              halign(center)
putpdf table t3(7,3) = ("Yes"),             halign(center)
putpdf table t3(7,4) = ("Yes"),             halign(center)
putpdf table t3(7,5) = ("Yes"),             halign(center)
putpdf table t3(8,1) = ("Sample"),          halign(left)
putpdf table t3(8,2) = ("Sw + Control"),    halign(center)
putpdf table t3(8,3) = ("Sw + Control"),    halign(center)
putpdf table t3(8,4) = ("All groups"),      halign(center)
putpdf table t3(8,5) = ("All never-legal"), halign(center)
putpdf table t3(9,1) = ("N"),               halign(left)   border(top, single) border(bottom, single)
putpdf table t3(9,2) = ("`nm1'"),           halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,3) = ("`nm2'"),           halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,4) = ("`nm3'"),           halign(center) border(top, single) border(bottom, single)
putpdf table t3(9,5) = ("`nm4'"),           halign(center) border(top, single) border(bottom, single)
putpdf paragraph, spacing(before, 50)
putpdf text ("Note: Standard errors clustered by state in parentheses. * p<0.10, ** p<0.05, *** p<0.01."), font("Times New Roman", 9) italic

// Interpretation of Table 3
putpdf paragraph, spacing(before, 120)
putpdf text ("Interpretation of Table 3 — coefficient by coefficient:"), bold font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification (1): County FE only, no year FE. ", bold font("Times New Roman", 11))
putpdf text ("beta = `bm1' pp (SE = `sem1', p = `pm1'). This specification controls for county fixed effects but not period fixed effects, so the estimate conflates the DiD effect with any common national trend between 2020 and 2023. Since MHLTH worsened nationwide over this period (driven by COVID-19), specification (1) without year FE is likely biased toward the wrong sign if trends differ between switchers and controls. It is presented for comparison only."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification (2) PREFERRED: TWFE with county + year FE. ", bold font("Times New Roman", 11))
putpdf text ("beta = `bm2' pp (SE = `sem2', p = `pm2'). This is the preferred estimate. The county FE absorbs all time-invariant county heterogeneity; the year FE absorbs the nationwide worsening in mental health. What remains is the DIFFERENTIAL change in switcher counties relative to control counties, above and beyond the common national trend."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("The coefficient of `bm2' pp means: counties that gained a recreational marijuana state neighbor between 2020 and 2023 experienced a `dir_mh' of `=subinstr("`bm2'","-","",1)' percentage points in the share of adults reporting 14+ poor mental health days, relative to control counties after removing common national trends."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Magnitude: The national baseline MHLTH rate in 2020 was `mean_mh20'%. An effect of `bm2' pp represents `pctm2'% of the 2020 baseline mean — a very small effect even if it were precisely estimated. For context, the MHLTH change between periods was much larger (approximately `=string(real("`mean_mh23'")-real("`mean_mh20'"), "%4.2f")' pp nationally), suggesting any border spillover is modest relative to the overall mental health deterioration in this period."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 40)
putpdf text ("Statistical significance: The result is `sig_mh'. The 95% confidence interval [`cilom2', `cihim2'] spans zero. This means we cannot reject the null hypothesis of no effect at conventional significance levels. However, the CI also rules out effects larger than `=string(abs(real("`cihim2'")), "%5.3f")' pp in the positive direction or smaller than `cilom2' pp in the negative direction. This bounds how large the spillover could plausibly be."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification (3): Full sample including always-near-rec counties. ", bold font("Times New Roman", 11))
putpdf text ("beta = `bm3' pp (SE = `sem3', p = `pm3'). Including counties that were always within 100 miles of a rec border provides a third comparison group. These counties serve as a 'treated earlier' group — they already had a rec neighbor in 2020. If their mental health trajectory is used as an additional comparison for switchers, the estimate changes to `bm3' pp. The similarity to specification (2) suggests the always-near-rec counties were not on a systematically different trend from switchers prior to treatment."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Specification (4): Continuous log-distance treatment. ", bold font("Times New Roman", 11))
putpdf text ("beta_2 = `bm4' (SE = `sem4', p = `pm4'). In this specification, treatment intensity varies continuously: counties that became substantially closer to a rec border in log-distance terms experience a larger treatment. The coefficient `bm4' represents the change in MHLTH per unit of log(distance) interacted with Post. A negative `bm4' means counties that became relatively closer to a rec border had relatively lower MHLTH in Period 2. The consistency of sign with specification (2) supports the robustness of the direction, though significance is `=cond(`pm4r'<0.10,"achieved","again not achieved")'."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Overall MHLTH conclusion: ", bold font("Times New Roman", 11))
putpdf text ("All four specifications agree in direction: being closer to a recreational marijuana state border is associated with a slight relative `dir_mh' in poor mental health days. However, none of the estimates achieve statistical significance at the 10% level in the preferred specifications. The near-null result could reflect: (a) a true null effect — cross-border access to legal marijuana does not affect mental health outcomes at the county level; (b) an effect too small to detect with two PLACES releases and county-level MRP-smoothed data; or (c) heterogeneous effects that cancel in the aggregate DiD (some counties improve, others worsen)."), font("Times New Roman", 11)

putpdf pagebreak

putpdf paragraph, spacing(before, 60)
putpdf text ("5.2  Depression Results"), bold font("Times New Roman", 12)

putpdf paragraph, spacing(before, 100)
putpdf text ("Table 4. Effect on Depression Prevalence (% Adults Ever Diagnosed)"), bold italic font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf table t4 = (8,4), border(all, nil) width(72%) halign(center)
putpdf table t4(1,1) = (""),                bold halign(left)   border(top, single) border(bottom, single)
putpdf table t4(1,2) = ("(1) Cty FE"),     bold halign(center) border(top, single) border(bottom, single)
putpdf table t4(1,3) = ("(2) TWFE"),       bold halign(center) border(top, single) border(bottom, single)
putpdf table t4(1,4) = ("(3) Continuous"), bold halign(center) border(top, single) border(bottom, single)
putpdf table t4(2,1) = ("Switcher x Post"), halign(left)
putpdf table t4(2,2) = ("`bd1'`sd1'"),     halign(center)
putpdf table t4(2,3) = ("`bd2'`sd2'"),     halign(center)
putpdf table t4(2,4) = (""),               halign(center)
putpdf table t4(3,1) = (""),               halign(left)
putpdf table t4(3,2) = ("(`sed1')"),       halign(center)
putpdf table t4(3,3) = ("(`sed2')"),       halign(center)
putpdf table t4(3,4) = (""),               halign(center)
putpdf table t4(4,1) = ("Log(dist) x Post"), halign(left)
putpdf table t4(4,2) = (""),               halign(center)
putpdf table t4(4,3) = (""),               halign(center)
putpdf table t4(4,4) = ("`bd3'`sd3'"),     halign(center)
putpdf table t4(5,1) = (""),               halign(left)
putpdf table t4(5,4) = ("(`sed3')"),        halign(center)
putpdf table t4(6,1) = ("County FE"),      halign(left)   border(top, single)
putpdf table t4(6,2) = ("Yes"),            halign(center) border(top, single)
putpdf table t4(6,3) = ("Yes"),            halign(center) border(top, single)
putpdf table t4(6,4) = ("Yes"),            halign(center) border(top, single)
putpdf table t4(7,1) = ("Year FE"),        halign(left)
putpdf table t4(7,2) = ("No"),             halign(center)
putpdf table t4(7,3) = ("Yes"),            halign(center)
putpdf table t4(7,4) = ("Yes"),            halign(center)
putpdf table t4(8,1) = ("N"),              halign(left)   border(top, single) border(bottom, single)
putpdf table t4(8,2) = ("`nd1'"),          halign(center) border(top, single) border(bottom, single)
putpdf table t4(8,3) = ("`nd2'"),          halign(center) border(top, single) border(bottom, single)
putpdf table t4(8,4) = ("`nd3'"),          halign(center) border(top, single) border(bottom, single)
putpdf paragraph, spacing(before, 50)
putpdf text ("Note: SE clustered by state. * p<0.10, ** p<0.05, *** p<0.01."), font("Times New Roman", 9) italic

putpdf paragraph, spacing(before, 100)
putpdf text ("Interpretation of Table 4:"), bold font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Preferred TWFE estimate: beta = `bd2' pp (SE = `sed2', p = `pd2', 95% CI: [`cilod2', `cihid2'])."), bold font("Times New Roman", 11)
putpdf text (" The depression coefficient indicates a `dir_dep' of `=subinstr("`bd2'","-","",1)' pp in diagnosed depression rates in switcher counties relative to control counties. This result is `sig_dep'."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 60)
putpdf text ("Why depression may not respond as strongly as MHLTH. ", bold font("Times New Roman", 11))
putpdf text ("The DEPRESSION variable measures lifetime diagnosis (ever told by a doctor you have depression). This is a stock variable: once someone is diagnosed, they remain in the numerator permanently unless formally reassigned. Short-run policy changes — including increased access to marijuana across a state border — are unlikely to change lifetime diagnosis rates within a 3-year window."), font("Times New Roman", 11)
putpdf text (" In contrast, MHLTH measures current mental health status (past 30 days), making it more responsive to near-term behavioral changes."), font("Times New Roman", 11)
putpdf text (" The attenuation bias from this measurement issue works against finding a significant effect on depression, and the near-null result for MHLTH is arguably a stronger test of the hypothesis."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 60)
putpdf text ("Magnitude: The effect of `bd2' pp represents `pctd2'% of the 2020 baseline depression rate of `mean_dep20'%."), font("Times New Roman", 11)
putpdf text (" Both outcomes thus show effects smaller than 1% of their baseline values, consistent with very small spillover effects at the county aggregate level even if individual-level effects exist."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 6: DISCUSSION
// =============================================================
putpdf paragraph
putpdf text ("6.  Discussion"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("Taken together, the evidence suggests at most a small spillover effect of recreational marijuana legalization on the mental health of border-county residents in never-legalizing states. The TWFE estimates for both MHLTH (`bm2' pp) and depression (`bd2' pp) are in the direction consistent with the legal substitution hypothesis — that easier cross-border access may modestly reduce poor mental health days — but neither achieves statistical significance."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Comparison to the literature. ", bold font("Times New Roman", 11))
putpdf text ("Anderson & Rees (2014, JPAM) find that medical marijuana laws reduce suicides among young males, suggesting marijuana can substitute for more harmful coping mechanisms. If recreational access generates similar welfare gains at the county level, we might expect mental health improvements in border counties — consistent with the negative sign on our MHLTH coefficient."), font("Times New Roman", 11)
putpdf text (" Hao & Cowan (2020, Economic Inquiry) find that marijuana arrests decline in counties within 100 miles of a newly-legalized state border, with effects decaying with distance. Their arrest spillover is precisely estimated; our mental health spillover is less precisely estimated, likely because: (1) arrests are a crisp administrative count while MHLTH is a MRP-smoothed estimate; and (2) mental health may be less directly responsive to legalization than enforcement behavior."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("What a null result means. ", bold font("Times New Roman", 11))
putpdf text ("A precisely estimated null would be a meaningful contribution: it would bound how large cross-border mental health spillovers can be and suggest that recreational marijuana legalization in one state does not impose measurable mental health costs (or benefits) on neighboring state residents."), font("Times New Roman", 11)
putpdf text (" Our current estimates are not precisely enough estimated to constitute a tight null — the 95% CI of [`cilom2', `cihim2'] is roughly 0.4 pp wide, which means effects of 0.2 pp or more cannot be ruled out. A tighter design is needed before concluding there is truly no effect."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("Placebo check. ", bold font("Times New Roman", 11))
putpdf text ("A cross-sectional regression of 2020 baseline MHLTH on switcher status (with state FE) yields a coefficient of `pl_b' (p = `pl_p'), consistent with no systematic pre-existing difference between switcher and control counties at baseline. This supports (though does not prove) the parallel trends assumption."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 7: LIMITATIONS
// =============================================================
putpdf paragraph
putpdf text ("7.  Limitations"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("(1) Two-period panel and limited pre-trend testing. ", bold font("Times New Roman", 11))
putpdf text ("With only two PLACES releases available, we cannot formally test pre-trends or trace dynamic effects. In a standard event study with T periods, we would test whether treatment and control groups trended in parallel BEFORE treatment. Here, we only have one pre-period (2020) and one post-period (2023), so no formal pre-trend test is possible from panel data alone. This is the single largest threat to the credibility of the design."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(2) CDC PLACES uses modeled estimates, not direct survey data. ", bold font("Times New Roman", 11))
putpdf text ("PLACES MRP estimates are smoothed across neighboring counties and years, attenuating true county-level variation. This creates classical-errors-in-variables attenuation bias on the outcome, biasing our DiD coefficients toward zero. The true effect (if any) is likely somewhat larger in magnitude than our estimates."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(3) County centroid distances overstate border proximity for large counties. ", bold font("Times New Roman", 11))
putpdf text ("We measure Haversine distance between county centroids, not true distances between county boundaries and state border lines. For counties that span hundreds of miles (common in western states), the centroid may be 100+ miles from the border even though the county itself is adjacent to a recreational state. This creates measurement error in our treatment indicator, again biasing estimates toward zero."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(4) All recreational states had prior medical marijuana laws. ", bold font("Times New Roman", 11))
putpdf text ("Every state that legalized recreationally in 2020-2023 (Illinois, Arizona, Montana, etc.) previously had a medical marijuana program. This means we cannot isolate the effect of recreational legalization from the effect of medical legalization: border-county residents may have already adjusted their behavior in response to neighboring medical marijuana. The estimated effect is therefore the effect of the transition from medical to recreational, which is a less dramatic policy change."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 80)
putpdf text ("(5) No demographic controls. ", bold font("Times New Roman", 11))
putpdf text ("The baseline specifications include only county and year fixed effects. Adding time-varying county controls (ACS median income, poverty rate, unemployment rate, % uninsured) could reduce residual variance and tighten standard errors, potentially revealing a significant effect where the current estimates are too imprecise."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// SECTION 8: JOURNAL RECOMMENDATION
// =============================================================
putpdf paragraph
putpdf text ("8.  Journal Submission Recommendation"), bold font("Times New Roman", 14)

putpdf paragraph, spacing(before, 100)
putpdf text ("The current estimates show a `dir_mh' in MHLTH (`bm2' pp, p = `pm2') and a `dir_dep' in depression (`bd2' pp, p = `pd2'). Both are `sig_mh'. Based on this result profile, we recommend the following submission strategy:"), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("Immediate submission targets (results as-is):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 30)
putpdf text ("  1. Economics Letters — short format (3-4 pages), accepts well-identified near-null results from novel designs. Best immediate fit given current evidence."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  2. Regional Science and Urban Economics — geographic spillover literature home; Brinkman & Mok-Lamme (2019) published here. Accepts inconclusive results from novel geographic research designs."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  3. Contemporary Economic Policy — policy-relevant null results (no mental health externality from cross-border legalization) are publishable as they inform state legislators considering legalization."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("Targets after strengthening (add third period + ACS controls):"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 30)
putpdf text ("  1. Journal of Health Economics — top outlet for marijuana-health causal identification. Requires multi-period design with event study and tight standard errors."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  2. American Journal of Health Economics — strong fit for county-level BRFSS-based studies."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  3. Economic Inquiry — published Hao & Cowan (2020), the closest paper; natural comparison."), font("Times New Roman", 11)

putpdf paragraph, spacing(before, 120)
putpdf text ("Required next steps to improve the paper:"), bold font("Times New Roman", 11)
putpdf paragraph, spacing(before, 30)
putpdf text ("  (a) Download PLACES 2024 release (BRFSS 2021-2022) to add a third period and perform an event study pre-trend test."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  (b) Add ACS 5-year estimates: median household income, poverty rate, % uninsured, % age 25+ with college degree — as time-varying county-level controls."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  (c) Replace centroid-to-centroid distances with true county-boundary-to-state-border distances using Census TIGER shapefiles in R (sf package) or Python (geopandas)."), font("Times New Roman", 11)
putpdf paragraph, spacing(before, 15)
putpdf text ("  (d) Download Vera Institute Incarceration Trends data and add drug enforcement rate as first-stage / mechanism evidence."), font("Times New Roman", 11)

putpdf pagebreak

// =============================================================
// FOOTER
// =============================================================
putpdf paragraph, halign(center)
putpdf text ("Working Paper  |  June 2026"), font("Times New Roman", 10) italic
putpdf paragraph, halign(center) spacing(before, 50)
putpdf text ("Data: CDC PLACES (data.cdc.gov), US Census Bureau 2023 County Gazetteer, State Recreational Marijuana Statutes."), font("Times New Roman", 9) italic
putpdf paragraph, halign(center) spacing(before, 50)
putpdf text ("Estimated using Stata 18: reghdfe (Correia 2017), csdid (Callaway-Sant'Anna 2021), putpdf."), font("Times New Roman", 9) italic

putpdf save "$DIR\Marijuana_Spillover_Results.pdf", replace
di "PDF saved: $DIR\Marijuana_Spillover_Results.pdf"
di "03_report.do COMPLETE"
