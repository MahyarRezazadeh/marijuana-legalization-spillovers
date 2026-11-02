/* ================================================================
   MASTER DO-FILE: Marijuana Legalization Spillover Effects
   Run this file to execute the full pipeline.
   ================================================================ */
clear all
set more off

global DIR `"C:\Users\mahya\OneDrive - Simon Fraser University (1sfu)\Desktop\Marijuana spollover effect"'
cd "$DIR"

// Log everything
cap log close
log using "$DIR\run_log.txt", text replace

di "====== STEP 1: DATA PREPARATION ======"
do "$DIR\01_data_prep.do"

di "====== STEP 2: ANALYSIS + PDF ======"
do "$DIR\03_report.do"

log close
di "Pipeline complete. Check Marijuana_Spillover_Results.pdf"
