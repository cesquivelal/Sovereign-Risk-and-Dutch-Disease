** Import full data base and do exercise with all possible observations
clear all
cls
*set more off
cd "CURRENT_DIRECTORY"
insheet using "Panel.csv"

** Sort variables, declare panel and destring main variable

sort c_num year
xtset c_num year

bysort c_num: egen av_nr_rents=mean(nr_rents)

** Regress embi vs iii
gen ln_embi=log(embi)
gen ln_iii=log(iii)
xtscc ln_embi ln_iii i.year, fe

** Label variables for table outputs
label variable av_nr_rents "100*(NR rents / GDP), 1979-2015 average"
label variable reserves "Reserves / GDP"
label variable debt_gdp "Total Debt / GDP"
label variable gov_debt_gdp "Gov Debt / GDP"

*** Run regressions Table 5, spreads and natural resources
xtscc embi av_nr_rents reserves iii debt_gdp i.year
outreg2 using RegTable_Spreads.tex, tex replace ctitle(EMBI) label keep(av_nr_rents reserves iii debt_gdp) addtext(Year FE, Yes)

xtscc embi av_nr_rents reserves iii gov_debt_gdp i.year
outreg2 using RegTable_Spreads.tex, tex append ctitle(EMBI) label keep(av_nr_rents reserves iii gov_debt_gdp) addtext(Year FE, Yes)

xtscc embi_cons av_nr_rents reserves debt_gdp i.year
outreg2 using RegTable_Spreads.tex, tex append ctitle(Constructed EMBI) label keep(av_nr_rents reserves debt_gdp) addtext(Year FE, Yes)

xtscc embi_cons av_nr_rents reserves gov_debt_gdp i.year
outreg2 using RegTable_Spreads.tex, tex append ctitle(Constructed EMBI) label keep(av_nr_rents reserves gov_debt_gdp) addtext(Year FE, Yes)

*** Run regressions Table 6, investment share and natural resources
gen lambda=100*inv_m/inv_tot
gen lambda_va=100*va_m/va_tot
gen inter=reserves*nr_rents
gen ln_nr_rents=log(nr_rents+1)
gen nr_rents2=nr_rents*nr_rents

xtscc lambda nr_rents i.year , fe
outreg2 using RegTable_Lambda.tex, tex replace ctitle(1) label keep(nr_rents) addtext(Year FE, Yes, Country FE, Yes)

xtscc lambda nr_rents nr_rents2 i.year , fe
outreg2 using RegTable_Lambda.tex, tex append ctitle(2) label keep(nr_rents nr_rents2) addtext(Year FE, Yes, Country FE, Yes)

xtscc lambda nr_rents nr_rents2 reserves i.year , fe
outreg2 using RegTable_Lambda.tex, tex append ctitle(2) label keep(nr_rents nr_rents2 reserves) addtext(Year FE, Yes, Country FE, Yes)

xtscc lambda nr_rents nr_rents2 reserves inter i.year , fe
outreg2 using RegTable_Lambda.tex, tex append ctitle(4) label keep(nr_rents nr_rents2 reserves inter) addtext(Year FE, Yes, Country FE, Yes)

bysort c_num: egen av_lambda=mean(lambda)
reg av_lambda av_nr_rents


