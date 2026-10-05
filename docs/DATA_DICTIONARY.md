# Data Dictionary

Data dictionary for the household SARS-CoV-2 secondary-case prediction study
(TabPFN with SHAP explanations, Swedish national register data, 2020).

This document is generated from the feature-engineering source code
(`data_preprocessing.py`, `feature_extraction.py`, `feature_aggregation.py`,
`config.py`) rather than written by hand, so that every claim below is
directly traceable to a specific line of code. Where a detail could not be
verified from code alone, it is marked **[VERIFY]** rather than guessed.

**Scope.** Household is the unit of prediction. Every predictor is
constructed from data recorded on or before the household's anchor date (the
earliest confirmed diagnosis date within the household) and reflects either (a) a fixed, hand-engineered
household aggregate computed once per household, or (b) a diagnosis code,
prescription code, or contact reason, counted per household member and
aggregated to the household as a binary/derived indicator. Category (b)'s
exact code vocabulary (which specific ICD-10/ATC codes/contact reasons appear
as columns) is determined at runtime from the raw register data and is not
available outside the secure analysis environment -- see "Per-code appendix"
at the end of this document for how to generate it.

**Feature count.** The number of model features varies slightly between
cross-validation folds (271--290; 287 in fold 1, the feature set used for the
explanations) because of the fold-wise chi-square feature-selection step
(Section 5), which is fit independently on each fold's training data.

---

## 1. Source registers

All data are accessed through the SWECOV national register-based research
project~\cite{SWECOV2020}. The pipeline reads the following pickled,
person-indexed extracts (file names as used in `feature_extraction.py`):

| Source table (author-confirmed) | Source file in pipeline | Holding authority / register | Contributes |
|---|---|---|---|
| `FHM_SMINET` | `FHM_SMINET_2020_duplicates.pkl` | Public Health Agency of Sweden (Folkhälsomyndigheten, FHM) -- SmiNet, the national notifiable communicable-disease surveillance system | Diagnosis dates used to construct case/index/secondary labels (`data_preprocessing.py`); not a predictor itself |
| `Population_PersonNr_20221231` | `Population_PersonNr_20221231_duplicates.pkl` | Statistics Sweden (SCB) / Swedish Tax Agency population register | `FodelseArMan` (birth year-month → age), `Kon` (sex) |
| `Fodelseuppg_20201231` | `Fodelseuppg_20201231_duplicates.pkl` | SCB population register (birth particulars) | `UtlSvBakg` (foreign/Swedish background), `Fodelseland` (country of birth) |
| `HushallPerson_2019/2020` | `HushallPerson_2019_duplicates.pkl` | SCB household register (household-person linkage) | `AntalBarnUnder18` (number of children under 18 in household) |
| `HushallBoende_2019/2020` | `HushallBoende_2019_duplicates.pkl` | SCB household register (household-housing linkage) | `Boarea_Person` (living area per person), `Boendeform` (housing/tenure type) |
| `Individ_2017/2018/2019/2020` | `Individ_2019_duplicates.pkl` | SCB individual-level socioeconomic microdata | `DispInk04` (individual disposable income), `DispInkFam04` (family disposable income) |
| `Inera 1177paTelefon` (call data to 1177) | `Inera_VPTU_Coronadata_2018_2020_duplicates.pkl` | Inera (Swedish eHealth infrastructure operator) -- 1177 Vårdguiden telephone/digital healthcare advice service | `contact_*` features (1177 call/contact reasons) |
| `SWECOV_SOS_LMED` | `SWECOV_SOS_LMED_2018_2020_duplicates.pkl` | National Board of Health and Welfare (Socialstyrelsen) -- Prescribed Drug Register (Läkemedelsregistret) | `lmed_*` features (dispensed ATC-coded prescriptions) |
| `SWECOV_SOS_OV` | `SWECOV_SOS_OV_duplicates.pkl` | Socialstyrelsen -- National Patient Register, outpatient care (öppenvård) | `ov_*` features (ICD-10-coded outpatient diagnoses) |
| `SWECOV_SOS_SV` | `SWECOV_SOS_SV_duplicates.pkl` | Socialstyrelsen -- National Patient Register, inpatient care (slutenvård) | `sv_*` features (ICD-10-coded inpatient/hospitalisation diagnoses) |
| `SWECOV_SOS_SOL` | `SWECOV_SOS_SOL_2018_2020_duplicates.pkl` | Socialstyrelsen -- social services statistics (Socialtjänstlagen, SoL) | `TRYGG_*` features (municipal security-alarm / home-care service indicators) |
| `RTB2019/2020` | not loaded by the feature-extraction stage | SCB Total Population Register (Registret över totalbefolkningen) | Used in `data_preprocessing.py` (not `feature_extraction.py`) to construct the base study population / household membership -- **not a predictor source** |
| `SWECOV_SOS_DORS` | `SWECOV_SOS_DORS_2020_duplicates.pkl` | Socialstyrelsen -- Cause of Death Register (Dödsorsaksregistret, DORS) | Used only in `data_preprocessing.py::filter_by_death_date` to drop logically invalid records where the recorded index/secondary-case date falls after the person's death date -- **not a predictor source**, a data-quality filter |

Per-column mapping verified against `feature_extraction.py` lines ~385-446
(demographic/socioeconomic), the `EXTRACT_INERA`/`EXTRACT_LMED`/
`EXTRACT_OV`/`EXTRACT_SV`/`EXTRACT_TRYGG` blocks (register-derived code
features), and `data_preprocessing.py` lines ~97-98, ~810-825, ~1160-1180
(`RTB`, `SWECOV_SOS_DORS`). Table column 1 (source-table names) supplied
directly by the authors; holding-authority names in column 3 follow standard
Swedish register-abbreviation conventions (FHM = Folkhälsomyndigheten;
SCB = Statistics Sweden; SOS = Socialstyrelsen) and are **[VERIFY]** against
the SWECOV project's own data-source documentation before publication --
in particular the exact register name behind `Individ_*` (e.g. whether it is
drawn from LISA, STATIV, or another SCB longitudinal database) and behind
`SOL` (Socialtjänstlagen social-services statistics) were inferred from
standard naming convention, not confirmed against SWECOV documentation.

---

## 2. Reference period / look-back window (per group)

| Feature group | Reference point | Window | Code reference |
|---|---|---|---|
| Demographic (age, sex, birth country, background) | Anchor date | Point-in-time (2020 snapshot) | `feature_aggregation.py::aggregate_household_features` |
| Socioeconomic (income, housing, children count) | Anchor date | Point-in-time (2019 register year) | same |
| `contact_*` (Inera 1177) | Anchor date | The 365 days up to and including the anchor date | `Config.INERA_WINDOW = 365`, `feature_extraction.py::count_codes` |
| `lmed_*` (prescriptions) | Anchor date | The 365 days up to and including the anchor date | `Config.LMED_WINDOW = 365` |
| `ov_*` (outpatient diagnoses) | Anchor date | All available history up to and including the anchor date | `feature_extraction.py::count_codes`, no `window_days` |
| `sv_*` (inpatient diagnoses) | Anchor date | All available history up to and including the anchor date | same |
| `TRYGG_*` (elderly care) | Anchor date | Calendar months that ended before the anchor date and started within the 365 days before it (records are monthly; the anchor month is excluded) | `Config.TRYGG_WINDOW = 365`, `feature_extraction.py::count_trygg` |

**Extract periods and window coverage.** Every anchor date lies between
2019-12-31 and 2020-12-31 (the SmiNet extract). So that the 365-day window is
complete for every household, the prescription and 1177 extracts are kept from
2018-12-31 (= 2019-12-31 − 365 days) to 2020-12-31
(`data_preprocessing.py::Config.LOOKBACK_START`); the SOL extract covers
2019-01 to 2020-12, which contains every 365-day window of completed months.
`feature_extraction.py::check_lookback_coverage` reports, at run time, the
earliest record in the prescription and 1177 extracts and the share of
household members whose window would be truncated (expected: 0%; a warning is
issued otherwise). An earlier version of the pipeline kept the
prescription and 1177 extracts only from 2019-12-31, which shortened their
window to between a few days and 12 months depending on the month of the
household's first case, and counted elderly-care records without an upper date
bound; both were corrected.

In all cases every household member's features are anchored to the single earliest confirmed diagnosis date within the
household, never to that member's own (possibly later) diagnosis date.

---

## 3. Fixed household-level features

These ~55 features are computed once per household directly in
`feature_aggregation.py::aggregate_household_features` (not derived from a
per-code vocabulary). `household_id`, `IndexDate_household`, and
`secondary_cases_count` are identifiers/label-construction fields, not
predictors (dropped before modelling -- see `tabpfn_train.py::DROP_COLS_BASE`
and the `label` derivation). Missing-data and transformation columns refer to
Sections 4 and 6 below.

| Feature | Description | Source | Missing-data rule |
|---|---|---|---|
| `household_size` | Number of valid household members | derived | n/a (always defined) |
| `index_cases_count` | Number of index cases in the household | derived from SMINET-based labels | n/a |
| `mean_age_2020` / `max_age_2020` / `min_age_2020` / `age_variance` / `age_IQR` / `age_range` | Household age summary statistics (age = 2020 − birth year) | Population Register (`FodelseArMan`) | median (Sec. 4) |
| `age_0_17_count` / `age_18_64_count` / `age_65plus_count` | Member counts by age band | derived | 0 |
| `has_member_75plus` | Indicator: any member aged ≥75 | derived | 0 |
| `proportion_children` / `proportion_elderly` | Share of members aged <18 / ≥65 | derived | 0 |
| `prop_foreign_background` / `has_any_foreign_background` / `all_foreign_background` | Share/indicator of members with foreign background (`UtlSvBakg == 11`) | Demographic Register | 0 |
| `Fodelseland_mode` → `Fodelseland_mode_freq` | Most common birth country in household, frequency-encoded | Demographic Register | "unknown" category |
| `Fodelseland_diversity` | Number of distinct birth countries in household | derived | median |
| `prop_born_sweden` | Share of members born in Sweden (`Fodelseland == 'SVERIGE'`) | Demographic Register | 0 |
| `male_count` / `female_count` / `proportion_male` / `proportion_female` | Household sex composition | Population Register (`Kon`) | 0 |
| `gender_diversity` | Indicator: household has both male and female members | derived | 0 |
| `has_child_under_6` / `has_child_6_17` / `has_elderly_65plus` | Family-structure indicators | derived | 0 |
| `multigenerational` | Indicator: max age − min age > 40 | derived | 0 |
| `three_generation` | Indicator: members present in all three of <18 / 18-64 / ≥65 | derived | 0 |
| `AntalBarnUnder18` | Number of children under 18 registered to the household | Household-Person register | median |
| `Boarea_Person` | Living area per person (m²) | Household-Housing register | median |
| `total_Boarea` | Household total living area (`Boarea_Person × household_size`) | derived | median |
| `crowding_index` | `household_size / Boarea_Person` | derived | median |
| `is_overcrowded` / `is_spacious` | Indicators: `crowding_index` > 1.5 / < 0.5 | derived | 0 |
| `Boendeform_mode` → `Boendeform_mode_<category>` | Most common housing/tenure type, one-hot encoded | Household-Housing register | "unknown" category |
| `mean_DispInk04` / `max_` / `min_` / `sd_` / `median_` / `range_DispInk04` | Household individual-disposable-income summary statistics | Socioeconomic Register | median |
| `mean_DispInkFam04` / `max_` / `min_` / `sd_` / `median_` / `range_DispInkFam04` | Household family-disposable-income summary statistics | Socioeconomic Register | median |
| `TRYGG_1_sum` / `TRYGG_total_sum` | Sum of security-alarm-type-1 / total elderly-care-service monthly records in the look-back window (Section 2) | Elderly Care Register | 0 |
| `any_TRYGG_1` / `any_TRYGG` | Indicators: household has any such record | derived | 0 |
| `proportion_with_TRYGG` | Share of members with any such record | derived | 0 |
| `TRYGG_1_per_elderly` / `TRYGG_total_per_capita` | Normalised by elderly-member count / household size | derived | 0 |
| `UtlSvBakg_mode` → `UtlSvBakg_mode_enc` | Most common foreign/Swedish-background code in household, label-encoded | Demographic Register | "unknown" category |

Columns with >20% missingness across the training fold are dropped entirely
(`Config.MISSING_RATE_THRESHOLD = 20`, fit on train only, same drop list
applied to val/test) -- see Section 4.

---

## 4. Missing-data handling

Fit on the training partition of each fold only; the same imputation
values/rules are applied to that fold's validation and test data (no
leakage). From `feature_aggregation.py::process_household_data`:

| Column type | Rule |
|---|---|
| Columns >20% missing (train fold) | Dropped entirely (`MISSING_RATE_THRESHOLD = 20`) |
| Age statistics (`mean_age_2020`, `max_age_2020`, `min_age_2020`, `age_variance`, `age_IQR`) | Filled with training-fold median |
| Binary/count columns (`has_*`, `all_*`, `multigenerational`, `three_generation`, `age_0_17_count`, `age_65plus_count`, `male_count`, `female_count`, `TRYGG_*`, `any_*`) | Filled with 0 |
| Proportion columns (`prop_*`, `proportion_*`) | Filled with 0 |
| Housing/income columns (`Boarea_Person`, `total_Boarea`, `crowding_index`, `Fodelseland_diversity`, `gender_diversity`, income statistics) | Filled with training-fold median |
| Categorical columns (`UtlSvBakg_mode`, `Fodelseland_mode`, `Boendeform_mode`) | Filled with an explicit `"unknown"` category before encoding |

---

## 5. Dynamic code-based feature families

`contact_*` (INERA), `lmed_*` (prescriptions), `ov_*` (outpatient), `sv_*`
(inpatient) are one raw count column per distinct code value observed in the
register extract. The code values are used exactly as delivered: for
`ov_*`/`sv_*` these are ICD-10 three-character categories (e.g. `ov_J45`,
asthma) or ICD-10 blocks (e.g. `ov_H65-H75`, diseases of the middle ear and
mastoid), and for `lmed_*` ATC groups at level 3--5 (e.g. `lmed_J01A`,
tetracyclines; `lmed_C07AB02`, metoprolol). Each column counts that code's
records for the household, up to the anchor date. The exact set of codes
is determined at runtime from the raw register extracts
(`feature_extraction.py::load_unique_codes`) and therefore is **not
enumerable from this codebase alone** -- see "Per-code appendix" below.

Each raw count column is then, per fold (`process_household_data`, fit on
that fold's train only):
1. Binarised into three derived features: `{col}_occurred` (count > 0),
   `{col}_exceeds_median` (count > training-fold median),
   `{col}_exceeds_p75` (count > training-fold 75th percentile). The raw count
   column itself is dropped.
2. Filtered by a chi-square test against the label: a derived feature is kept
   only if `p < 0.02` (`CHI2_P_VALUE_THRESHOLD`) **and** its most frequent
   value accounts for less than 95% of households (`MAX_FREQ_THRESHOLD`,
   guards against near-constant features). This selection is fit
   independently per fold, which is why the feature count varies slightly
   between folds (see above).

This matches the suffix notation already used in the manuscript's global
feature-importance figure (`↑` = exceeds median, `↑↑` = exceeds 75th
percentile).

---

## 6. Encoding and transformation summary

| Step | Applies to | Method | Fit on |
|---|---|---|---|
| Label encoding | `UtlSvBakg_mode` | `sklearn.LabelEncoder` | train fold |
| Frequency encoding | `Fodelseland_mode` | value-count frequency map | train fold |
| One-hot encoding | `Boendeform_mode` | `sklearn.OneHotEncoder` | train fold |
| Log transform | `AntalBarnUnder18`, `Boarea_Person`, `total_Boarea`, `mean_DispInk04`, `mean_DispInkFam04` | `log1p`, suffix `_log` | n/a (deterministic) |
| Winsorisation | all numeric columns, before standardisation | clip to that split's own [1st, 99th] percentile | each split independently |
| Standardisation | all continuous/log/frequency columns and `index_cases_count` (full list in `standardize_household_data`) | `sklearn.StandardScaler`, suffix `_std`, original column dropped | train fold |
| Binary encoding + chi-square selection | `contact_*`, `lmed_*`, `ov_*`, `sv_*` | see Section 5 | train fold, per fold |

---

## 7. Software, versions, seeds

Environment that produced the reported results:

- TabPFN package 6.3.2; checkpoint `tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt` (TabPFN-2.5 classifier, local, offline-loaded; SHA-256 `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`)
- Python version: 3.12.12 (conda env `env_py312`, Linux, glibc 2.17)
- Key library versions: torch 2.4.1+cu124 (CUDA 12.4), numpy 2.2.6, pandas 2.3.2, scikit-learn 1.7.2, scipy 1.16.3, shap 0.50.0, xgboost 3.1.2, lightgbm 3.3.5, catboost 1.2.10, matplotlib 3.10.8; full list in `requirements.txt`
- Random seeds: `RANDOM_STATE = 42` (train/test split and `StratifiedKFold`, `feature_aggregation.py`); `random_state=42` (bagging/ensemble, `tabpfn_train.py::CV_CONFIG`); SHAP explain/background sampling seeds 1 and 0 respectively (`tabpfn_xai.py::_stratified_sample` calls)
- Code version: the tagged release of this repository that accompanies the publication.

---

## 8. Per-code appendix -- DONE (generated 2026-09-22)

The full pre-selection code vocabulary has been exported from the secure
environment (`code_dictionary_appendix.py`, run by the authors) and enriched
here: `code_dictionary_appendix_enriched.csv` (2,380 rows: 189
`contact_reasons`, 1,060 `ov_codes`, 816 `sv_codes`, 315 `atc_codes`), with
three added columns:

- **`category`** -- for `atc_codes`, the WHO ATC level-1 anatomical main
  group (e.g. `J` = Antiinfectives for systemic use); for `ov_codes`/`sv_codes`,
  the WHO ICD-10 chapter (e.g. `J00-J99` = Diseases of the respiratory
  system), including the special supplementary `U00-U49` chapter. Both are
  public, standard classifications applied here programmatically by code
  prefix (chapter / anatomical-group level). Names of the codes retained in
  the final model are given in Section 9.
- **`clinical_interpretation`** -- filled in for all 189 `contact_reasons`
  (hand-translated from the source Swedish 1177 Vårdguiden triage category
  names; left blank for `ov_codes`/`sv_codes`/`atc_codes`, where only the
  chapter-level `category` is populated, not a per-code gloss).
- **`data_quality_flag`** -- notes codes that don't fit a standard pattern,
  rather than silently miscategorising them (see finding below).

**Notable finding: COVID-19-specific ICD-10 codes are present as features.**
The vocabulary includes WHO's special supplementary ICD-10 codes for
COVID-19 (`U07.1`/`U07.2` confirmed/probable COVID-19, `U08.9` personal
history of COVID-19, `U09.9` post-COVID-19 condition, `U10.9` paediatric
multisystem inflammatory syndrome) among both `ov_codes` and `sv_codes`.
Because every feature is bounded by the household anchor date, any such code counted for a household
member is, by construction, dated on or before that household's anchor date
-- i.e.\ it can only reflect a *prior* (pre-anchor) COVID-19 episode, not the
secondary case that defines the outcome. This is legitimate predictive
signal (prior infection/immunity status), not leakage, provided the anchor
bound holds.

**Data-quality note: not every `ov_codes`/`sv_codes` entry is a standard
ICD-10 code.** 198 of 1,876 (10.6%) don't match the standard
letter-plus-two-digits pattern, including 4 literal `?`/`??`/`???`/`????`
placeholder values (unspecified/unknown diagnosis code in the source
register) and a number of short numeric-only or malformed fragments (e.g.\
`'Z718`, `-EJ`, `0O21`) that look like export artefacts rather than genuine
codes. These are flagged in `data_quality_flag` rather than guessed at; the
authors should decide whether any warrant a data-cleaning pass upstream in
`feature_extraction.py`, or a brief acknowledgement in the manuscript's
Limitations that a small fraction of source diagnosis codes are
unspecified/malformed in the raw register export.

This appendix lists the full pre-selection vocabulary; the codes retained in
the final model, with their names, are listed in Section 9.

### How this was generated (for reference / reproducibility)
```python
# code_dictionary_appendix.py -- run inside the secure environment
import pandas as pd
from feature_extraction import Config, load_unique_codes

codes = load_unique_codes()  # {'contact_reasons', 'atc_codes', 'ov_codes', 'sv_codes'}
rows = []
for group, code_set in codes.items():
    for code in sorted(code_set):
        rows.append({'group': group, 'code': code})
pd.DataFrame(rows).to_csv('code_dictionary_appendix.csv', index=False)
```
The `category`/`clinical_interpretation`/`data_quality_flag` columns were
then added by a local enrichment pass (rule-based ATC/ICD-10 chapter lookup
+ hand-translated contact-reason dictionary), not by re-running anything in
the secure environment.

---

## 9. Final model features and figure labels

The table below lists the 287 features used by the explained TabPFN ensemble
(fold-1 feature set; column order as in the SHAP matrices) with the label
used for that feature in every manuscript figure. It is generated from
`feature_labels.py` (`python feature_labels.py --markdown`), which
`tabpfn_xai.py` also imports, so the figure labels and this table cannot
diverge.

Label conventions: **(z)** = standardised (z-score, fit on the training
fold); **(log, z)** = log1p-transformed, then standardised; unmarked numeric
features are on their original scale after winsorisation. For code-derived
features, **OP** = outpatient ICD-10 diagnosis, **Rx** = dispensed ATC
prescription; **any** = at least one record, **>med** / **>P75** = household
count above the training-fold median / 75th percentile (Section 5). Figure
labels are deliberately short; the Description column gives the full name. No
`sv_*` (inpatient) or `contact_*` (1177) feature survived the chi-square
selection in this fold, so all diagnosis features below are outpatient.
ICD-10 block and ATC group names are abbreviated from the WHO
classifications. Housing-type (`Boendeform`) category names follow SCB's
tenure/housing-type classification **[VERIFY against the SCB variable
documentation]**.

| Column | Figure label | Description | Group |
|---|---|---|---|
| `household_size` | Household size | Number of household members | Household-level |
| `age_0_17_count` | Members aged 0-17 | Number of members aged 0-17 | Household-level |
| `age_65plus_count` | Members aged 65+ | Number of members aged 65 or older | Household-level |
| `has_member_75plus` | Any member 75+ | Indicator: any member aged 75 or older | Household-level |
| `proportion_children` | Share of children | Share of members aged under 18 | Household-level |
| `proportion_elderly` | Share aged 65+ | Share of members aged 65 or older | Household-level |
| `has_any_foreign_background` | Any foreign background | Indicator: any member with foreign background | Household-level |
| `all_foreign_background` | All foreign background | Indicator: all members with foreign background | Household-level |
| `male_count` | No. of males | Number of male members | Household-level |
| `female_count` | No. of females | Number of female members | Household-level |
| `proportion_male` | Share male | Share of male members | Household-level |
| `proportion_female` | Share female | Share of female members | Household-level |
| `gender_diversity` | Both sexes present | Indicator: household has both male and female members | Household-level |
| `has_child_under_6` | Child under 6 | Indicator: any member aged under 6 | Household-level |
| `has_child_6_17` | Child aged 6-17 | Indicator: any member aged 6-17 | Household-level |
| `has_elderly_65plus` | Any member 65+ | Indicator: any member aged 65 or older | Household-level |
| `multigenerational` | Multigenerational | Indicator: oldest minus youngest member age > 40 years | Household-level |
| `three_generation` | Three generations | Indicator: members in all of <18, 18-64 and 65+ | Household-level |
| `AntalBarnUnder18` | Children <18 (register) | Number of children under 18 registered to the household | Household-level |
| `is_overcrowded` | Overcrowded | Indicator: crowding index > 1.5 | Household-level |
| `is_spacious` | Spacious | Indicator: crowding index < 0.5 | Household-level |
| `median_DispInk04` | Individual income, median | median of members' individual disposable income (SEK) | Household-level |
| `range_DispInk04` | Individual income, range | range of members' individual disposable income (SEK) | Household-level |
| `median_DispInkFam04` | Family income, median | median of members' family disposable income (SEK) | Household-level |
| `range_DispInkFam04` | Family income, range | range of members' family disposable income (SEK) | Household-level |
| `TRYGG_1_sum` | Safety-alarm records | Sum of security-alarm (type 1) elderly-care records | Household-level |
| `TRYGG_total_sum` | Elderly-care records | Sum of all elderly-care service records | Household-level |
| `any_TRYGG_1` | Any safety-alarm record | Indicator: any security-alarm (type 1) record | Household-level |
| `any_TRYGG` | Any elderly-care record | Indicator: any elderly-care service record | Household-level |
| `proportion_with_TRYGG` | Share with elderly care | Share of members with any elderly-care service record | Household-level |
| `TRYGG_total_per_capita` | Elderly-care records/member | Elderly-care service records per household member | Household-level |
| `age_18_64_count` | Members aged 18-64 | Number of members aged 18-64 | Household-level |
| `age_range` | Age, range | Oldest minus youngest member age | Household-level |
| `prop_born_sweden` | Share born in Sweden | Share of members born in Sweden | Household-level |
| `Boendeform_mode_FlerAg` | Housing: flat, owned | Indicator: modal housing/tenure type is "multi-dwelling, owner-occupied" | Household-level |
| `Boendeform_mode_FlerBo` | Housing: flat, tenant-owned | Indicator: modal housing/tenure type is "multi-dwelling, tenant-owned" | Household-level |
| `Boendeform_mode_FlerHy` | Housing: flat, rented | Indicator: modal housing/tenure type is "multi-dwelling, rented" | Household-level |
| `Boendeform_mode_Ovr` | Housing: other | Indicator: modal housing/tenure type is "other housing" | Household-level |
| `Boendeform_mode_SmahusAg` | Housing: house, owned | Indicator: modal housing/tenure type is "small house, owner-occupied" | Household-level |
| `Boendeform_mode_SmahusBo` | Housing: house, tenant-owned | Indicator: modal housing/tenure type is "small house, tenant-owned" | Household-level |
| `Boendeform_mode_SmahusHy` | Housing: house, rented | Indicator: modal housing/tenure type is "small house, rented" | Household-level |
| `Boendeform_mode_SpecAF` | Housing: special, elderly/disabled | Indicator: modal housing/tenure type is "special housing, elderly/disabled" | Household-level |
| `Boendeform_mode_SpecOvr` | Housing: special, other | Indicator: modal housing/tenure type is "special housing, other" | Household-level |
| `Boendeform_mode_SpecStu` | Housing: student housing | Indicator: modal housing/tenure type is "special housing, students" | Household-level |
| `Boendeform_mode_unknown` | Housing: unknown | Indicator: modal housing/tenure type is "unknown" | Household-level |
| `lmed_A01A_occurred` | Rx A01A stomatologicals: any | Dispensed prescriptions (Prescribed Drug Register), ATC A01A (stomatological preparations): at least one record | Dispensed prescriptions |
| `lmed_A01A_exceeds_median` | Rx A01A stomatologicals: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A01A (stomatological preparations): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A01A_exceeds_p75` | Rx A01A stomatologicals: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A01A (stomatological preparations): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_A02B_occurred` | Rx A02B ulcer/reflux drugs: any | Dispensed prescriptions (Prescribed Drug Register), ATC A02B (drugs for peptic ulcer & GORD): at least one record | Dispensed prescriptions |
| `lmed_A02B_exceeds_median` | Rx A02B ulcer/reflux drugs: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A02B (drugs for peptic ulcer & GORD): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A02B_exceeds_p75` | Rx A02B ulcer/reflux drugs: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A02B (drugs for peptic ulcer & GORD): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_A06A_occurred` | Rx A06A laxatives: any | Dispensed prescriptions (Prescribed Drug Register), ATC A06A (drugs for constipation): at least one record | Dispensed prescriptions |
| `lmed_A06A_exceeds_median` | Rx A06A laxatives: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A06A (drugs for constipation): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A06A_exceeds_p75` | Rx A06A laxatives: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A06A (drugs for constipation): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_A10B_occurred` | Rx A10B oral antidiabetics: any | Dispensed prescriptions (Prescribed Drug Register), ATC A10B (blood-glucose-lowering drugs (non-insulin)): at least one record | Dispensed prescriptions |
| `lmed_A10B_exceeds_median` | Rx A10B oral antidiabetics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A10B (blood-glucose-lowering drugs (non-insulin)): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A10B_exceeds_p75` | Rx A10B oral antidiabetics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A10B (blood-glucose-lowering drugs (non-insulin)): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_A11C_occurred` | Rx A11C vitamin A/D: any | Dispensed prescriptions (Prescribed Drug Register), ATC A11C (vitamin A and D): at least one record | Dispensed prescriptions |
| `lmed_A11C_exceeds_median` | Rx A11C vitamin A/D: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A11C (vitamin A and D): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A11C_exceeds_p75` | Rx A11C vitamin A/D: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A11C (vitamin A and D): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_A12A_occurred` | Rx A12A calcium: any | Dispensed prescriptions (Prescribed Drug Register), ATC A12A (calcium): at least one record | Dispensed prescriptions |
| `lmed_A12A_exceeds_median` | Rx A12A calcium: >med | Dispensed prescriptions (Prescribed Drug Register), ATC A12A (calcium): household count above the training-fold median | Dispensed prescriptions |
| `lmed_A12A_exceeds_p75` | Rx A12A calcium: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC A12A (calcium): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_B01A_occurred` | Rx B01A antithrombotics: any | Dispensed prescriptions (Prescribed Drug Register), ATC B01A (antithrombotic agents): at least one record | Dispensed prescriptions |
| `lmed_B01A_exceeds_median` | Rx B01A antithrombotics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC B01A (antithrombotic agents): household count above the training-fold median | Dispensed prescriptions |
| `lmed_B01A_exceeds_p75` | Rx B01A antithrombotics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC B01A (antithrombotic agents): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_B03B_occurred` | Rx B03B B12/folic acid: any | Dispensed prescriptions (Prescribed Drug Register), ATC B03B (vitamin B12 and folic acid): at least one record | Dispensed prescriptions |
| `lmed_B03B_exceeds_median` | Rx B03B B12/folic acid: >med | Dispensed prescriptions (Prescribed Drug Register), ATC B03B (vitamin B12 and folic acid): household count above the training-fold median | Dispensed prescriptions |
| `lmed_B03B_exceeds_p75` | Rx B03B B12/folic acid: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC B03B (vitamin B12 and folic acid): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C07A_occurred` | Rx C07A beta blockers: any | Dispensed prescriptions (Prescribed Drug Register), ATC C07A (beta blockers): at least one record | Dispensed prescriptions |
| `lmed_C07A_exceeds_median` | Rx C07A beta blockers: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C07A (beta blockers): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C07A_exceeds_p75` | Rx C07A beta blockers: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C07A (beta blockers): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C07AB02_occurred` | Rx C07AB02 metoprolol: any | Dispensed prescriptions (Prescribed Drug Register), ATC C07AB02 (metoprolol): at least one record | Dispensed prescriptions |
| `lmed_C07AB02_exceeds_median` | Rx C07AB02 metoprolol: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C07AB02 (metoprolol): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C07AB02_exceeds_p75` | Rx C07AB02 metoprolol: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C07AB02 (metoprolol): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C08CA_occurred` | Rx C08CA dihydropyridines: any | Dispensed prescriptions (Prescribed Drug Register), ATC C08CA (dihydropyridine calcium-channel blockers): at least one record | Dispensed prescriptions |
| `lmed_C08CA_exceeds_median` | Rx C08CA dihydropyridines: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C08CA (dihydropyridine calcium-channel blockers): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C08CA_exceeds_p75` | Rx C08CA dihydropyridines: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C08CA (dihydropyridine calcium-channel blockers): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C09A_occurred` | Rx C09A ACE inhibitors: any | Dispensed prescriptions (Prescribed Drug Register), ATC C09A (ACE inhibitors): at least one record | Dispensed prescriptions |
| `lmed_C09A_exceeds_median` | Rx C09A ACE inhibitors: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C09A (ACE inhibitors): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C09A_exceeds_p75` | Rx C09A ACE inhibitors: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C09A (ACE inhibitors): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C09C_occurred` | Rx C09C ARBs: any | Dispensed prescriptions (Prescribed Drug Register), ATC C09C (angiotensin II receptor blockers): at least one record | Dispensed prescriptions |
| `lmed_C09C_exceeds_median` | Rx C09C ARBs: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C09C (angiotensin II receptor blockers): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C09C_exceeds_p75` | Rx C09C ARBs: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C09C (angiotensin II receptor blockers): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_C10A_occurred` | Rx C10A lipid modifiers: any | Dispensed prescriptions (Prescribed Drug Register), ATC C10A (lipid-modifying agents): at least one record | Dispensed prescriptions |
| `lmed_C10A_exceeds_median` | Rx C10A lipid modifiers: >med | Dispensed prescriptions (Prescribed Drug Register), ATC C10A (lipid-modifying agents): household count above the training-fold median | Dispensed prescriptions |
| `lmed_C10A_exceeds_p75` | Rx C10A lipid modifiers: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC C10A (lipid-modifying agents): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_D01A_occurred` | Rx D01A topical antifungals: any | Dispensed prescriptions (Prescribed Drug Register), ATC D01A (topical antifungals): at least one record | Dispensed prescriptions |
| `lmed_D01A_exceeds_median` | Rx D01A topical antifungals: >med | Dispensed prescriptions (Prescribed Drug Register), ATC D01A (topical antifungals): household count above the training-fold median | Dispensed prescriptions |
| `lmed_D01A_exceeds_p75` | Rx D01A topical antifungals: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC D01A (topical antifungals): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_D02A_occurred` | Rx D02A emollients: any | Dispensed prescriptions (Prescribed Drug Register), ATC D02A (emollients & protectives): at least one record | Dispensed prescriptions |
| `lmed_D02A_exceeds_median` | Rx D02A emollients: >med | Dispensed prescriptions (Prescribed Drug Register), ATC D02A (emollients & protectives): household count above the training-fold median | Dispensed prescriptions |
| `lmed_D02A_exceeds_p75` | Rx D02A emollients: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC D02A (emollients & protectives): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_D07A_occurred` | Rx D07A topical steroids: any | Dispensed prescriptions (Prescribed Drug Register), ATC D07A (topical corticosteroids): at least one record | Dispensed prescriptions |
| `lmed_D07A_exceeds_median` | Rx D07A topical steroids: >med | Dispensed prescriptions (Prescribed Drug Register), ATC D07A (topical corticosteroids): household count above the training-fold median | Dispensed prescriptions |
| `lmed_D07A_exceeds_p75` | Rx D07A topical steroids: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC D07A (topical corticosteroids): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_D10A_occurred` | Rx D10A anti-acne: any | Dispensed prescriptions (Prescribed Drug Register), ATC D10A (topical anti-acne preparations): at least one record | Dispensed prescriptions |
| `lmed_D10A_exceeds_median` | Rx D10A anti-acne: >med | Dispensed prescriptions (Prescribed Drug Register), ATC D10A (topical anti-acne preparations): household count above the training-fold median | Dispensed prescriptions |
| `lmed_D10A_exceeds_p75` | Rx D10A anti-acne: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC D10A (topical anti-acne preparations): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_G03A_occurred` | Rx G03A contraceptives: any | Dispensed prescriptions (Prescribed Drug Register), ATC G03A (hormonal contraceptives): at least one record | Dispensed prescriptions |
| `lmed_G03A_exceeds_median` | Rx G03A contraceptives: >med | Dispensed prescriptions (Prescribed Drug Register), ATC G03A (hormonal contraceptives): household count above the training-fold median | Dispensed prescriptions |
| `lmed_G03A_exceeds_p75` | Rx G03A contraceptives: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC G03A (hormonal contraceptives): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_G03C_occurred` | Rx G03C oestrogens: any | Dispensed prescriptions (Prescribed Drug Register), ATC G03C (oestrogens): at least one record | Dispensed prescriptions |
| `lmed_G03C_exceeds_median` | Rx G03C oestrogens: >med | Dispensed prescriptions (Prescribed Drug Register), ATC G03C (oestrogens): household count above the training-fold median | Dispensed prescriptions |
| `lmed_G03C_exceeds_p75` | Rx G03C oestrogens: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC G03C (oestrogens): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_G04B_occurred` | Rx G04B urologicals: any | Dispensed prescriptions (Prescribed Drug Register), ATC G04B (urologicals): at least one record | Dispensed prescriptions |
| `lmed_G04B_exceeds_median` | Rx G04B urologicals: >med | Dispensed prescriptions (Prescribed Drug Register), ATC G04B (urologicals): household count above the training-fold median | Dispensed prescriptions |
| `lmed_G04B_exceeds_p75` | Rx G04B urologicals: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC G04B (urologicals): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_H02A_occurred` | Rx H02A systemic steroids: any | Dispensed prescriptions (Prescribed Drug Register), ATC H02A (systemic corticosteroids): at least one record | Dispensed prescriptions |
| `lmed_H02A_exceeds_median` | Rx H02A systemic steroids: >med | Dispensed prescriptions (Prescribed Drug Register), ATC H02A (systemic corticosteroids): household count above the training-fold median | Dispensed prescriptions |
| `lmed_H02A_exceeds_p75` | Rx H02A systemic steroids: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC H02A (systemic corticosteroids): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_H03A_occurred` | Rx H03A thyroid: any | Dispensed prescriptions (Prescribed Drug Register), ATC H03A (thyroid preparations): at least one record | Dispensed prescriptions |
| `lmed_H03A_exceeds_median` | Rx H03A thyroid: >med | Dispensed prescriptions (Prescribed Drug Register), ATC H03A (thyroid preparations): household count above the training-fold median | Dispensed prescriptions |
| `lmed_H03A_exceeds_p75` | Rx H03A thyroid: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC H03A (thyroid preparations): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_J01A_occurred` | Rx J01A tetracyclines: any | Dispensed prescriptions (Prescribed Drug Register), ATC J01A (tetracyclines): at least one record | Dispensed prescriptions |
| `lmed_J01A_exceeds_median` | Rx J01A tetracyclines: >med | Dispensed prescriptions (Prescribed Drug Register), ATC J01A (tetracyclines): household count above the training-fold median | Dispensed prescriptions |
| `lmed_J01A_exceeds_p75` | Rx J01A tetracyclines: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC J01A (tetracyclines): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_J01C_occurred` | Rx J01C penicillins: any | Dispensed prescriptions (Prescribed Drug Register), ATC J01C (penicillins): at least one record | Dispensed prescriptions |
| `lmed_J01C_exceeds_median` | Rx J01C penicillins: >med | Dispensed prescriptions (Prescribed Drug Register), ATC J01C (penicillins): household count above the training-fold median | Dispensed prescriptions |
| `lmed_J01C_exceeds_p75` | Rx J01C penicillins: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC J01C (penicillins): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_J01X_occurred` | Rx J01X other antibiotics: any | Dispensed prescriptions (Prescribed Drug Register), ATC J01X (other antibacterials): at least one record | Dispensed prescriptions |
| `lmed_J01X_exceeds_median` | Rx J01X other antibiotics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC J01X (other antibacterials): household count above the training-fold median | Dispensed prescriptions |
| `lmed_J01X_exceeds_p75` | Rx J01X other antibiotics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC J01X (other antibacterials): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_M01A_occurred` | Rx M01A NSAIDs: any | Dispensed prescriptions (Prescribed Drug Register), ATC M01A (NSAIDs): at least one record | Dispensed prescriptions |
| `lmed_M01A_exceeds_median` | Rx M01A NSAIDs: >med | Dispensed prescriptions (Prescribed Drug Register), ATC M01A (NSAIDs): household count above the training-fold median | Dispensed prescriptions |
| `lmed_M01A_exceeds_p75` | Rx M01A NSAIDs: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC M01A (NSAIDs): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_M03B_occurred` | Rx M03B muscle relaxants: any | Dispensed prescriptions (Prescribed Drug Register), ATC M03B (centrally acting muscle relaxants): at least one record | Dispensed prescriptions |
| `lmed_M03B_exceeds_median` | Rx M03B muscle relaxants: >med | Dispensed prescriptions (Prescribed Drug Register), ATC M03B (centrally acting muscle relaxants): household count above the training-fold median | Dispensed prescriptions |
| `lmed_M03B_exceeds_p75` | Rx M03B muscle relaxants: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC M03B (centrally acting muscle relaxants): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_N02A_occurred` | Rx N02A opioids: any | Dispensed prescriptions (Prescribed Drug Register), ATC N02A (opioids): at least one record | Dispensed prescriptions |
| `lmed_N02A_exceeds_median` | Rx N02A opioids: >med | Dispensed prescriptions (Prescribed Drug Register), ATC N02A (opioids): household count above the training-fold median | Dispensed prescriptions |
| `lmed_N02A_exceeds_p75` | Rx N02A opioids: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC N02A (opioids): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_N02B_occurred` | Rx N02B other analgesics: any | Dispensed prescriptions (Prescribed Drug Register), ATC N02B (other analgesics & antipyretics): at least one record | Dispensed prescriptions |
| `lmed_N02B_exceeds_median` | Rx N02B other analgesics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC N02B (other analgesics & antipyretics): household count above the training-fold median | Dispensed prescriptions |
| `lmed_N02B_exceeds_p75` | Rx N02B other analgesics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC N02B (other analgesics & antipyretics): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_N05B_occurred` | Rx N05B anxiolytics: any | Dispensed prescriptions (Prescribed Drug Register), ATC N05B (anxiolytics): at least one record | Dispensed prescriptions |
| `lmed_N05B_exceeds_median` | Rx N05B anxiolytics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC N05B (anxiolytics): household count above the training-fold median | Dispensed prescriptions |
| `lmed_N05B_exceeds_p75` | Rx N05B anxiolytics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC N05B (anxiolytics): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R01A_occurred` | Rx R01A nasal decongestants: any | Dispensed prescriptions (Prescribed Drug Register), ATC R01A (nasal decongestants (topical)): at least one record | Dispensed prescriptions |
| `lmed_R01A_exceeds_median` | Rx R01A nasal decongestants: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R01A (nasal decongestants (topical)): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R01A_exceeds_p75` | Rx R01A nasal decongestants: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R01A (nasal decongestants (topical)): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R03A_occurred` | Rx R03A inhaled adrenergics: any | Dispensed prescriptions (Prescribed Drug Register), ATC R03A (inhaled adrenergics): at least one record | Dispensed prescriptions |
| `lmed_R03A_exceeds_median` | Rx R03A inhaled adrenergics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R03A (inhaled adrenergics): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R03A_exceeds_p75` | Rx R03A inhaled adrenergics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R03A (inhaled adrenergics): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R03AK_occurred` | Rx R03AK combination inhalers: any | Dispensed prescriptions (Prescribed Drug Register), ATC R03AK (adrenergic + corticosteroid inhalers): at least one record | Dispensed prescriptions |
| `lmed_R03AK_exceeds_median` | Rx R03AK combination inhalers: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R03AK (adrenergic + corticosteroid inhalers): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R03AK_exceeds_p75` | Rx R03AK combination inhalers: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R03AK (adrenergic + corticosteroid inhalers): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R03BA_occurred` | Rx R03BA inhaled steroids: any | Dispensed prescriptions (Prescribed Drug Register), ATC R03BA (inhaled glucocorticoids): at least one record | Dispensed prescriptions |
| `lmed_R03BA_exceeds_median` | Rx R03BA inhaled steroids: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R03BA (inhaled glucocorticoids): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R03BA_exceeds_p75` | Rx R03BA inhaled steroids: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R03BA (inhaled glucocorticoids): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R05C_occurred` | Rx R05C expectorants: any | Dispensed prescriptions (Prescribed Drug Register), ATC R05C (expectorants): at least one record | Dispensed prescriptions |
| `lmed_R05C_exceeds_median` | Rx R05C expectorants: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R05C (expectorants): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R05C_exceeds_p75` | Rx R05C expectorants: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R05C (expectorants): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R05F_occurred` | Rx R05F cough combinations: any | Dispensed prescriptions (Prescribed Drug Register), ATC R05F (cough suppressant/expectorant combinations): at least one record | Dispensed prescriptions |
| `lmed_R05F_exceeds_median` | Rx R05F cough combinations: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R05F (cough suppressant/expectorant combinations): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R05F_exceeds_p75` | Rx R05F cough combinations: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R05F (cough suppressant/expectorant combinations): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_R06A_occurred` | Rx R06A antihistamines: any | Dispensed prescriptions (Prescribed Drug Register), ATC R06A (systemic antihistamines): at least one record | Dispensed prescriptions |
| `lmed_R06A_exceeds_median` | Rx R06A antihistamines: >med | Dispensed prescriptions (Prescribed Drug Register), ATC R06A (systemic antihistamines): household count above the training-fold median | Dispensed prescriptions |
| `lmed_R06A_exceeds_p75` | Rx R06A antihistamines: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC R06A (systemic antihistamines): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_S01G_occurred` | Rx S01G eye antiallergics: any | Dispensed prescriptions (Prescribed Drug Register), ATC S01G (ophthalmic decongestants & antiallergics): at least one record | Dispensed prescriptions |
| `lmed_S01G_exceeds_median` | Rx S01G eye antiallergics: >med | Dispensed prescriptions (Prescribed Drug Register), ATC S01G (ophthalmic decongestants & antiallergics): household count above the training-fold median | Dispensed prescriptions |
| `lmed_S01G_exceeds_p75` | Rx S01G eye antiallergics: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC S01G (ophthalmic decongestants & antiallergics): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `lmed_S03C_occurred` | Rx S03C eye/ear steroid+AB: any | Dispensed prescriptions (Prescribed Drug Register), ATC S03C (eye/ear corticosteroid + anti-infective): at least one record | Dispensed prescriptions |
| `lmed_S03C_exceeds_median` | Rx S03C eye/ear steroid+AB: >med | Dispensed prescriptions (Prescribed Drug Register), ATC S03C (eye/ear corticosteroid + anti-infective): household count above the training-fold median | Dispensed prescriptions |
| `lmed_S03C_exceeds_p75` | Rx S03C eye/ear steroid+AB: >P75 | Dispensed prescriptions (Prescribed Drug Register), ATC S03C (eye/ear corticosteroid + anti-infective): household count above the training-fold 75th percentile | Dispensed prescriptions |
| `ov_D22_occurred` | OP D22 naevi: any | Outpatient diagnoses (National Patient Register), ICD-10 D22 (melanocytic naevi): at least one record | Outpatient diagnoses |
| `ov_D22_exceeds_median` | OP D22 naevi: >med | Outpatient diagnoses (National Patient Register), ICD-10 D22 (melanocytic naevi): household count above the training-fold median | Outpatient diagnoses |
| `ov_D22_exceeds_p75` | OP D22 naevi: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 D22 (melanocytic naevi): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_F41_occurred` | OP F41 anxiety: any | Outpatient diagnoses (National Patient Register), ICD-10 F41 (other anxiety disorders): at least one record | Outpatient diagnoses |
| `ov_F41_exceeds_median` | OP F41 anxiety: >med | Outpatient diagnoses (National Patient Register), ICD-10 F41 (other anxiety disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_F41_exceeds_p75` | OP F41 anxiety: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 F41 (other anxiety disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_G40-G47_occurred` | OP G40-G47 epilepsy/migraine: any | Outpatient diagnoses (National Patient Register), ICD-10 G40-G47 (episodic & paroxysmal disorders): at least one record | Outpatient diagnoses |
| `ov_G40-G47_exceeds_median` | OP G40-G47 epilepsy/migraine: >med | Outpatient diagnoses (National Patient Register), ICD-10 G40-G47 (episodic & paroxysmal disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_G40-G47_exceeds_p75` | OP G40-G47 epilepsy/migraine: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 G40-G47 (episodic & paroxysmal disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_H49-H52_occurred` | OP H49-H52 refraction/eye muscle: any | Outpatient diagnoses (National Patient Register), ICD-10 H49-H52 (ocular muscle & refraction disorders): at least one record | Outpatient diagnoses |
| `ov_H49-H52_exceeds_median` | OP H49-H52 refraction/eye muscle: >med | Outpatient diagnoses (National Patient Register), ICD-10 H49-H52 (ocular muscle & refraction disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_H49-H52_exceeds_p75` | OP H49-H52 refraction/eye muscle: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 H49-H52 (ocular muscle & refraction disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_H65-H75_occurred` | OP H65-H75 middle ear: any | Outpatient diagnoses (National Patient Register), ICD-10 H65-H75 (middle ear & mastoid): at least one record | Outpatient diagnoses |
| `ov_H65-H75_exceeds_median` | OP H65-H75 middle ear: >med | Outpatient diagnoses (National Patient Register), ICD-10 H65-H75 (middle ear & mastoid): household count above the training-fold median | Outpatient diagnoses |
| `ov_H65-H75_exceeds_p75` | OP H65-H75 middle ear: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 H65-H75 (middle ear & mastoid): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_H90-H95_occurred` | OP H90-H95 other ear: any | Outpatient diagnoses (National Patient Register), ICD-10 H90-H95 (other disorders of ear): at least one record | Outpatient diagnoses |
| `ov_H90-H95_exceeds_median` | OP H90-H95 other ear: >med | Outpatient diagnoses (National Patient Register), ICD-10 H90-H95 (other disorders of ear): household count above the training-fold median | Outpatient diagnoses |
| `ov_H90-H95_exceeds_p75` | OP H90-H95 other ear: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 H90-H95 (other disorders of ear): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_J30_occurred` | OP J30 allergic rhinitis: any | Outpatient diagnoses (National Patient Register), ICD-10 J30 (vasomotor & allergic rhinitis): at least one record | Outpatient diagnoses |
| `ov_J30_exceeds_median` | OP J30 allergic rhinitis: >med | Outpatient diagnoses (National Patient Register), ICD-10 J30 (vasomotor & allergic rhinitis): household count above the training-fold median | Outpatient diagnoses |
| `ov_J30_exceeds_p75` | OP J30 allergic rhinitis: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 J30 (vasomotor & allergic rhinitis): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_J35_occurred` | OP J35 tonsils/adenoids: any | Outpatient diagnoses (National Patient Register), ICD-10 J35 (chronic tonsil & adenoid disease): at least one record | Outpatient diagnoses |
| `ov_J35_exceeds_median` | OP J35 tonsils/adenoids: >med | Outpatient diagnoses (National Patient Register), ICD-10 J35 (chronic tonsil & adenoid disease): household count above the training-fold median | Outpatient diagnoses |
| `ov_J35_exceeds_p75` | OP J35 tonsils/adenoids: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 J35 (chronic tonsil & adenoid disease): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_J45_occurred` | OP J45 asthma: any | Outpatient diagnoses (National Patient Register), ICD-10 J45 (asthma): at least one record | Outpatient diagnoses |
| `ov_J45_exceeds_median` | OP J45 asthma: >med | Outpatient diagnoses (National Patient Register), ICD-10 J45 (asthma): household count above the training-fold median | Outpatient diagnoses |
| `ov_J45_exceeds_p75` | OP J45 asthma: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 J45 (asthma): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_K20-K31_occurred` | OP K20-K31 oesophagus/stomach: any | Outpatient diagnoses (National Patient Register), ICD-10 K20-K31 (oesophagus, stomach & duodenum): at least one record | Outpatient diagnoses |
| `ov_K20-K31_exceeds_median` | OP K20-K31 oesophagus/stomach: >med | Outpatient diagnoses (National Patient Register), ICD-10 K20-K31 (oesophagus, stomach & duodenum): household count above the training-fold median | Outpatient diagnoses |
| `ov_K20-K31_exceeds_p75` | OP K20-K31 oesophagus/stomach: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 K20-K31 (oesophagus, stomach & duodenum): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_K40-K46_occurred` | OP K40-K46 hernia: any | Outpatient diagnoses (National Patient Register), ICD-10 K40-K46 (hernia): at least one record | Outpatient diagnoses |
| `ov_K40-K46_exceeds_median` | OP K40-K46 hernia: >med | Outpatient diagnoses (National Patient Register), ICD-10 K40-K46 (hernia): household count above the training-fold median | Outpatient diagnoses |
| `ov_K40-K46_exceeds_p75` | OP K40-K46 hernia: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 K40-K46 (hernia): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_K55-K64_occurred` | OP K55-K64 other intestinal: any | Outpatient diagnoses (National Patient Register), ICD-10 K55-K64 (other diseases of intestines): at least one record | Outpatient diagnoses |
| `ov_K55-K64_exceeds_median` | OP K55-K64 other intestinal: >med | Outpatient diagnoses (National Patient Register), ICD-10 K55-K64 (other diseases of intestines): household count above the training-fold median | Outpatient diagnoses |
| `ov_K55-K64_exceeds_p75` | OP K55-K64 other intestinal: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 K55-K64 (other diseases of intestines): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_L60-L75_occurred` | OP L60-L75 skin appendages: any | Outpatient diagnoses (National Patient Register), ICD-10 L60-L75 (disorders of skin appendages): at least one record | Outpatient diagnoses |
| `ov_L60-L75_exceeds_median` | OP L60-L75 skin appendages: >med | Outpatient diagnoses (National Patient Register), ICD-10 L60-L75 (disorders of skin appendages): household count above the training-fold median | Outpatient diagnoses |
| `ov_L60-L75_exceeds_p75` | OP L60-L75 skin appendages: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 L60-L75 (disorders of skin appendages): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_L80-L99_occurred` | OP L80-L99 other skin: any | Outpatient diagnoses (National Patient Register), ICD-10 L80-L99 (other skin disorders): at least one record | Outpatient diagnoses |
| `ov_L80-L99_exceeds_median` | OP L80-L99 other skin: >med | Outpatient diagnoses (National Patient Register), ICD-10 L80-L99 (other skin disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_L80-L99_exceeds_p75` | OP L80-L99 other skin: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 L80-L99 (other skin disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_M15-M19_occurred` | OP M15-M19 arthrosis: any | Outpatient diagnoses (National Patient Register), ICD-10 M15-M19 (arthrosis): at least one record | Outpatient diagnoses |
| `ov_M15-M19_exceeds_median` | OP M15-M19 arthrosis: >med | Outpatient diagnoses (National Patient Register), ICD-10 M15-M19 (arthrosis): household count above the training-fold median | Outpatient diagnoses |
| `ov_M15-M19_exceeds_p75` | OP M15-M19 arthrosis: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 M15-M19 (arthrosis): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_M20-M25_occurred` | OP M20-M25 other joint: any | Outpatient diagnoses (National Patient Register), ICD-10 M20-M25 (other joint disorders): at least one record | Outpatient diagnoses |
| `ov_M20-M25_exceeds_median` | OP M20-M25 other joint: >med | Outpatient diagnoses (National Patient Register), ICD-10 M20-M25 (other joint disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_M20-M25_exceeds_p75` | OP M20-M25 other joint: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 M20-M25 (other joint disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_M50-M54_occurred` | OP M50-M54 back/neck: any | Outpatient diagnoses (National Patient Register), ICD-10 M50-M54 (other dorsopathies (incl. back pain)): at least one record | Outpatient diagnoses |
| `ov_M50-M54_exceeds_median` | OP M50-M54 back/neck: >med | Outpatient diagnoses (National Patient Register), ICD-10 M50-M54 (other dorsopathies (incl. back pain)): household count above the training-fold median | Outpatient diagnoses |
| `ov_M50-M54_exceeds_p75` | OP M50-M54 back/neck: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 M50-M54 (other dorsopathies (incl. back pain)): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_M70-M79_occurred` | OP M70-M79 soft tissue: any | Outpatient diagnoses (National Patient Register), ICD-10 M70-M79 (other soft-tissue disorders): at least one record | Outpatient diagnoses |
| `ov_M70-M79_exceeds_median` | OP M70-M79 soft tissue: >med | Outpatient diagnoses (National Patient Register), ICD-10 M70-M79 (other soft-tissue disorders): household count above the training-fold median | Outpatient diagnoses |
| `ov_M70-M79_exceeds_p75` | OP M70-M79 soft tissue: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 M70-M79 (other soft-tissue disorders): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_N40-N51_occurred` | OP N40-N51 male genital: any | Outpatient diagnoses (National Patient Register), ICD-10 N40-N51 (male genital organs): at least one record | Outpatient diagnoses |
| `ov_N40-N51_exceeds_median` | OP N40-N51 male genital: >med | Outpatient diagnoses (National Patient Register), ICD-10 N40-N51 (male genital organs): household count above the training-fold median | Outpatient diagnoses |
| `ov_N40-N51_exceeds_p75` | OP N40-N51 male genital: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 N40-N51 (male genital organs): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_R00-R09_occurred` | OP R00-R09 circ./resp. symptoms: any | Outpatient diagnoses (National Patient Register), ICD-10 R00-R09 (symptoms: circulatory & respiratory): at least one record | Outpatient diagnoses |
| `ov_R00-R09_exceeds_median` | OP R00-R09 circ./resp. symptoms: >med | Outpatient diagnoses (National Patient Register), ICD-10 R00-R09 (symptoms: circulatory & respiratory): household count above the training-fold median | Outpatient diagnoses |
| `ov_R00-R09_exceeds_p75` | OP R00-R09 circ./resp. symptoms: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 R00-R09 (symptoms: circulatory & respiratory): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_R10-R19_occurred` | OP R10-R19 abdominal symptoms: any | Outpatient diagnoses (National Patient Register), ICD-10 R10-R19 (symptoms: digestive & abdomen): at least one record | Outpatient diagnoses |
| `ov_R10-R19_exceeds_median` | OP R10-R19 abdominal symptoms: >med | Outpatient diagnoses (National Patient Register), ICD-10 R10-R19 (symptoms: digestive & abdomen): household count above the training-fold median | Outpatient diagnoses |
| `ov_R10-R19_exceeds_p75` | OP R10-R19 abdominal symptoms: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 R10-R19 (symptoms: digestive & abdomen): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_R20-R23_occurred` | OP R20-R23 skin symptoms: any | Outpatient diagnoses (National Patient Register), ICD-10 R20-R23 (symptoms: skin): at least one record | Outpatient diagnoses |
| `ov_R20-R23_exceeds_median` | OP R20-R23 skin symptoms: >med | Outpatient diagnoses (National Patient Register), ICD-10 R20-R23 (symptoms: skin): household count above the training-fold median | Outpatient diagnoses |
| `ov_R20-R23_exceeds_p75` | OP R20-R23 skin symptoms: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 R20-R23 (symptoms: skin): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_R30-R39_occurred` | OP R30-R39 urinary symptoms: any | Outpatient diagnoses (National Patient Register), ICD-10 R30-R39 (symptoms: urinary): at least one record | Outpatient diagnoses |
| `ov_R30-R39_exceeds_median` | OP R30-R39 urinary symptoms: >med | Outpatient diagnoses (National Patient Register), ICD-10 R30-R39 (symptoms: urinary): household count above the training-fold median | Outpatient diagnoses |
| `ov_R30-R39_exceeds_p75` | OP R30-R39 urinary symptoms: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 R30-R39 (symptoms: urinary): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_R50-R69_occurred` | OP R50-R69 general symptoms: any | Outpatient diagnoses (National Patient Register), ICD-10 R50-R69 (general symptoms & signs): at least one record | Outpatient diagnoses |
| `ov_R50-R69_exceeds_median` | OP R50-R69 general symptoms: >med | Outpatient diagnoses (National Patient Register), ICD-10 R50-R69 (general symptoms & signs): household count above the training-fold median | Outpatient diagnoses |
| `ov_R50-R69_exceeds_p75` | OP R50-R69 general symptoms: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 R50-R69 (general symptoms & signs): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_S40-S49_occurred` | OP S40-S49 shoulder injury: any | Outpatient diagnoses (National Patient Register), ICD-10 S40-S49 (injury: shoulder & upper arm): at least one record | Outpatient diagnoses |
| `ov_S40-S49_exceeds_median` | OP S40-S49 shoulder injury: >med | Outpatient diagnoses (National Patient Register), ICD-10 S40-S49 (injury: shoulder & upper arm): household count above the training-fold median | Outpatient diagnoses |
| `ov_S40-S49_exceeds_p75` | OP S40-S49 shoulder injury: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 S40-S49 (injury: shoulder & upper arm): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_S50-S59_occurred` | OP S50-S59 forearm injury: any | Outpatient diagnoses (National Patient Register), ICD-10 S50-S59 (injury: elbow & forearm): at least one record | Outpatient diagnoses |
| `ov_S50-S59_exceeds_median` | OP S50-S59 forearm injury: >med | Outpatient diagnoses (National Patient Register), ICD-10 S50-S59 (injury: elbow & forearm): household count above the training-fold median | Outpatient diagnoses |
| `ov_S50-S59_exceeds_p75` | OP S50-S59 forearm injury: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 S50-S59 (injury: elbow & forearm): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_S60-S69_occurred` | OP S60-S69 hand injury: any | Outpatient diagnoses (National Patient Register), ICD-10 S60-S69 (injury: wrist & hand): at least one record | Outpatient diagnoses |
| `ov_S60-S69_exceeds_median` | OP S60-S69 hand injury: >med | Outpatient diagnoses (National Patient Register), ICD-10 S60-S69 (injury: wrist & hand): household count above the training-fold median | Outpatient diagnoses |
| `ov_S60-S69_exceeds_p75` | OP S60-S69 hand injury: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 S60-S69 (injury: wrist & hand): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_S80-S89_occurred` | OP S80-S89 knee/leg injury: any | Outpatient diagnoses (National Patient Register), ICD-10 S80-S89 (injury: knee & lower leg): at least one record | Outpatient diagnoses |
| `ov_S80-S89_exceeds_median` | OP S80-S89 knee/leg injury: >med | Outpatient diagnoses (National Patient Register), ICD-10 S80-S89 (injury: knee & lower leg): household count above the training-fold median | Outpatient diagnoses |
| `ov_S80-S89_exceeds_p75` | OP S80-S89 knee/leg injury: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 S80-S89 (injury: knee & lower leg): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_S90-S99_occurred` | OP S90-S99 ankle/foot injury: any | Outpatient diagnoses (National Patient Register), ICD-10 S90-S99 (injury: ankle & foot): at least one record | Outpatient diagnoses |
| `ov_S90-S99_exceeds_median` | OP S90-S99 ankle/foot injury: >med | Outpatient diagnoses (National Patient Register), ICD-10 S90-S99 (injury: ankle & foot): household count above the training-fold median | Outpatient diagnoses |
| `ov_S90-S99_exceeds_p75` | OP S90-S99 ankle/foot injury: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 S90-S99 (injury: ankle & foot): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_Z00-Z13_occurred` | OP Z00-Z13 examination: any | Outpatient diagnoses (National Patient Register), ICD-10 Z00-Z13 (examination & investigation): at least one record | Outpatient diagnoses |
| `ov_Z00-Z13_exceeds_median` | OP Z00-Z13 examination: >med | Outpatient diagnoses (National Patient Register), ICD-10 Z00-Z13 (examination & investigation): household count above the training-fold median | Outpatient diagnoses |
| `ov_Z00-Z13_exceeds_p75` | OP Z00-Z13 examination: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 Z00-Z13 (examination & investigation): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_Z40-Z54_occurred` | OP Z40-Z54 procedures/aftercare: any | Outpatient diagnoses (National Patient Register), ICD-10 Z40-Z54 (specific procedures & aftercare): at least one record | Outpatient diagnoses |
| `ov_Z40-Z54_exceeds_median` | OP Z40-Z54 procedures/aftercare: >med | Outpatient diagnoses (National Patient Register), ICD-10 Z40-Z54 (specific procedures & aftercare): household count above the training-fold median | Outpatient diagnoses |
| `ov_Z40-Z54_exceeds_p75` | OP Z40-Z54 procedures/aftercare: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 Z40-Z54 (specific procedures & aftercare): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `ov_Z70-Z76_occurred` | OP Z70-Z76 other contact: any | Outpatient diagnoses (National Patient Register), ICD-10 Z70-Z76 (health services, other circumstances): at least one record | Outpatient diagnoses |
| `ov_Z70-Z76_exceeds_median` | OP Z70-Z76 other contact: >med | Outpatient diagnoses (National Patient Register), ICD-10 Z70-Z76 (health services, other circumstances): household count above the training-fold median | Outpatient diagnoses |
| `ov_Z70-Z76_exceeds_p75` | OP Z70-Z76 other contact: >P75 | Outpatient diagnoses (National Patient Register), ICD-10 Z70-Z76 (health services, other circumstances): household count above the training-fold 75th percentile | Outpatient diagnoses |
| `index_cases_count_std` | No. of index cases (z) | Number of index cases in the household; standardised (z-score) | Household-level |
| `mean_age_2020_std` | Age, mean (z) | Mean age of household members (2020); standardised (z-score) | Household-level |
| `max_age_2020_std` | Age, max (z) | Age of the oldest member; standardised (z-score) | Household-level |
| `min_age_2020_std` | Age, min (z) | Age of the youngest member; standardised (z-score) | Household-level |
| `age_variance_std` | Age, variance (z) | Variance of member ages; standardised (z-score) | Household-level |
| `age_IQR_std` | Age, IQR (z) | Interquartile range of member ages; standardised (z-score) | Household-level |
| `prop_foreign_background_std` | Share foreign background (z) | Share of members with foreign background; standardised (z-score) | Household-level |
| `Fodelseland_diversity_std` | Birth-country diversity (z) | Number of distinct birth countries in household; standardised (z-score) | Household-level |
| `Boarea_Person_std` | Living area per person (z) | Living area per person (m2); standardised (z-score) | Household-level |
| `total_Boarea_std` | Total living area (z) | Household total living area (area per person x household size); standardised (z-score) | Household-level |
| `crowding_index_std` | Crowding index (z) | Household size / living area per person; standardised (z-score) | Household-level |
| `mean_DispInk04_std` | Individual income, mean (z) | mean of members' individual disposable income (SEK); standardised (z-score) | Household-level |
| `max_DispInk04_std` | Individual income, max (z) | max of members' individual disposable income (SEK); standardised (z-score) | Household-level |
| `min_DispInk04_std` | Individual income, min (z) | min of members' individual disposable income (SEK); standardised (z-score) | Household-level |
| `sd_DispInk04_std` | Individual income, SD (z) | SD of members' individual disposable income (SEK); standardised (z-score) | Household-level |
| `mean_DispInkFam04_std` | Family income, mean (z) | mean of members' family disposable income (SEK); standardised (z-score) | Household-level |
| `max_DispInkFam04_std` | Family income, max (z) | max of members' family disposable income (SEK); standardised (z-score) | Household-level |
| `min_DispInkFam04_std` | Family income, min (z) | min of members' family disposable income (SEK); standardised (z-score) | Household-level |
| `sd_DispInkFam04_std` | Family income, SD (z) | SD of members' family disposable income (SEK); standardised (z-score) | Household-level |
| `TRYGG_1_per_elderly_std` | Safety-alarm records/65+ (z) | Security-alarm records per member aged 65+; standardised (z-score) | Household-level |
| `Fodelseland_mode_freq_std` | Birth country (frequency) (z) | Most common birth country in household, frequency-encoded; standardised (z-score) | Household-level |
| `AntalBarnUnder18_log_std` | Children <18 (register) (log, z) | Number of children under 18 registered to the household; log1p-transformed, standardised (z-score) | Household-level |
| `Boarea_Person_log_std` | Living area per person (log, z) | Living area per person (m2); log1p-transformed, standardised (z-score) | Household-level |
| `total_Boarea_log_std` | Total living area (log, z) | Household total living area (area per person x household size); log1p-transformed, standardised (z-score) | Household-level |
| `mean_DispInk04_log_std` | Individual income, mean (log, z) | mean of members' individual disposable income (SEK); log1p-transformed, standardised (z-score) | Household-level |
| `mean_DispInkFam04_log_std` | Family income, mean (log, z) | mean of members' family disposable income (SEK); log1p-transformed, standardised (z-score) | Household-level |

