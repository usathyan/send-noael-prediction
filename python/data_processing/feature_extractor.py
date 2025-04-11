import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import re

# Import necessary ML preprocessing tools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO, # Keep INFO for general steps
    format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
)
# Get a specific logger instance for this module if desired, or use root logger
logger = logging.getLogger(__name__)
# Set higher level for debugging messages if needed (can be controlled by env var later)
# logger.setLevel(logging.DEBUG) 

# Removed previous merge/statistical functions as we need text output

def _summarize_findings(findings: Dict[str, pd.DataFrame], exposure: Optional[pd.DataFrame], demographics: Optional[pd.DataFrame]) -> List[str]:
    """Helper function to create a basic summary of findings data."""
    summary_lines = []
    present_domains = [d.upper() for d in findings if findings[d] is not None and not findings[d].empty]
    summary_lines.append("Findings Summary:")
    if not present_domains:
        summary_lines.append("- No significant findings data parsed.")
        return summary_lines

    summary_lines.append(f"- Data domains present: {', '.join(present_domains)}")

    # Link exposure/dose to findings (requires merging - simplified here)
    # Ideally merge demographics/exposure to findings based on USUBJID
    # For a simple example, let's look at LB data if present
    lb_df = findings.get('lb')
    if lb_df is not None and not lb_df.empty and exposure is not None and demographics is not None:
        summary_lines.append("  Lab Findings (LB - Basic Overview):")
        # Attempt to merge to get dose info per lab result
        try:
            # Use only essential columns for merging
            demo_subset = demographics[['USUBJID', 'ACTARM']] if 'ACTARM' in demographics.columns else demographics[['USUBJID']]
            exp_subset = exposure[['USUBJID', 'EXDOSE', 'EXDOSU', 'EXTRT']].drop_duplicates(subset=['USUBJID'], keep='first') # Simplification
            
            lb_merged = pd.merge(lb_df, demo_subset, on='USUBJID', how='left')
            lb_merged = pd.merge(lb_merged, exp_subset, on='USUBJID', how='left')

            # List unique tests present
            unique_tests = lb_merged['LBTESTCD'].unique()
            summary_lines.append(f"  - Tests measured: {', '.join(unique_tests[:15])}{'...' if len(unique_tests) > 15 else ''}")

            # Example: Show mean of numeric results for highest dose group (VERY SIMPLIFIED)
            if 'EXDOSE' in lb_merged.columns:
                 # Ensure EXDOSE is numeric
                 lb_merged['EXDOSE_num'] = pd.to_numeric(lb_merged['EXDOSE'], errors='coerce')
                 highest_dose = lb_merged['EXDOSE_num'].max()
                 if pd.notna(highest_dose):
                     highest_dose_df = lb_merged[lb_merged['EXDOSE_num'] == highest_dose]
                     if not highest_dose_df.empty:
                         # Ensure LBSTRESN is numeric
                         highest_dose_df['LBSTRESN_num'] = pd.to_numeric(highest_dose_df['LBSTRESN'], errors='coerce')
                         mean_vals = highest_dose_df.groupby('LBTESTCD')['LBSTRESN_num'].mean().dropna()
                         if not mean_vals.empty:
                              summary_lines.append(f"  - Mean results at highest dose ({highest_dose} {exp_subset['EXDOSU'].iloc[0]}):")
                              # Extract top N findings
                              for test, mean_val in mean_vals.head(10).items(): # Limit output
                                  summary_lines.append(f"    - {test}: {mean_val:.2f}")
                              if len(mean_vals) > 10:
                                  summary_lines.append("      ...")
            
        except Exception as e:
            logger.warning(f"Could not summarize LB findings: {e}")
            summary_lines.append("  - Could not generate detailed LB summary.")
    
    # Similar summaries could be added for CL, BW, MI etc.
    
    return summary_lines


def encode_categorical(series: pd.Series, col_name: str) -> Tuple[pd.Series, Dict[str, int]]:
    """Encodes a categorical series using LabelEncoder, handling NaNs.
    Returns the encoded series and the class mapping.
    """
    # Fill NaN with a placeholder string before encoding
    # Ensure the placeholder doesn't clash with real data
    placeholder = '__MISSING__'
    encoded_series = series.fillna(placeholder).astype(str)
    le = LabelEncoder()
    encoded_values = le.fit_transform(encoded_series)
    # Create mapping from original value (including placeholder) to encoded integer
    mapping = {cls: int(le.transform([cls])[0]) for cls in le.classes_}
    logger.info(f"Encoded column '{col_name}'. Mapping: {mapping}")
    # Return as a pandas Series with the original index
    return pd.Series(encoded_values, index=series.index), mapping


def generate_study_summary_prompt(parsed_data: Dict[str, Any]) -> Optional[str]:
    """Generates a textual summary of the study suitable for prompting TxGemma.

    Args:
        parsed_data: The dictionary output from domain_parser.parse_study_data.

    Returns:
        A string containing the study summary prompt, or None if essential data is missing.
    """
    logger.info("Generating study summary prompt for TxGemma...")

    demographics = parsed_data.get('demographics')
    exposure = parsed_data.get('exposure')
    trial_summary = parsed_data.get('trial_summary', {})
    findings = parsed_data.get('findings', {})

    # Basic checks for essential data
    if demographics is None or exposure is None:
        logger.error("Cannot generate prompt: Demographics or Exposure data is missing.")
        return None

    prompt_lines = []

    # --- Study Design --- #
    prompt_lines.append("Study Design Summary:")
    study_id = trial_summary.get('STUDYID', 'Unknown Study ID')
    prompt_lines.append(f"- Study ID: {study_id}")
    prompt_lines.append(f"- Study Title: {trial_summary.get('STITLE', 'N/A')}")
    prompt_lines.append(f"- Phase: {trial_summary.get('SPHASE', 'N/A')}")
    prompt_lines.append(f"- Type: {trial_summary.get('STYPE', 'N/A')}")
    prompt_lines.append(f"- Species: {trial_summary.get('SPECIES', 'N/A')}")
    prompt_lines.append(f"- Strain: {trial_summary.get('STRAIN', 'N/A')}")
    prompt_lines.append(f"- Route of Administration: {trial_summary.get('ROUTE', 'N/A')}")
    prompt_lines.append(f"- Study Duration (Planned): {trial_summary.get('TRTDURP', 'N/A')}")
    prompt_lines.append("\n")

    # --- Dosing Information & Control Groups --- #
    prompt_lines.append("Dosing Information:")
    dose_groups = {} # Store info per dose group
    control_treatment = None
    if exposure is not None and not exposure.empty and all(c in exposure.columns for c in ['USUBJID', 'EXTRT', 'EXDOSE', 'EXDOSU']):
        # Ensure EXDOSE is numeric for sorting and identifying control
        exposure['EXDOSE_num'] = pd.to_numeric(exposure['EXDOSE'], errors='coerce')
        # Identify potential control (dose 0 or contains 'vehicle'/'control')
        control_exp = exposure[(exposure['EXDOSE_num'] == 0) | (exposure['EXTRT'].str.contains('vehicle|control', case=False, na=False))]
        if not control_exp.empty:
             control_treatment = control_exp['EXTRT'].iloc[0]
             dose_groups[0.0] = {'treatment': control_treatment, 'unit': control_exp['EXDOSU'].iloc[0], 'is_control': True}
             prompt_lines.append(f"- Control Group: Treatment='{control_treatment}', Dose=0 {dose_groups[0.0]['unit']}")
        
        # Get unique non-control dose levels
        dose_info = exposure[exposure['EXTRT'] != control_treatment][['EXTRT', 'EXDOSE', 'EXDOSU', 'EXDOSE_num']].drop_duplicates().sort_values('EXDOSE_num')
        for _, row in dose_info.iterrows():
            dose_val = row['EXDOSE_num']
            if pd.notna(dose_val) and dose_val != 0.0:
                dose_groups[dose_val] = {'treatment': row['EXTRT'], 'unit': row['EXDOSU'], 'is_control': False}
                prompt_lines.append(f"- Treatment Group: Treatment='{row['EXTRT']}', Dose={row['EXDOSE']} {row['EXDOSU']}")
    else:
        prompt_lines.append("- Exposure data incomplete or missing.")
    prompt_lines.append("\n")
    
    # --- Subject Information --- #
    prompt_lines.append("Subject Information:")
    if demographics is not None and not demographics.empty and 'USUBJID' in demographics.columns and 'SEX' in demographics.columns:
        total_subjects = demographics['USUBJID'].nunique()
        sex_distribution = demographics['SEX'].value_counts().to_dict()
        prompt_lines.append(f"- Total Subjects: {total_subjects}")
        prompt_lines.append(f"- Sex Distribution: {sex_distribution}")
    else:
         prompt_lines.append("- Demographics data incomplete or missing.")
    prompt_lines.append("\n")

    # --- Findings Summary --- #
    findings_summary_lines = _summarize_findings(findings, exposure, demographics)
    prompt_lines.extend(findings_summary_lines)
    prompt_lines.append("\n")

    # --- Final Prompt Instruction --- #
    prompt_lines.append("Based on the comprehensive SEND study data summarized above (including study design, dosing, subjects, and key findings overviews), predict the No Observed Adverse Effect Level (NOAEL). Provide only the final NOAEL value and its units (e.g., '100 mg/kg/day').") # More specific instruction

    final_prompt = "\n".join(prompt_lines)
    logger.info(f"Generated prompt for study {study_id} (length: {len(final_prompt)} chars).")
    # Log the full prompt at DEBUG level for inspection - CHANGED TO PRINT
    # logger.debug(f"\n----- FULL PROMPT for {study_id} -----\n{final_prompt}\n----- END PROMPT for {study_id} -----")
    print(f"\n----- FULL PROMPT for {study_id} -----\n{final_prompt}\n----- END PROMPT for {study_id} -----") # Use print for direct output
    return final_prompt


def _aggregate_lb_features(lb_df: pd.DataFrame, subjects_df: pd.DataFrame, control_dose: Optional[float]) -> pd.DataFrame:
    """Aggregates LB domain data, calculating stats per subject/test."""
    # Ensure required columns exist
    if not all(col in lb_df.columns for col in ['USUBJID', 'LBTESTCD', 'LBSTRESN']):
        logger.error("LB DataFrame missing required columns (USUBJID, LBTESTCD, LBSTRESN). Cannot aggregate.")
        return pd.DataFrame()

    # Filter out non-numeric results before aggregation
    lb_numeric = lb_df.dropna(subset=['LBSTRESN'])
    if lb_numeric.empty:
        logger.warning("No numeric LB results (LBSTRESN) found after dropping NaNs. Cannot aggregate.")
        return pd.DataFrame()

    # Aggregate LB results per subject and test code
    agg_funcs = ['mean', 'max', 'min', 'std']
    try:
        lb_agg = lb_numeric.groupby(['USUBJID', 'LBTESTCD'])['LBSTRESN'].agg(agg_funcs).unstack()
    except Exception as e:
        logger.error(f"Error during LB aggregation: {e}")
        return pd.DataFrame()

    # --- Fix Column Naming ---
    # Flatten MultiIndex columns: (agg_func, LBTESTCD) -> LB_{agg_func}__{LBTESTCD}
    lb_agg.columns = [f"LB_{col[0]}__{col[1]}" for col in lb_agg.columns]
    lb_agg = lb_agg.reindex(subjects_df['USUBJID']) # Ensure all subjects are present

    logger.info(f"Generated {lb_agg.shape[1]} aggregated LB features.")

    # --- Dose Group Comparison (Placeholder - add if needed) ---
    # if control_dose is not None and not pd.isna(control_dose) and 'DOSE_NUM' in subjects_df.columns:
    #     # Add comparison features here, e.g., LB_mean_pct_change_vs_control__{TESTCD}
    #     pass
    # else:
    #     logger.warning("Skipping LB dose-group comparison (control dose undetermined or DOSE_NUM missing).")

    return lb_agg

def _aggregate_bw_features(bw_df: pd.DataFrame, subjects_df: pd.DataFrame, control_dose: Optional[float]) -> pd.DataFrame:
    """Aggregates BW domain data, calculating baseline, terminal, and change features."""
    if not all(col in bw_df.columns for col in ['USUBJID', 'BWSTRESN', 'BWDY']):
         logger.error("BW DataFrame missing required columns (USUBJID, BWSTRESN, BWDY). Cannot aggregate.")
         return pd.DataFrame()

    # Ensure BWDY is numeric
    bw_df = bw_df.dropna(subset=['BWSTRESN', 'BWDY'])
    bw_df['BWDY'] = pd.to_numeric(bw_df['BWDY'], errors='coerce')
    bw_df = bw_df.dropna(subset=['BWDY'])

    if bw_df.empty:
        logger.warning("No valid numeric BW data (BWSTRESN, BWDY) found. Cannot aggregate.")
        return pd.DataFrame()

    # --- Baseline and Terminal BW ---
    baseline_day = bw_df['BWDY'][bw_df['BWDY'] <= 1].max() # Typically day 1 or earliest pre-dose
    terminal_day = bw_df['BWDY'].max()
    if pd.isna(baseline_day): baseline_day = bw_df['BWDY'].min() # Fallback if no day <= 1
    logger.info(f"BW: Using Baseline Day {baseline_day}, Terminal Day {terminal_day}")

    bw_baseline = bw_df[bw_df['BWDY'] == baseline_day].set_index('USUBJID')['BWSTRESN']
    bw_terminal = bw_df[bw_df['BWDY'] == terminal_day].set_index('USUBJID')['BWSTRESN']

    bw_features = pd.DataFrame(index=subjects_df['USUBJID'])
    bw_features['BW_baseline'] = bw_features.index.map(bw_baseline)
    bw_features['BW_terminal'] = bw_features.index.map(bw_terminal)

    # --- BW Change Calculation ---
    bw_df = bw_df.sort_values(by=['USUBJID', 'BWDY'])
    bw_df['BW_pct_change'] = bw_df.groupby('USUBJID')['BWSTRESN'].pct_change() * 100

    # Max % decrease (more robust than min % change)
    bw_pct_change_agg = bw_df.groupby('USUBJID')['BW_pct_change'].agg(['min', 'mean']) # Get min for max decrease
    bw_features['BW_max_pct_decrease'] = -bw_pct_change_agg['min'] # Max decrease is negative of min change
    bw_features['BW_mean_pct_change_overall'] = bw_pct_change_agg['mean'] # Overall mean change

    # --- Dose Group Comparison (Placeholder - add if needed) ---
    # if control_dose is not None and not pd.isna(control_dose) and 'DOSE_NUM' in subjects_df.columns:
    #     # Calculate BW_max_pct_decrease_from_control etc.
    #     # bw_features['BW_max_pct_decrease_from_control'] = ... calculation ...
    #     pass # Placeholder for the complex logic
    # else:
    #     logger.warning("Skipping BW dose-group comparison (control dose undetermined or DOSE_NUM missing).")
        # Ensure column exists if expected by model, even if calculated via fallback
        # if 'BW_max_pct_decrease_from_control' in expected_features: # Check against final expected list
        #     bw_features['BW_max_pct_decrease_from_control'] = np.nan

    return bw_features

def preprocess_for_ml(parsed_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesses parsed SEND data to create features for a traditional ML model.
    Ensures the output matches the exact feature set expected by the model.
    """
    logger.info("Preprocessing data for traditional ML model...")

    # --- Define Expected Features (CRITICAL - Must match trained model) ---
    # Based on the ValueError log provided by the user
    expected_features = [
        'MAX_DOSE', 'SEX_encoded', 'EXTRT_encoded', 'EXROUTE_encoded', 'DOSE_UNIT_encoded',
        'LB_mean__ALB', 'LB_mean__ALBGLOB', 'LB_mean__ALP', 'LB_mean__ALT', 'LB_mean__ANISO',
        'LB_mean__APTT', 'LB_mean__AST', 'LB_mean__BACT', 'LB_mean__BASO', 'LB_mean__BASOLE',
        'LB_mean__BILDIR', 'LB_mean__BILI', 'LB_mean__CA', 'LB_mean__CASTS', 'LB_mean__CHOL',
        'LB_mean__CK', 'LB_mean__CL', 'LB_mean__CLARITY', 'LB_mean__COLOR', 'LB_mean__CREAT',
        'LB_mean__CRYSTALS', 'LB_mean__CSUNCLA', 'LB_mean__CYUNCLA', 'LB_mean__EOS', 'LB_mean__EOSLE',
        'LB_mean__EPIC', 'LB_mean__GGT', 'LB_mean__GLOBUL', 'LB_mean__GLUC', 'LB_mean__HCT',
        'LB_mean__HGB', 'LB_mean__HPOCROM', 'LB_mean__K', 'LB_mean__KETONES', 'LB_mean__LGLUCLE',
        'LB_mean__LGUNSCE', 'LB_mean__LYM', 'LB_mean__LYMLE', 'LB_mean__MCH', 'LB_mean__MCHC',
        'LB_mean__MCV', 'LB_mean__MONO', 'LB_mean__MONOLE', 'LB_mean__NEUT', 'LB_mean__NEUTLE',
        'LB_mean__OCCBLD', 'LB_mean__OTHR', 'LB_mean__PH', 'LB_mean__PHOS', 'LB_mean__PLAT',
        'LB_mean__POIKILO', 'LB_mean__POLYCHR', 'LB_mean__PROT', 'LB_mean__PT', 'LB_mean__RBC',
        'LB_mean__RETI', 'LB_mean__RETIRBC', 'LB_mean__SODIUM', 'LB_mean__SPGRAV', 'LB_mean__TOXGRAN',
        'LB_mean__TRIG', 'LB_mean__UREAN', 'LB_mean__UROBIL', 'LB_mean__VOLUME', 'LB_mean__WBC',
        'LB_max__ALB', 'LB_max__ALBGLOB', 'LB_max__ALP', 'LB_max__ALT', 'LB_max__ANISO',
        'LB_max__APTT', 'LB_max__AST', 'LB_max__BACT', 'LB_max__BASO', 'LB_max__BASOLE',
        'LB_max__BILDIR', 'LB_max__BILI', 'LB_max__CA', 'LB_max__CASTS', 'LB_max__CHOL',
        'LB_max__CK', 'LB_max__CL', 'LB_max__CLARITY', 'LB_max__COLOR', 'LB_max__CREAT',
        'LB_max__CRYSTALS', 'LB_max__CSUNCLA', 'LB_max__CYUNCLA', 'LB_max__EOS', 'LB_max__EOSLE',
        'LB_max__EPIC', 'LB_max__GGT', 'LB_max__GLOBUL', 'LB_max__GLUC', 'LB_max__HCT',
        'LB_max__HGB', 'LB_max__HPOCROM', 'LB_max__K', 'LB_max__KETONES', 'LB_max__LGLUCLE',
        'LB_max__LGUNSCE', 'LB_max__LYM', 'LB_max__LYMLE', 'LB_max__MCH', 'LB_max__MCHC',
        'LB_max__MCV', 'LB_max__MONO', 'LB_max__MONOLE', 'LB_max__NEUT', 'LB_max__NEUTLE',
        'LB_max__OCCBLD', 'LB_max__OTHR', 'LB_max__PH', 'LB_max__PHOS', 'LB_max__PLAT',
        'LB_max__POIKILO', 'LB_max__POLYCHR', 'LB_max__PROT', 'LB_max__PT', 'LB_max__RBC',
        'LB_max__RETI', 'LB_max__RETIRBC', 'LB_max__SODIUM', 'LB_max__SPGRAV', 'LB_max__TOXGRAN',
        'LB_max__TRIG', 'LB_max__UREAN', 'LB_max__UROBIL', 'LB_max__VOLUME', 'LB_max__WBC',
        'TS_PLANNED_DURATION'
        # NOTE: This list assumes the model was trained WITHOUT LB min/std aggregates
        # and WITHOUT BW features. Add them if the model expects them.
        # Add 'LB_min__...', 'LB_std__...' features here if needed
        # Add 'BW_baseline', 'BW_terminal', 'BW_max_pct_decrease', etc. here if needed
    ]


    # --- 1. Extract Basic Study Info & Subject Data ---
    dm_df = parsed_data.get('dm')
    ts_df = parsed_data.get('ts')
    ex_df = parsed_data.get('ex')

    if dm_df is None or dm_df.empty:
        logger.error("DM domain is missing or empty. Cannot proceed.")
        return pd.DataFrame(columns=expected_features) # Return empty df with expected columns

    # Unique subjects
    subjects = dm_df[['USUBJID', 'SEX', 'ARMCD']].drop_duplicates(subset=['USUBJID'])

    # Planned duration from TS
    planned_duration = np.nan # Default to NaN
    if ts_df is not None and not ts_df.empty:
        # Find parameter for planned duration (adjust TSPARMCD if needed)
        duration_param = ts_df[ts_df['TSPARMCD'].isin(['TRTDUR', 'STUDYDAY'])].sort_values('TSVAL', ascending=False) # Try common codes
        if not duration_param.empty and 'TSVAL' in duration_param.columns:
            duration_val = duration_param['TSVAL'].iloc[0]
            try:
                # Attempt numeric conversion first, then regex extraction
                planned_duration = pd.to_numeric(duration_val, errors='coerce')
                if pd.isna(planned_duration):
                    match = re.search(r'(\d+(\.\d+)?)', str(duration_val)) # Look for numbers
                    if match:
                        planned_duration = float(match.group(1))
                if not pd.isna(planned_duration):
                     logger.info(f"Extracted planned duration: {planned_duration}")
                else:
                     logger.warning(f"Could not parse planned duration from TSVAL: {duration_val}")
            except Exception as e:
                logger.warning(f"Error parsing planned duration: {e}")


    # --- 2. Process Dose Information ---
    study_subjects = subjects.set_index('USUBJID') # Set index early
    control_dose = np.nan
    max_dose = np.nan

    # Add columns for dose info, default to NaN
    study_subjects['DOSE_NUM'] = np.nan
    study_subjects['EXDOSU'] = np.nan # Dose Unit
    study_subjects['EXROUTE'] = np.nan # Route
    study_subjects['EXTRT'] = np.nan # Treatment

    if ex_df is not None and not ex_df.empty and 'EXDOSE' in ex_df.columns:
        ex_df_copy = ex_df.copy() # Work on a copy
        ex_df_copy['DOSE_NUM'] = ex_df_copy['EXDOSE'].apply(_extract_numeric_dose)

        # Get first non-NaN dose info per subject
        first_dose_info = ex_df_copy.dropna(subset=['DOSE_NUM']).sort_values('EXSTDY', ascending=True) # Sort by study day
        first_dose_info = first_dose_info.drop_duplicates(subset=['USUBJID'], keep='first')
        first_dose_info = first_dose_info.set_index('USUBJID')

        if not first_dose_info.empty:
            # Update subjects with dose info where available
            study_subjects.update(first_dose_info[['DOSE_NUM', 'EXDOSU', 'EXROUTE', 'EXTRT']])

            # Identify control dose (assuming 0 or lowest numeric dose)
            numeric_doses = study_subjects['DOSE_NUM'].dropna().unique()
            if len(numeric_doses) > 0:
                if 0.0 in numeric_doses:
                    control_dose = 0.0
                else:
                     control_dose = np.min(numeric_doses) # Fallback to lowest dose
                logger.info(f"Identified control dose: {control_dose}")

                # Calculate Max Dose (highest non-control dose)
                non_control_doses = numeric_doses[numeric_doses != control_dose]
                if len(non_control_doses) > 0:
                    max_dose = np.max(non_control_doses)
                    logger.info(f"Identified max non-control dose: {max_dose}")
                else:
                    logger.warning("Only control dose found. Max dose set to control dose.")
                    max_dose = control_dose
            else:
                logger.warning("No valid numeric doses found after extraction.")
        else:
            logger.warning("No subjects with extractable numeric EXDOSE found.")
    else:
        logger.warning("EX domain missing, empty, or lacks EXDOSE. Cannot extract dose information.")

    # Add Max Dose and Planned Duration as single study-wide features
    study_subjects['MAX_DOSE'] = max_dose
    study_subjects['TS_PLANNED_DURATION'] = planned_duration

    # --- 3. Aggregate Time-Series Data (LB, BW) ---
    combined_features = study_subjects # Start with subject/dose info

    # Lab findings (LB)
    lb_df = parsed_data.get('lb')
    if lb_df is not None and not lb_df.empty:
        lb_features = _aggregate_lb_features(lb_df, combined_features.reset_index(), control_dose) # Pass reset index df
        if not lb_features.empty:
            combined_features = combined_features.join(lb_features, how='left')
    else:
        logger.warning("LB domain missing or empty. Skipping LB feature aggregation.")

    # Body weight (BW)
    bw_df = parsed_data.get('bw')
    if bw_df is not None and not bw_df.empty:
        bw_df_filtered = bw_df[bw_df['BWTESTCD'] == 'BW'].copy() if 'BWTESTCD' in bw_df.columns else pd.DataFrame()
        if not bw_df_filtered.empty:
             bw_features = _aggregate_bw_features(bw_df_filtered, combined_features.reset_index(), control_dose) # Pass reset index df
             if not bw_features.empty:
                 combined_features = combined_features.join(bw_features, how='left')
        else:
             logger.warning("No records found with BWTESTCD='BW' or BW domain missing BWTESTCD. Skipping BW feature aggregation.")
    else:
        logger.warning("BW domain missing or empty. Skipping BW feature aggregation.")


    # --- 4. Encode Categorical Features ---
    encoded_cols_map = {}
    categorical_cols = ['SEX', 'ARMCD', 'EXDOSU', 'EXROUTE', 'EXTRT'] # Ensure these are present
    for col in categorical_cols:
        encoded_col_name = f'{col}_encoded'
        if col in combined_features.columns:
            combined_features[encoded_col_name], mapping = encode_categorical(combined_features[col], col)
            encoded_cols_map[col] = mapping
            # combined_features = combined_features.drop(columns=[col]) # Optional: drop original
        else:
            logger.warning(f"Categorical column '{col}' not found. Adding '{encoded_col_name}' filled with NaN.")
            combined_features[encoded_col_name] = np.nan # Add as NaN if missing

    # --- 5. Align with Expected Features ---
    logger.info(f"Shape before aligning to expected features: {combined_features.shape}")
    logger.info(f"Columns available before aligning: {combined_features.columns.tolist()}")

    # Create the final DataFrame with exactly the expected features
    # Use reindex to add missing columns (filled with NaN) and ensure correct order
    try:
        final_features = combined_features.reindex(columns=expected_features)
        logger.info(f"Shape after aligning to expected features: {final_features.shape}")
        if final_features.shape[1] != len(expected_features):
             logger.error("Column count mismatch after reindexing!")
             # Log difference
             available_cols = set(combined_features.columns)
             expected_cols = set(expected_features)
             logger.info(f"Expected not found: {list(expected_cols - available_cols)}")
             logger.info(f"Found but not expected: {list(available_cols - expected_cols)}")


    except Exception as e:
         logger.error(f"Error during reindexing to expected features: {e}")
         # Fallback: return empty df with expected columns? Or try to proceed?
         return pd.DataFrame(columns=expected_features)


    # --- 6. Impute Missing Values ---
    if not final_features.empty:
        numeric_cols = final_features.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            # Check for columns that are all NaN before imputation
            all_nan_cols = numeric_cols[final_features[numeric_cols].isna().all()]
            if not all_nan_cols.empty:
                logger.warning(f"Columns are entirely NaN before imputation: {all_nan_cols.tolist()}. Imputing will fill with mean/median/etc.")

            imputer = SimpleImputer(strategy='mean') # Consider 'median' or 'constant' fill_value=0
            try:
                final_features[numeric_cols] = imputer.fit_transform(final_features[numeric_cols])
                logger.info(f"Imputed missing values in {len(numeric_cols)} numeric features using '{imputer.strategy}'.")
            except Exception as e:
                 logger.error(f"Error during imputation: {e}")
                 # Handle error - maybe return NaNs or the pre-imputation frame?
                 return final_features # Return pre-imputation state on error
        else:
             logger.warning("No numeric columns found in the final feature set to impute.")
    else:
        logger.warning("Feature DataFrame is empty after alignment, skipping imputation.")

    # --- 7. Final Check and Return ---
    logger.info(f"Final feature set shape after imputation: {final_features.shape}")

    # Ensure only one row if expecting study-level prediction (might need aggregation)
    if final_features.shape[0] > 1:
        logger.warning(f"Feature set has {final_features.shape[0]} rows (subjects). Aggregating for study-level prediction (using mean). Adapt if subject-level needed.")
        # Example: Aggregate numeric features by mean. Adapt as needed.
        numeric_cols = final_features.select_dtypes(include=np.number).columns
        final_features = final_features[numeric_cols].mean().to_frame().T
        # Re-add any necessary non-numeric study-level features if required by model


    # Final check for columns before returning
    if not all(col in final_features.columns for col in expected_features):
        logger.error("Final check failed: Missing expected features before returning!")
        missing_final = list(set(expected_features) - set(final_features.columns))
        logger.error(f"Missing: {missing_final}")
        # Add missing columns back as NaN just in case
        for col in missing_final:
             final_features[col] = np.nan
        final_features = final_features.reindex(columns=expected_features) # Try reindexing again


    return final_features[expected_features] # Return only expected features in correct order


# --- Helper to extract numeric dose ---
def _extract_numeric_dose(dose_val: Any) -> Optional[float]:
    """Robustly extracts a numeric dose value from various string formats or numbers."""
    if pd.isna(dose_val):
        return np.nan
    if isinstance(dose_val, (int, float)):
        return float(dose_val)
    try:
        # Try direct conversion first
        return float(dose_val)
    except (ValueError, TypeError):
        try:
            # Try regex for patterns like "10 mg/kg", "5", "0.5 unit" etc.
            # This regex looks for a number (int or float) potentially followed by space and units
            match = re.search(r'^(\d+(\.\d+)?)\b', str(dose_val).strip())
            if match:
                return float(match.group(1))
            else:
                # logger.debug(f"Could not extract numeric dose via regex from: {dose_val}")
                return np.nan
        except Exception as e:
            # logger.warning(f"Exception during regex dose extraction for '{dose_val}': {e}")
            return np.nan

def extract_features(parsed_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Orchestrates the feature extraction pipeline for ML models.

    Args:
        parsed_data: The dictionary output from domain_parser.parse_study_data.

    Returns:
        A DataFrame containing numerical features for the ML model, or None if failed.
    """
    logger.info("Starting feature extraction for ML model...")
    features_df = preprocess_for_ml(parsed_data)
    if features_df is None:
        logger.error("Feature extraction failed.")
        return None

    # NOTE: Scaling/Normalization is often done as part of the ML pipeline (e.g., using StandardScaler)
    #       rather than here, to prevent data leakage if using cross-validation.
    #       Leaving it out here for now.
    # scaler = StandardScaler()
    # scaled_features = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)
    
    logger.info("Finished feature extraction for ML model.")
    return features_df # Return unscaled features for now


# Example Usage (Optional - for testing)
# (Keep example usage similar, but expect a DataFrame output now)
# if __name__ == '__main__':
#     from pathlib import Path
#     import sys
#     sys.path.append(str(Path(__file__).parent.parent)) # Add parent dir to path
#     from data_processing.send_loader import load_send_study
#     from data_processing.domain_parser import parse_study_data
#
#     example_study_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'phuse-scripts' / 'data' / 'send' / 'CBER-POC-Pilot-Study1-Vaccine'
#
#     if example_study_path.exists():
#         print(f"Loading study from: {example_study_path}")
#         loaded_data = load_send_study(example_study_path)
#         if loaded_data:
#             parsed_results = parse_study_data(loaded_data)
#             features = extract_features(parsed_results)
#
#             if features is not None:
#                 print("\n--- Extracted ML Features (Sample) ---")
#                 print(features.head())
#                 print(f"\nFeature shape: {features.shape}")
#             else:
#                 print("ML Feature extraction failed.")
#         else:
#             print("Failed to load study data.")
#     else:
#         print(f"Example study path not found: {example_study_path}") 