import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Import necessary ML preprocessing tools
from sklearn.preprocessing import LabelEncoder
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


def preprocess_for_ml(parsed_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Merges parsed data and engineers basic numerical features for ML models."""
    logger.info("Preprocessing data for traditional ML model...")

    demographics = parsed_data.get('demographics')
    exposure = parsed_data.get('exposure')
    trial_summary = parsed_data.get('trial_summary', {})
    findings = parsed_data.get('findings', {})

    if demographics is None or exposure is None:
        logger.error("Cannot preprocess: Demographics or Exposure data is missing.")
        return None

    # --- 1. Combine Demo and Exposure (simplified: one row per subject, first exposure) ---
    subject_features = demographics.copy()
    exp_subset = exposure[['USUBJID', 'EXTRT', 'EXDOSE', 'EXDOSU', 'EXROUTE']]
    exp_subset['EXDOSE_num'] = pd.to_numeric(exp_subset['EXDOSE'], errors='coerce')
    exp_agg = exp_subset.groupby('USUBJID').agg(
        EXTRT=('EXTRT', 'first'),
        EXROUTE=('EXROUTE', 'first'),
        MAX_DOSE=('EXDOSE_num', 'max'), # Example feature: Max dose subject received
        DOSE_UNIT=('EXDOSU', 'first')
    ).reset_index()
    
    subject_features = pd.merge(subject_features, exp_agg, on='USUBJID', how='left')

    # --- 2. Encode Categorical Subject Features ---
    cat_cols = ['SEX', 'EXTRT', 'EXROUTE', 'DOSE_UNIT'] # Add others like SPECIES, STRAIN from TS if needed
    for col in cat_cols:
        if col in subject_features.columns:
            # Basic imputation for missing categories
            subject_features[col] = subject_features[col].fillna('Missing')
            # Label Encoding (simple approach; OneHotEncoder is often better but creates more cols)
            le = LabelEncoder()
            subject_features[col + '_encoded'] = le.fit_transform(subject_features[col])
        else:
             logger.warning(f"Categorical column '{col}' not found for encoding.")

    # Select key subject-level features (numerical + encoded categoricals)
    subject_feature_cols = ['USUBJID', 'MAX_DOSE'] + [c + '_encoded' for c in cat_cols if c + '_encoded' in subject_features.columns]
    subject_features = subject_features[subject_feature_cols]

    # --- 3. Process Findings (Example: Aggregate LB results) ---
    # This part needs significant expansion for real-world use
    aggregated_findings = pd.DataFrame() # Empty df for now
    lb_df = findings.get('lb')
    if lb_df is not None and not lb_df.empty:
        logger.info("Aggregating basic LB findings...")
        # Ensure numeric result column exists and is numeric
        if 'LBSTRESN' in lb_df.columns:
            lb_df['LBSTRESN_num'] = pd.to_numeric(lb_df['LBSTRESN'], errors='coerce')
            # Simple aggregation: Calculate mean and max value for each test per subject
            lb_agg = lb_df.groupby(['USUBJID', 'LBTESTCD'])['LBSTRESN_num'].agg(['mean', 'max']).unstack()
            lb_agg.columns = ['_ '.join(col).strip() for col in lb_agg.columns.values] # Flatten multi-index
            lb_agg.columns = [f'LB_{c.replace(" ", "_")}' for c in lb_agg.columns] # Prefix columns
            aggregated_findings = lb_agg
            logger.info(f"Generated {aggregated_findings.shape[1]} aggregated LB features.")
        else:
             logger.warning("LB domain missing numeric result column LBSTRESN for aggregation.")
    else:
         logger.info("No LB data found for aggregation.")
         
    # --- 4. Combine Subject Features and Aggregated Findings ---
    if not aggregated_findings.empty:
        final_features = pd.merge(subject_features, aggregated_findings, on='USUBJID', how='left')
    else:
        final_features = subject_features

    # --- 5. Aggregate per Study --- 
    # Current features are per-subject. Need to aggregate to one row per study.
    # Very basic aggregation: take the mean of numeric features across all subjects.
    # This loses a lot of info - more sophisticated aggregation per dose group is better.
    study_level_features = final_features.drop(columns=['USUBJID']).mean(axis=0).to_frame().T
    logger.info(f"Aggregated features to study level, shape: {study_level_features.shape}")

    # --- 6. Handle Missing Values (Imputation) --- 
    # Impute missing values (using mean strategy)
    logger.info(f"Imputing missing values in {study_level_features.shape[1]} features...")
    imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
    final_numeric_features = pd.DataFrame(imputer.fit_transform(study_level_features), columns=study_level_features.columns)

    # --- 7. Add Study-Level Parameters from TS --- 
    # Example: Add planned duration if available
    if 'TRTDURP' in trial_summary:
        try:
            # Extract numeric part of duration (e.g., '28 days' -> 28)
            duration = float(str(trial_summary['TRTDURP']).split()[0])
            final_numeric_features['TS_PLANNED_DURATION'] = duration
        except (ValueError, IndexError):
             logger.warning(f"Could not parse numeric duration from TSPARMCD TRTDURP: {trial_summary['TRTDURP']}")
             final_numeric_features['TS_PLANNED_DURATION'] = np.nan # Impute later if needed
    
    # Impute again if new NaNs were introduced
    final_numeric_features = pd.DataFrame(imputer.fit_transform(final_numeric_features), columns=final_numeric_features.columns)

    logger.info(f"Final feature set shape: {final_numeric_features.shape}")
    return final_numeric_features

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