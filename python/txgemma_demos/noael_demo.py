"""
TxGemma Demo: Automated NOAEL Determination (Simulated)

This module refactors logic from the original manus/AutomatedNOAELDetermination.py
to demonstrate NOAEL determination based on statistical analysis of endpoints,
followed by a *simulated* TxGemma response summarizing the findings.

NOTE: This script does NOT actually call the TxGemma model for the final summary.
It calculates NOAELs based on statistical tests and then formats a simulated response.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# --- Helper Functions (Adapted from original script) ---

def get_dose_groups_from_parsed(parsed_data: Dict[str, pd.DataFrame]) -> Tuple[Optional[Dict[Any, List[str]]], Optional[List[Any]]]:
    """Extracts dose groups and ordered doses from parsed DM and EX data."""
    if 'dm' not in parsed_data or 'ex' not in parsed_data:
        logger.warning("DM or EX data not found in parsed_data. Cannot determine dose groups.")
        return None, None
        
    dm_df = parsed_data['dm']
    ex_df = parsed_data['ex']

    if 'USUBJID' not in dm_df.columns or 'ARMCD' not in dm_df.columns:
         logger.warning("DM DataFrame missing USUBJID or ARMCD.")
         return None, None
    if 'USUBJID' not in ex_df.columns or 'EXDOSE' not in ex_df.columns:
         logger.warning("EX DataFrame missing USUBJID or EXDOSE.")
         return None, None

    # Use SUBJECT level dose if available (more precise than ARMCD)
    subjects = dm_df[['USUBJID', 'ARMCD', 'SEX']].copy() # Add SEX if needed later
    # Ensure EXDOSE is numeric, coercing errors
    ex_df['EXDOSE_numeric'] = pd.to_numeric(ex_df['EXDOSE'], errors='coerce')
    doses = ex_df[['USUBJID', 'EXDOSE_numeric']].dropna(subset=['EXDOSE_numeric']).drop_duplicates()
    
    if doses.empty:
        logger.warning("No valid numeric EXDOSE found. Cannot determine dose groups.")
        return None, None

    subjects = subjects.merge(doses, on='USUBJID', how='left')
    
    # Handle subjects potentially missing dose info (though dropna should prevent this)
    subjects = subjects.dropna(subset=['EXDOSE_numeric']) 

    # Group subjects by dose
    dose_groups = subjects.groupby('EXDOSE_numeric')['USUBJID'].apply(list).to_dict()
    ordered_doses = sorted(dose_groups.keys())
    
    logger.info(f"Identified {len(ordered_doses)} dose groups (numeric doses): {ordered_doses}")
    return dose_groups, ordered_doses


def extract_endpoint_features(parsed_data: Dict[str, pd.DataFrame], dose_groups: Dict[Any, List[str]]) -> Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]]:
    """
    Extracts features (mean, std, n, values) for key endpoints (LB, BW) per dose group.
    """
    logger.info("Extracting endpoint features per dose group...")
    endpoint_features = {'laboratory_tests': {}, 'body_weights': {}}

    # --- Laboratory Tests (LB) ---
    if 'lb' in parsed_data:
        lb_data = parsed_data['lb']
        if 'LBTEST' in lb_data.columns and 'LBSTRESN' in lb_data.columns and 'USUBJID' in lb_data.columns:
             # Ensure LBSTRESN is numeric
            lb_data['LBSTRESN_numeric'] = pd.to_numeric(lb_data['LBSTRESN'], errors='coerce')
            lb_data_valid = lb_data.dropna(subset=['LBSTRESN_numeric'])

            lb_test_features = {}
            for test in lb_data_valid['LBTEST'].unique():
                test_data = lb_data_valid[lb_data_valid['LBTEST'] == test]
                test_stats = {}
                for dose, subjects_in_group in dose_groups.items():
                    dose_data = test_data[test_data['USUBJID'].isin(subjects_in_group)]
                    if not dose_data.empty:
                        values = dose_data['LBSTRESN_numeric'].values
                        if len(values) > 0:
                            test_stats[dose] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'n': len(values),
                                'values': values.tolist()
                            }
                if test_stats: # Only add if data exists for at least one group
                    lb_test_features[test] = test_stats
            endpoint_features['laboratory_tests'] = lb_test_features
            logger.info(f"Extracted features for {len(lb_test_features)} laboratory tests.")
        else:
            logger.warning("LB domain missing required columns (USUBJID, LBTEST, LBSTRESN) or no numeric results.")
            
    # --- Body Weights (BW) ---
    if 'bw' in parsed_data:
        bw_data = parsed_data['bw']
        if 'BWSTRESN' in bw_data.columns and 'USUBJID' in bw_data.columns:
             # Ensure BWSTRESN is numeric
             # Consider filtering by VISITDY if multiple timepoints exist - using last available for now
            bw_data['BWSTRESN_numeric'] = pd.to_numeric(bw_data['BWSTRESN'], errors='coerce')
            bw_data_valid = bw_data.dropna(subset=['BWSTRESN_numeric'])
            # Simple approach: Use the last recorded weight per subject if multiple exist
            bw_data_last = bw_data_valid.loc[bw_data_valid.groupby('USUBJID')['VISITDY'].idxmax()]


            bw_dose_stats = {}
            for dose, subjects_in_group in dose_groups.items():
                dose_data = bw_data_last[bw_data_last['USUBJID'].isin(subjects_in_group)]
                if not dose_data.empty:
                    values = dose_data['BWSTRESN_numeric'].values
                    if len(values) > 0:
                         bw_dose_stats[dose] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'n': len(values),
                            'values': values.tolist()
                        }
            if bw_dose_stats:
                endpoint_features['body_weights'] = {'BW': bw_dose_stats} # Structure similar to LB
                logger.info("Extracted features for Body Weight.")
        else:
             logger.warning("BW domain missing required columns (USUBJID, BWSTRESN) or no numeric results.")

    return endpoint_features

def analyze_dose_response(endpoint_features: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]], 
                          ordered_doses: List[Any], 
                          control_dose: Any = 0) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Performs statistical tests (ANOVA, t-tests) to compare dose groups against control.
    Assumes control_dose is the lowest dose in ordered_doses if not specified or found.
    """
    logger.info("Analyzing dose-response using statistical tests...")
    analysis_results = {}
    
    if not ordered_doses:
        logger.error("No ordered doses provided for analysis.")
        return analysis_results

    # Determine control group dose
    actual_control_dose = control_dose if control_dose in ordered_doses else ordered_doses[0]
    logger.info(f"Using dose {actual_control_dose} as control group.")

    for domain, tests in endpoint_features.items():
        domain_results = {}
        for test_name, dose_data in tests.items():
            test_results = {'significant_changes': {}, 'stats': {}}
            
            control_values = dose_data.get(actual_control_dose, {}).get('values', [])
            
            if not control_values:
                logger.warning(f"No control group data found for {domain} - {test_name}. Skipping analysis.")
                continue
                
            # Perform ANOVA across all groups if multiple groups exist
            all_group_values = [group_data.get('values', []) for dose, group_data in dose_data.items() if len(group_data.get('values', [])) > 0]
            
            if len(all_group_values) >= 2:
                 # Ensure all groups have sufficient data points for ANOVA
                 min_points_anova = 2 # Typically need at least 2 per group
                 valid_groups_anova = [g for g in all_group_values if len(g) >= min_points_anova]
                 
                 if len(valid_groups_anova) >= 2: # Need at least two valid groups
                    try:
                        f_val, p_anova = stats.f_oneway(*valid_groups_anova)
                        test_results['stats']['anova_p_value'] = p_anova
                    except ValueError as e:
                         logger.warning(f"ANOVA failed for {domain} - {test_name}: {e}")
                         test_results['stats']['anova_p_value'] = None
                 else:
                    logger.warning(f"Insufficient valid groups for ANOVA for {domain} - {test_name}. Skipping ANOVA.")
                    test_results['stats']['anova_p_value'] = None


            # Perform t-tests comparing each treated group to control
            for dose in ordered_doses:
                if dose == actual_control_dose:
                    continue
                
                treated_values = dose_data.get(dose, {}).get('values', [])
                
                if not treated_values:
                    logger.warning(f"No data for dose {dose} in {domain} - {test_name}. Skipping t-test.")
                    continue
                
                # Ensure sufficient data for t-test (e.g., at least 2 points each)
                if len(control_values) >= 2 and len(treated_values) >= 2:
                    try:
                         # Welch's t-test (assumes unequal variances by default)
                        t_stat, p_ttest = stats.ttest_ind(control_values, treated_values, equal_var=False) 
                        
                        if p_ttest < 0.05: # Significance threshold
                            mean_control = dose_data[actual_control_dose]['mean']
                            mean_treated = dose_data[dose]['mean']
                            percent_change = ((mean_treated - mean_control) / mean_control) * 100 if mean_control != 0 else 0
                            
                            test_results['significant_changes'][dose] = {
                                'p_value': p_ttest,
                                'percent_change': percent_change,
                                'direction': 'increase' if mean_treated > mean_control else 'decrease'
                            }
                    except ValueError as e:
                        logger.warning(f"T-test failed for {domain} - {test_name} (Dose {dose} vs Control): {e}")

                else:
                     logger.warning(f"Insufficient data points for t-test between control and dose {dose} for {domain} - {test_name}. Skipping.")

            if test_results['significant_changes'] or test_results['stats']: # Only add if something was calculated
                domain_results[test_name] = test_results
        
        if domain_results: # Only add domain if it has results
             analysis_results[domain] = domain_results

    logger.info("Finished statistical analysis.")
    return analysis_results


def determine_per_endpoint_noael(analysis_results: Dict[str, Dict[str, Dict[str, Any]]], 
                                 ordered_doses: List[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Determines the NOAEL for each endpoint based on the statistical analysis results.
    NOAEL is the highest dose tested *before* the LOAEL (Lowest Observed Adverse Effect Level).
    """
    logger.info("Determining NOAEL for each endpoint...")
    noael_results = {}

    if not ordered_doses or len(ordered_doses) == 0:
        logger.error("Cannot determine NOAEL without ordered doses.")
        return noael_results
        
    for domain, tests in analysis_results.items():
        domain_noaels = {}
        for test_name, test_results in tests.items():
            significant_changes = test_results.get('significant_changes', {})
            
            if not significant_changes:
                # No significant changes observed, NOAEL is the highest dose tested
                noael = ordered_doses[-1]
                loael = None
            else:
                # Find the lowest dose with a significant change (LOAEL)
                significant_doses = sorted([dose for dose in significant_changes.keys() if dose in ordered_doses])
                
                if not significant_doses:
                     # This case should ideally not happen if significant_changes is populated correctly
                     logger.warning(f"Significant changes reported for {domain}-{test_name}, but no valid doses found. Setting NOAEL to highest dose.")
                     noael = ordered_doses[-1]
                     loael = None
                else:
                    loael = significant_doses[0]
                    loael_index = ordered_doses.index(loael)
                    
                    if loael_index == 0:
                        # LOAEL is the lowest dose (or control), implies effect even at lowest tested dose
                        noael = None # Cannot determine NOAEL
                    else:
                        # NOAEL is the dose immediately preceding the LOAEL
                        noael = ordered_doses[loael_index - 1]

            domain_noaels[test_name] = {'noael': noael, 'loael': loael}
        
        if domain_noaels:
            noael_results[domain] = domain_noaels

    logger.info("Finished per-endpoint NOAEL determination.")
    return noael_results


def get_overall_noael(per_endpoint_noael: Dict[str, Dict[str, Any]]) -> Optional[Any]:
    """Determines the overall study NOAEL based on the minimum determined endpoint NOAEL."""
    min_noael = None
    noael_found = False

    for domain, tests in per_endpoint_noael.items():
        for test_name, result in tests.items():
            noael = result.get('noael')
            if noael is not None: # Consider only determined NOAELs
                if not noael_found or noael < min_noael:
                    min_noael = noael
                    noael_found = True
            # If any endpoint has noael=None (effect at lowest dose), overall NOAEL is undetermined
            elif noael is None and result.get('loael') is not None:
                 logger.warning(f"Effect observed at lowest dose for {domain}-{test_name}. Overall NOAEL is undetermined.")
                 return None 

    return min_noael


def prepare_summary_prompt(dose_groups: Dict[Any, List[str]],
                             ordered_doses: List[Any],
                             endpoint_features: Dict[str, Dict[str, Dict[Any, Dict[str, Any]]]],
                             analysis_results: Dict[str, Dict[str, Dict[str, Any]]],
                             per_endpoint_noael: Dict[str, Dict[str, Any]],
                             overall_noael: Optional[Any],
                             dose_units: str = "mg/kg/day") -> str:
    """Prepares a detailed text prompt summarizing the study findings."""
    logger.info("Preparing summary prompt for simulated response...")
    
    lines = []
    lines.append("Toxicology Study Summary and NOAEL Determination:")
    lines.append("")
    lines.append("Study Design:")
    lines.append(f"- Dose Levels Tested: {', '.join(map(str, ordered_doses))} {dose_units}")
    lines.append(f"- Control Group Dose: {ordered_doses[0]} {dose_units}")
    # Could add species, route etc. from parsed_data['ts'] if needed

    lines.append("")
    lines.append("Key Findings (Statistically Significant Changes vs. Control):")
    findings_summary = []
    for domain, tests in analysis_results.items():
        for test_name, results in tests.items():
            changes = results.get('significant_changes', {})
            if changes:
                for dose, change_details in sorted(changes.items()):
                    findings_summary.append(
                        f"- {domain.replace('_',' ').title()} - {test_name}: "
                        f"{change_details['direction']} observed at {dose} {dose_units} "
                        f"(p={change_details['p_value']:.3f}, {change_details['percent_change']:.1f}% change)."
                    )
                    
    if not findings_summary:
        lines.append("- No statistically significant changes observed in evaluated endpoints (Lab Tests, Body Weight).")
    else:
        lines.extend(findings_summary)

    lines.append("")
    lines.append("Per-Endpoint NOAEL Assessment:")
    noael_summary = []
    for domain, tests in per_endpoint_noael.items():
        for test_name, result in tests.items():
            noael_val = result['noael']
            loael_val = result['loael']
            noael_str = f"{noael_val} {dose_units}" if noael_val is not None else "Undetermined"
            loael_str = f"LOAEL at {loael_val} {dose_units}" if loael_val is not None else "No LOAEL identified"
            noael_summary.append(f"- {domain.replace('_',' ').title()} - {test_name}: NOAEL = {noael_str} ({loael_str}).")
    
    if not noael_summary:
         lines.append("- NOAEL assessment could not be performed for evaluated endpoints.")
    else:
         lines.extend(noael_summary)

    lines.append("")
    lines.append("Overall Study NOAEL Determination:")
    if overall_noael is not None:
        lines.append(f"- Based on the most sensitive endpoint, the overall study NOAEL is determined to be {overall_noael} {dose_units}.")
    else:
        lines.append("- The overall study NOAEL could not be determined, likely due to effects observed at the lowest tested dose for one or more endpoints.")
        
    return "\n".join(lines)

def simulate_txgemma_response(prompt_summary: str, overall_noael: Optional[Any], dose_units: str = "mg/kg/day") -> str:
    """
    Simulates an ideal TxGemma response based on the calculated results.
    Does NOT actually call the model.
    """
    logger.info("Simulating TxGemma response...")
    
    response = "## Toxicology Study Analysis Report\n\n"
    response += f"**Input Summary Provided:**\n```\n{prompt_summary}\n```\n\n"
    response += "**Automated Analysis and Interpretation:**\n\n"
    
    response += "Based on the provided study data summary and statistical analysis of key endpoints (including laboratory tests and body weight), the following conclusions are drawn:\n\n"
    
    # Extract key findings from the prompt (simple example)
    key_findings_lines = [line for line in prompt_summary.split('\n') if line.strip().startswith('-') and ':' in line and ('observed at' in line or 'No statistically' in line)]
    if key_findings_lines:
         response += "*   **Key Dose-Response Effects:** Significant changes compared to control were noted for specific endpoints at various dose levels, as detailed in the input summary.\n" # Could be more specific if needed
    else:
         response += "*   **Key Dose-Response Effects:** No statistically significant treatment-related effects were identified in the evaluated endpoints.\n"

    # Reference per-endpoint NOAELs
    response += "*   **Endpoint Sensitivity:** The NOAEL varied across different endpoints, indicating differing sensitivity to the test article. Refer to the per-endpoint assessment in the input summary for details.\n"
    
    # State overall NOAEL
    if overall_noael is not None:
        response += f"*   **Overall NOAEL:** Considering the most sensitive endpoint demonstrating a statistically significant effect, the overall study NOAEL is established at **{overall_noael} {dose_units}**. This represents the highest dose level at which no statistically significant adverse effects were observed compared to the control group for the endpoints evaluated.\n"
    else:
        response += "*   **Overall NOAEL:** An overall study NOAEL could not be determined from the provided analysis. This may be due to statistically significant effects observed even at the lowest tested dose for critical endpoints, indicating the NOAEL is below the lowest dose administered in this study.\n"
        
    response += "\n**Disclaimer:** This automated analysis is based on statistical evaluation of the provided data summary. Final NOAEL determination should be confirmed by expert toxicological review considering biological relevance and other study factors."
    
    return response


# --- Main Orchestrating Function ---

def run_noael_determination_demo(parsed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Runs the NOAEL determination demo pipeline using parsed SEND data.

    Args:
        parsed_data: Dictionary of DataFrames, output from domain_parser.parse_study_data.

    Returns:
        A dictionary containing demo results:
        - 'dose_groups': Dictionary mapping dose level to list of USUBJIDs.
        - 'ordered_doses': Sorted list of numerical dose levels.
        - 'endpoint_features': Extracted features (mean, std, n) per endpoint per dose.
        - 'analysis_results': Statistical comparison results (p-values, % change).
        - 'per_endpoint_noael': Determined NOAEL/LOAEL for each endpoint.
        - 'overall_noael': The overall study NOAEL based on the most sensitive endpoint.
        - 'summary_prompt': The detailed text prompt generated for the simulation.
        - 'simulated_response': The simulated TxGemma text response.
        - 'error': Error message if pipeline fails at any step.
    """
    results = {"error": None}
    try:
        logger.info("Starting TxGemma NOAEL Determination Demo Pipeline...")

        # 1. Get Dose Groups
        dose_groups, ordered_doses = get_dose_groups_from_parsed(parsed_data)
        if dose_groups is None or ordered_doses is None:
            results["error"] = "Failed to determine dose groups from parsed data."
            logger.error(results["error"])
            return results
        results['dose_groups'] = dose_groups
        results['ordered_doses'] = ordered_doses
        
        # Assume dose units from EX domain if possible, otherwise default
        dose_units = "mg/kg/day" # Default
        if 'ex' in parsed_data and 'EXDOSU' in parsed_data['ex'].columns:
             # Get the most frequent unit if multiple exist
             unit_counts = parsed_data['ex']['EXDOSU'].value_counts()
             if not unit_counts.empty:
                 dose_units = unit_counts.index[0]
        results['dose_units'] = dose_units
        logger.info(f"Using dose units: {dose_units}")


        # 2. Extract Endpoint Features
        endpoint_features = extract_endpoint_features(parsed_data, dose_groups)
        results['endpoint_features'] = endpoint_features
        if not endpoint_features.get('laboratory_tests') and not endpoint_features.get('body_weights'):
             results["error"] = "Failed to extract features for any endpoints (LB, BW)."
             logger.error(results["error"])
             return results

        # 3. Analyze Dose Response (Stats)
        analysis_results = analyze_dose_response(endpoint_features, ordered_doses)
        results['analysis_results'] = analysis_results
        # Don't stop if analysis fails for some endpoints, NOAEL determination handles missing results

        # 4. Determine Per-Endpoint NOAEL
        per_endpoint_noael = determine_per_endpoint_noael(analysis_results, ordered_doses)
        results['per_endpoint_noael'] = per_endpoint_noael
        if not per_endpoint_noael:
             logger.warning("Could not determine NOAEL for any endpoint.")
             # Allow continuing to generate summary based on stats info if available

        # 5. Determine Overall NOAEL
        overall_noael = get_overall_noael(per_endpoint_noael)
        results['overall_noael'] = overall_noael
        logger.info(f"Determined Overall NOAEL (based on stats): {overall_noael}")

        # 6. Prepare Summary Prompt
        summary_prompt = prepare_summary_prompt(
            dose_groups, ordered_doses, endpoint_features,
            analysis_results, per_endpoint_noael, overall_noael, dose_units
        )
        results['summary_prompt'] = summary_prompt

        # 7. Simulate TxGemma Response
        simulated_response = simulate_txgemma_response(summary_prompt, overall_noael, dose_units)
        results['simulated_response'] = simulated_response

        logger.info("TxGemma NOAEL Determination Demo Pipeline finished successfully.")
        return results

    except Exception as e:
        logger.error(f"Error during TxGemma NOAEL demo pipeline: {e}", exc_info=True)
        results["error"] = f"Pipeline failed: {e}"
        return results

# Note: The original main() function and visualization parts are removed
# as this module is intended to be imported and called via run_noael_determination_demo. 