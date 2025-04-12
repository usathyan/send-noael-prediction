import logging
import os
import re
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
# Remove OpenAI import if no longer needed elsewhere, keep requests
# from openai import OpenAI 
import requests 
import json # Needed for parsing streamed response chunks

logger = logging.getLogger(__name__)

# --- Helper Functions ---

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
            match = re.search(r'^(\d+(\.\d+)?)\b', str(dose_val).strip())
            if match:
                return float(match.group(1))
            else:
                # logger.debug(f"Could not extract numeric dose via regex from: {dose_val}")
                return np.nan
        except Exception as e:
            # logger.warning(f"Exception during regex dose extraction for '{dose_val}': {e}")
            return np.nan

def _get_dose_groups(dm_df: pd.DataFrame, ex_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
    """Merges subject demographics with exposure to get dose groups."""
    if dm_df is None or ex_df is None or not all(c in dm_df.columns for c in ['USUBJID']) or not all(c in ex_df.columns for c in ['USUBJID', 'EXDOSE']):
        logger.warning("Cannot determine dose groups: Missing DM/EX data or required columns.")
        return dm_df.copy().set_index('USUBJID'), np.nan
    
    # Use only necessary columns from DM
    subjects_df = dm_df[['USUBJID', 'SEX', 'ARMCD']].drop_duplicates(subset=['USUBJID']).set_index('USUBJID')

    # Extract numeric dose from EX
    ex_df_copy = ex_df[['USUBJID', 'EXDOSE', 'EXDOSU']].copy()
    ex_df_copy['DOSE_NUM'] = ex_df_copy['EXDOSE'].apply(_extract_numeric_dose)
    
    # Keep first valid numeric dose per subject (simplification)
    first_dose = ex_df_copy.dropna(subset=['DOSE_NUM']).drop_duplicates(subset=['USUBJID'], keep='first').set_index('USUBJID')
    
    # Merge dose info into subjects
    subjects_with_dose = subjects_df.join(first_dose[['DOSE_NUM', 'EXDOSU']], how='left')
    
    # Identify control dose (0.0 or minimum)
    control_dose = np.nan
    valid_doses = subjects_with_dose['DOSE_NUM'].dropna().unique()
    if len(valid_doses) > 0:
        if 0.0 in valid_doses:
            control_dose = 0.0
        else:
            control_dose = np.min(valid_doses)
        logger.info(f"Identified control dose: {control_dose}")
    else:
        logger.warning("No valid numeric doses found to determine control group.")
        
    return subjects_with_dose, control_dose

# --- Helper Function for Domain Summaries ---
def summarize_domain(df: pd.DataFrame, domain_name: str, subjects_df: pd.DataFrame, control_dose: Optional[float], dose_unit_str: str) -> str:
    summary_lines = [f"\n--- {domain_name} Summary ---"]
    if df is None or df.empty:
        summary_lines.append("No data available.")
        return "\n".join(summary_lines)

    # Basic check for USUBJID
    if 'USUBJID' not in df.columns:
         summary_lines.append("Data format issue: Missing USUBJID.")
         return "\n".join(summary_lines)

    # Merge dose info
    df_with_dose = pd.merge(df, subjects_df[['DOSE_NUM']], left_on='USUBJID', right_index=True, how='left').dropna(subset=['DOSE_NUM'])
    if df_with_dose.empty:
         summary_lines.append("No data linked to dose groups.")
         return "\n".join(summary_lines)

    # --- Domain-Specific Summarization ---
    try:
        if domain_name == "Clinical Observations (CL)":
            if 'CLTERM' in df_with_dose.columns:
                # Count unique non-normal observations per dose group
                cl_counts = df_with_dose[df_with_dose['CLSTRESC'] != 'NORMAL'].groupby('DOSE_NUM')['CLTERM'].nunique()
                if not cl_counts.empty:
                    for dose, count in cl_counts.items():
                        group_label = f"Dose Group ({dose:.2f} {dose_unit_str})"
                        if not pd.isna(control_dose) and dose == control_dose: group_label = f"Control Group ({dose:.2f} {dose_unit_str})"
                        summary_lines.append(f"- {group_label}: {count} distinct non-normal observation(s) recorded.")
                else:
                    summary_lines.append("No non-normal observations recorded.")
            else:
                summary_lines.append("Relevant columns (e.g., CLTERM) not found.")

        elif domain_name == "Laboratory Tests (LB)":
            key_tests = ['ALT', 'AST', 'BUN', 'CREA'] # Example key tests
            lb_data = df_with_dose[df_with_dose['LBTESTCD'].isin(key_tests)].copy()
            if not lb_data.empty and 'LBSTRESN' in lb_data.columns and 'LBSTRESU' in lb_data.columns:
                lb_data['LBSTRESN'] = pd.to_numeric(lb_data['LBSTRESN'], errors='coerce')
                lb_data = lb_data.dropna(subset=['LBSTRESN'])
                # Calculate mean value per dose group for key tests
                mean_lb = lb_data.groupby(['DOSE_NUM', 'LBTESTCD'])['LBSTRESN'].mean().unstack()
                if not mean_lb.empty:
                    summary_lines.append("Mean values for key tests:")
                    for dose, results in mean_lb.iterrows():
                        group_label = f"Dose Group ({dose:.2f} {dose_unit_str})"
                        if not pd.isna(control_dose) and dose == control_dose: group_label = f"Control Group ({dose:.2f} {dose_unit_str})"
                        test_results = []
                        for test, value in results.dropna().items():
                             # Find unit for this test (could be improved for robustness)
                             unit = lb_data.loc[lb_data['LBTESTCD']==test, 'LBSTRESU'].iloc[0] if not lb_data.loc[lb_data['LBTESTCD']==test, 'LBSTRESU'].empty else ''
                             test_results.append(f"{test}: {value:.2f} {unit}")
                        if test_results:
                            summary_lines.append(f"- {group_label}: {'; '.join(test_results)}")
                else:
                   summary_lines.append("No numeric results for key tests found.")
            else:
                summary_lines.append("No data for key tests (ALT, AST, BUN, CREA) or relevant columns missing.")

        elif domain_name == "Macroscopic Findings (MA)":
             if 'MANTERM' in df_with_dose.columns:
                 ma_counts = df_with_dose[df_with_dose['MASTRESC'] != 'NORMAL'].groupby('DOSE_NUM')['MANTERM'].nunique()
                 if not ma_counts.empty:
                      for dose, count in ma_counts.items():
                          group_label = f"Dose Group ({dose:.2f} {dose_unit_str})"
                          if not pd.isna(control_dose) and dose == control_dose: group_label = f"Control Group ({dose:.2f} {dose_unit_str})"
                          summary_lines.append(f"- {group_label}: {count} distinct non-normal gross finding(s) recorded.")
                 else:
                      summary_lines.append("No non-normal gross findings recorded.")
             else:
                 summary_lines.append("Relevant columns (e.g., MANTERM) not found.")


        elif domain_name == "Microscopic Findings (MI)":
            if 'MITERM' in df_with_dose.columns and 'MISEV' in df_with_dose.columns:
                mi_counts = df_with_dose[df_with_dose['MISTRESC'] != 'NORMAL'].groupby('DOSE_NUM')['USUBJID'].nunique()
                if not mi_counts.empty:
                     for dose, count in mi_counts.items():
                         group_label = f"Dose Group ({dose:.2f} {dose_unit_str})"
                         if not pd.isna(control_dose) and dose == control_dose: group_label = f"Control Group ({dose:.2f} {dose_unit_str})"
                         # Add most frequent finding per group? (More complex)
                         summary_lines.append(f"- {group_label}: {count} animal(s) with non-normal microscopic finding(s).")
                else:
                    summary_lines.append("No non-normal microscopic findings recorded.")
            else:
                summary_lines.append("Relevant columns (e.g., MITERM, MISEV) not found.")


        elif domain_name == "Organ Measurements (OM)":
            key_organs = ['LIVER', 'KIDNEY', 'SPLEEN'] # Example key organs
            om_data = df_with_dose[df_with_dose['OMTESTCD'].isin(key_organs)].copy()
            if not om_data.empty and 'OMSTRESN' in om_data.columns and 'OMSTRESU' in om_data.columns:
                om_data['OMSTRESN'] = pd.to_numeric(om_data['OMSTRESN'], errors='coerce')
                om_data = om_data.dropna(subset=['OMSTRESN'])
                mean_om = om_data.groupby(['DOSE_NUM', 'OMTESTCD'])['OMSTRESN'].mean().unstack()
                if not mean_om.empty:
                    summary_lines.append("Mean weights for key organs:")
                    for dose, results in mean_om.iterrows():
                        group_label = f"Dose Group ({dose:.2f} {dose_unit_str})"
                        if not pd.isna(control_dose) and dose == control_dose: group_label = f"Control Group ({dose:.2f} {dose_unit_str})"
                        organ_results = []
                        for organ, value in results.dropna().items():
                             unit = om_data.loc[om_data['OMTESTCD']==organ, 'OMSTRESU'].iloc[0] if not om_data.loc[om_data['OMTESTCD']==organ, 'OMSTRESU'].empty else ''
                             organ_results.append(f"{organ}: {value:.3f} {unit}") # Use 3 decimal places for weight
                        if organ_results:
                            summary_lines.append(f"- {group_label}: {'; '.join(organ_results)}")
                else:
                   summary_lines.append("No numeric results for key organs found.")
            else:
                summary_lines.append("No data for key organs (LIVER, KIDNEY, SPLEEN) or relevant columns missing.")

        else: # Fallback for unexpected domain names
            summary_lines.append("Summarization logic not implemented for this domain.")
    except Exception as e:
         logger.warning(f"Error summarizing domain {domain_name}: {e}", exc_info=True)
         summary_lines.append(f"An error occurred during {domain_name} summarization.")

    return "\n".join(summary_lines)

# --- Main Processing Function ---

def process_study_for_txgemma(parsed_data: Dict[str, Any], study_id: str) -> Dict:
    """
    Analyzes parsed SEND data (focusing on Body Weight),
    generates a prompt for an LLM via OpenRouter, calls the LLM, 
    and returns the result.
    """
    logger.info(f"Starting analysis for study: {study_id}")
    bw_analysis_summary = "Analysis not performed."
    llm_prompt = "No prompt generated."
    llm_response = "LLM not called."
    status = "Analysis Failed"
    error_message = None
    analysis_type = "Body Weight"
    results = {}

    try:
        # 1. Extract required dataframes
        dm_df = parsed_data.get('dm')
        ex_df = parsed_data.get('ex')
        ts_df = parsed_data.get('ts')
        bw_df = parsed_data.get('bw')
        # ---> Extract additional domains
        cl_df = parsed_data.get('cl')
        lb_df = parsed_data.get('lb')
        ma_df = parsed_data.get('ma')
        mi_df = parsed_data.get('mi')
        om_df = parsed_data.get('om')

        if dm_df is None or dm_df.empty:
            raise ValueError("DM domain data is missing or empty.")
        if ex_df is None or ex_df.empty:
            raise ValueError("EX domain data is missing or empty.")
        if bw_df is None or bw_df.empty:
            raise ValueError("BW domain data is missing or empty.")
        # TS is useful but potentially optional for basic BW analysis
        if not ts_df:
            logger.warning("TS domain data missing or empty. Study metadata might be incomplete.")

        # 2. Identify Dose Groups
        subjects_df, control_dose = _get_dose_groups(dm_df, ex_df)
        dose_groups = subjects_df['DOSE_NUM'].dropna().unique()
        dose_units = subjects_df['EXDOSU'].dropna().unique() 
        dose_unit_str = dose_units[0] if len(dose_units) > 0 else "units"
        logger.info(f"Identified dose groups: {sorted(dose_groups)}")
        
        # 3. Perform Body Weight Analysis
        logger.info("Performing Body Weight analysis...")
        bw_df_clean = bw_df[bw_df['BWTESTCD'] == 'BW'].copy()
        bw_df_clean['BWSTRESN'] = pd.to_numeric(bw_df_clean['BWSTRESN'], errors='coerce')
        bw_df_clean['BWDY'] = pd.to_numeric(bw_df_clean['BWDY'], errors='coerce')
        bw_df_clean = bw_df_clean.dropna(subset=['USUBJID', 'BWSTRESN', 'BWDY'])

        if bw_df_clean.empty:
            raise ValueError("No valid numeric Body Weight data found after cleaning.")
        
        # Merge dose info with BW data
        bw_data_with_dose = pd.merge(bw_df_clean, subjects_df[['DOSE_NUM']], left_on='USUBJID', right_index=True, how='left')
        bw_data_with_dose = bw_data_with_dose.dropna(subset=['DOSE_NUM'])
        
        # Calculate baseline (using earliest day <= 1)
        bw_data_with_dose = bw_data_with_dose.sort_values(by=['USUBJID', 'BWDY'])
        baseline_day = bw_data_with_dose['BWDY'][bw_data_with_dose['BWDY'] <= 1].max()
        if pd.isna(baseline_day):
            baseline_day = bw_data_with_dose['BWDY'].min() # Fallback
        logger.info(f"Using baseline day: {baseline_day}")
            
        baseline_bw = bw_data_with_dose[bw_data_with_dose['BWDY'] == baseline_day]
        baseline_bw = baseline_bw.set_index('USUBJID')[['BWSTRESN']].rename(columns={'BWSTRESN':'BW_BASELINE'})

        bw_data_with_baseline = pd.merge(bw_data_with_dose, baseline_bw, left_on='USUBJID', right_index=True, how='left')
        bw_data_with_baseline = bw_data_with_baseline.dropna(subset=['BW_BASELINE'])

        # Calculate % change from baseline
        bw_data_with_baseline['BW_PCT_CHANGE'] = ((bw_data_with_baseline['BWSTRESN'] - bw_data_with_baseline['BW_BASELINE']) / bw_data_with_baseline['BW_BASELINE']) * 100

        # Aggregate mean % change per dose group at study end
        terminal_day = bw_data_with_baseline['BWDY'].max()
        logger.info(f"Using terminal day: {terminal_day}")
        terminal_bw_change = bw_data_with_baseline[bw_data_with_baseline['BWDY'] == terminal_day]
        
        # Group by DOSE_NUM and calculate mean pct change
        mean_terminal_change_per_dose = terminal_bw_change.groupby('DOSE_NUM')['BW_PCT_CHANGE'].mean().sort_index()
        logger.info(f"Mean terminal BW % change by dose:\n{mean_terminal_change_per_dose}")
        
        # Prepare summary text for the prompt
        bw_analysis_summary_lines = []
        for dose, change in mean_terminal_change_per_dose.items():
            group_label = f"Control Group ({dose:.2f} {dose_unit_str})" if not pd.isna(control_dose) and dose == control_dose else f"Dose Group ({dose:.2f} {dose_unit_str})"
            bw_analysis_summary_lines.append(f"- {group_label}: Mean terminal BW change: {change:.2f}%")
            # TODO: Add statistical comparison summary if implemented
        bw_analysis_summary = "\n".join(bw_analysis_summary_lines)

        # 3b. Summarize other domains
        logger.info("Summarizing findings from other available domains...")
        all_summaries = [bw_analysis_summary] # Start with BW summary

        # ---> Add summaries from other domains using the helper function
        all_summaries.append(summarize_domain(cl_df, "Clinical Observations (CL)", subjects_df, control_dose, dose_unit_str))
        all_summaries.append(summarize_domain(lb_df, "Laboratory Tests (LB)", subjects_df, control_dose, dose_unit_str))
        all_summaries.append(summarize_domain(ma_df, "Macroscopic Findings (MA)", subjects_df, control_dose, dose_unit_str))
        all_summaries.append(summarize_domain(mi_df, "Microscopic Findings (MI)", subjects_df, control_dose, dose_unit_str))
        all_summaries.append(summarize_domain(om_df, "Organ Measurements (OM)", subjects_df, control_dose, dose_unit_str))

        # ---> Combine into comprehensive summary
        comprehensive_findings_summary = "\n".join(s for s in all_summaries if s and "No data available." not in s and "Summarization logic not implemented" not in s) # Filter out empty/unavailable

        # 4. Generate LLM Prompt (Update to use comprehensive summary)
        logger.info("Generating LLM prompt...")
        
        # --- Extract Study Metadata --- 
        study_metadata_lines = []
        # Species (already extracted)
        species = dm_df['SPECIES'].iloc[0] if 'SPECIES' in dm_df.columns and not dm_df['SPECIES'].empty else 'Not specified'
        study_metadata_lines.append(f"- Species: {species}")
        
        # Sexes Tested
        sexes = dm_df['SEX'].unique() if 'SEX' in dm_df.columns else []
        study_metadata_lines.append(f"- Sexes Tested: {', '.join(sex for sex in sexes if pd.notna(sex))}")

        # Duration (from TS dictionary)
        planned_duration_str = "Not specified"
        # Check if ts_df dictionary exists and is not empty
        if ts_df:
            # Look for relevant keys directly in the dictionary
            duration_keys = ['TRTDUR', 'STUDYDAY'] # Common parameter codes for duration
            for key in duration_keys:
                 duration_val = ts_df.get(key) # Use .get() to avoid KeyError
                 if duration_val is not None:
                      planned_duration_str = str(duration_val)
                      break # Use the first one found
        study_metadata_lines.append(f"- Planned Duration: {planned_duration_str}")
        
        # Route of Administration (from EX)
        route = "Not specified"
        if ex_df is not None and not ex_df.empty and 'EXROUTE' in ex_df.columns:
             valid_routes = ex_df['EXROUTE'].dropna()
             if not valid_routes.empty:
                  route = valid_routes.mode()[0] if not valid_routes.mode().empty else valid_routes.iloc[0]
        study_metadata_lines.append(f"- Route of Administration: {route}")
        
        # Test Article (from TS dictionary)
        test_article = "Not specified"
        # Check if ts_df dictionary exists and is not empty
        if ts_df: 
            # Look for relevant keys directly in the dictionary
            ta_keys = ['TSTIND', 'TRT', 'TSTNAM', 'TEST ARTICLE'] # Common parameter codes
            for key in ta_keys:
                ta_val = ts_df.get(key)
                if ta_val is not None:
                    test_article = str(ta_val)
                    break # Use the first one found
        study_metadata_lines.append(f"- Test Article: {test_article}")
        
        study_metadata_summary = "Study Metadata:\n" + "\n".join(study_metadata_lines)
        
        # --- Construct Final Prompt --- (Update to use comprehensive summary)
        llm_prompt = f"""
Analyze the following preclinical toxicology study data to help assess the No Observed Adverse Effect Level (NOAEL):

{study_metadata_summary}

Comprehensive Findings Summary:
{comprehensive_findings_summary}

Based on the provided study metadata and comprehensive findings summary:
1. Identify the key toxicology findings suggested by the data.
2. Provide an overall toxicological assessment based on these findings.
3. Assess the characteristics relevant to determining the No Observed Adverse Effect Level (NOAEL).
4. Discuss any limitations in the provided data for making a definitive NOAEL determination.

Please ensure your response specifies the dose units ({dose_unit_str}).
"""
        logger.debug(f"Generated Prompt:\n{llm_prompt}")

        # 5. Call LLM API via Friendli
        logger.info("Calling LLM API via Friendli...")
        friendli_token = os.getenv("FRIENDLI_TOKEN")
        friendli_url = "https://api.friendli.ai/dedicated/v1/chat/completions"
        friendli_model_id = "2c137my37hew" # Your specific model ID

        if not friendli_token:
            raise ValueError("FRIENDLI_TOKEN environment variable not set.")

        try:
            headers = {
                "Authorization": "Bearer " + friendli_token,
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": friendli_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": llm_prompt
                    }
                ],
                # Consider adjusting these parameters based on your model's needs
                "max_tokens": 4096, # Adjust if needed, OpenRouter used default
                "top_p": 0.8,     # Match your example
                "stream": True,
                "stream_options": {
                    "include_usage": True 
                }
            }

            logger.info(f"Using Friendli model: {friendli_model_id}")
            response = requests.post(friendli_url, json=payload, headers=headers, stream=True)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            full_response_content = ""
            usage_data = None

            # Process the streamed response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            # logger.debug("Stream finished.")
                            break
                        try:
                            chunk = json.loads(json_str)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content'] is not None:
                                    full_response_content += delta['content']
                            # Check for usage data at the end of the stream
                            if 'usage' in chunk and chunk['usage']:
                                usage_data = chunk['usage']
                                # logger.debug(f"Received usage data: {usage_data}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON chunk: {json_str}")
                            continue # Skip malformed lines
                    else:
                        logger.warning(f"Unexpected line format in stream: {decoded_line}")

            # --- Status and Response Handling ---
            if full_response_content:
                llm_response = full_response_content.strip()
                status = "Analysis Successful" # Set status to Success
                if usage_data: # Log usage if available
                    logger.info(f"Friendli API usage: {usage_data}")
                logger.info("Received and processed response from Friendli LLM.")
            elif response.status_code == 200: # Succeeded but no content
                 llm_response = "Received empty but successful response from Friendli LLM."
                 status = "Analysis Successful" # Still successful
                 logger.info("Received empty but successful response from Friendli LLM.")
            # Note: Non-200 status should have been caught by raise_for_status() 
            # If we somehow get here with non-200, status remains "Analysis Failed"
            # and llm_response might be the default "LLM not called."

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Friendli API: {e}", exc_info=True)
            error_message = f"Error calling LLM via Friendli: {e}"
            llm_response = "Failed to get response from Friendli LLM."
            # status remains "Analysis Failed"
        except Exception as e:
            logger.error(f"Error processing Friendli response: {e}", exc_info=True)
            error_message = f"Error processing Friendli response: {e}"
            llm_response = "Failed to process response from Friendli LLM."
            # status remains "Analysis Failed"


    except ValueError as ve:
        logger.error(f"Data validation or processing error: {ve}")
        error_message = str(ve)
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {e}"
    
    # 6. Format and return results
    results = {
        "study_id": study_id,
        "status": status,
        "analysis_type": "Comprehensive", # Update analysis type
        "comprehensive_findings_summary": comprehensive_findings_summary, # Add comprehensive summary
        "llm_prompt": llm_prompt,
        "llm_response": llm_response,
        "error": error_message
    }
    logger.info(f"Finished processing for study: {study_id}. Status: {status}")
    return results 