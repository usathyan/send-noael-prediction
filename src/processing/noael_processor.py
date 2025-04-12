import logging
import os
import re
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from openai import OpenAI

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

        # 4. Generate LLM Prompt (Enhanced structure)
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
        
        # --- Construct Final Prompt --- 
        llm_prompt = f"""
Analyze the following preclinical toxicology study data to determine the No Observed Adverse Effect Level (NOAEL) based primarily on the Body Weight findings:

{study_metadata_summary}

Body Weight Findings Summary:
{bw_analysis_summary}

Based **only** on the provided Body Weight findings summary, what is the NOAEL for this study in {dose_unit_str}? Provide your reasoning clearly, referencing specific dose groups and changes.
"""
        logger.debug(f"Generated Prompt:\n{llm_prompt}")

        # 5. Call LLM API via OpenRouter using OpenAI SDK
        logger.info("Calling LLM API via OpenRouter...")
        api_key = os.getenv("OPENROUTER_API_KEY")
        # Use the specific model from env or fallback
        model_name = os.getenv("LLM_MODEL_NAME", "google/gemini-2.5-pro-exp-03-25:free") 
        site_url = os.getenv("OPENROUTER_SITE_URL", "") # Optional header
        site_name = os.getenv("OPENROUTER_SITE_NAME", "") # Optional header

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            
            headers = {}
            if site_url: headers["HTTP-Referer"] = site_url
            if site_name: headers["X-Title"] = site_name
            
            logger.info(f"Using LLM model: {model_name}")
            completion = client.chat.completions.create(
                extra_headers=headers if headers else None,
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": llm_prompt
                    }
                ]
                # Add other parameters like temperature, max_tokens if needed
            )
            
            # Extract the response text
            if completion.choices and completion.choices[0].message:
                 llm_response = completion.choices[0].message.content
            else:
                 llm_response = "Received empty response from LLM."
                 
            logger.info("Received response from LLM via OpenRouter.")
            status = "Analysis Successful"
            
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}", exc_info=True)
            error_message = f"Error calling LLM via OpenRouter: {e}"
            llm_response = "Failed to get response from LLM."
            # Keep status as Analysis Failed if LLM call fails

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
        "analysis_type": analysis_type,
        "bw_analysis_summary": bw_analysis_summary, # Include the generated summary
        "llm_prompt": llm_prompt,
        "llm_response": llm_response,
        "error": error_message
    }
    logger.info(f"Finished processing for study: {study_id}. Status: {status}")
    return results 