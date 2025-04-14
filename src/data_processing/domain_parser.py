import pandas as pd
from typing import Dict, Optional, Any
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_demographics(dm_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Parses the Demographics (DM) domain.

    Extracts key information like USUBJID, ARM, ARMCD, ACTARM, SEX.
    Returns a DataFrame with essential demographic info or None if USUBJID is missing.
    """
    if dm_df is None or dm_df.empty:
        logging.warning("DM domain is missing or empty, cannot parse demographics.")
        return None

    # Define columns considered essential for basic identification and grouping
    essential_cols = ["USUBJID", "ARMCD", "SEX"]
    # Define columns that are good to have but might be missing
    optional_cols = ["ARM", "ACTARM", "SETCD"]

    present_essential = [col for col in essential_cols if col in dm_df.columns]
    present_optional = [col for col in optional_cols if col in dm_df.columns]

    # Must have USUBJID at a minimum
    if "USUBJID" not in present_essential:
        logging.error(
            "DM domain missing critical USUBJID column. Cannot parse demographics."
        )
        return None

    # Check if other essential columns are missing for grouping
    missing_essential = [col for col in essential_cols if col not in present_essential]
    if missing_essential:
        logging.warning(
            f"DM domain missing essential columns needed for some analyses: {missing_essential}. Proceeding with available data."
        )

    # Check for missing optional columns (previously required)
    missing_optional = [col for col in optional_cols if col not in present_optional]
    if missing_optional:
        logging.warning(
            f"DM domain missing optional columns: {missing_optional}. Parsing accuracy for some fields may be affected."
        )

    # Select all available essential and optional columns
    cols_to_keep = present_essential + present_optional
    if not cols_to_keep:
        # Should not happen due to USUBJID check, but as safety
        return None

    # Select relevant columns and remove duplicates (one row per subject)
    # Use USUBJID and SETCD (if available) for uniqueness check
    unique_subset = ["USUBJID"]
    if "SETCD" in cols_to_keep:
        unique_subset.append("SETCD")

    demographics = dm_df[cols_to_keep].drop_duplicates(subset=unique_subset)
    logging.info(
        f"Parsed DM domain for {demographics.shape[0]} unique subjects (using {unique_subset} for uniqueness)."
    )
    return demographics


def parse_exposure(ex_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Parses the Exposure (EX) domain.

    Extracts key dosing information: USUBJID, EXDOSE, EXDOSU, EXTRT, EXROUTE, EXSTDY, EXENDY.
    Returns a DataFrame with essential exposure info or None.
    """
    if ex_df is None or ex_df.empty:
        logging.warning("EX domain is missing or empty, cannot parse exposure.")
        return None

    # Core columns needed for dose information
    required_cols = ["USUBJID", "EXDOSE", "EXDOSU", "EXTRT", "EXROUTE"]
    # Optional but useful time columns
    optional_cols = ["EXSTDY", "EXENDY"]  # Start/End Day
    available_cols = [col for col in required_cols if col in ex_df.columns]
    available_cols.extend([col for col in optional_cols if col in ex_df.columns])

    if "USUBJID" not in available_cols or "EXDOSE" not in available_cols:
        logging.warning(
            "EX domain missing essential columns (USUBJID, EXDOSE). Cannot parse exposure."
        )
        return None

    exposure = ex_df[available_cols].copy()
    # Potential cleaning: Convert EXDOSE to numeric if not already
    if "EXDOSE" in exposure.columns:
        exposure["EXDOSE"] = pd.to_numeric(exposure["EXDOSE"], errors="coerce")
        # Log how many rows failed conversion
        failed_conversions = (
            exposure["EXDOSE"].isna().sum() - ex_df["EXDOSE"].isna().sum()
        )
        if failed_conversions > 0:
            logging.warning(
                f"Could not convert {failed_conversions} EXDOSE values to numeric."
            )

    logging.info(f"Parsed EX domain with {exposure.shape[0]} records.")
    return exposure


def parse_trial_summary(ts_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Parses the Trial Summary (TS) domain.

    Extracts key trial parameters (TSPARMCD) into a dictionary.
    Returns a dictionary of trial parameters or an empty dict.
    """
    trial_summary = {}
    if ts_df is None or ts_df.empty:
        logging.warning("TS domain is missing or empty, cannot parse trial summary.")
        return trial_summary

    if not all(col in ts_df.columns for col in ["TSPARMCD", "TSVAL"]):
        logging.warning(
            "TS domain missing TSPARMCD or TSVAL columns. Cannot parse trial summary."
        )
        return trial_summary

    # Create a dictionary from parameter codes and values
    # Handle potential duplicate TSPARMCDs (e.g., multiple PCLAS values) by taking the first one
    ts_df_unique = ts_df.drop_duplicates(subset=["TSPARMCD"], keep="first")
    trial_summary = pd.Series(
        ts_df_unique.TSVAL.values, index=ts_df_unique.TSPARMCD
    ).to_dict()

    logging.info(f"Parsed TS domain, extracted {len(trial_summary)} parameters.")
    return trial_summary


def parse_findings_domain(
    domain_df: Optional[pd.DataFrame], domain_name: str
) -> Optional[pd.DataFrame]:
    """Basic parser for findings domains (e.g., LB, CL, BW, PC, MI).

    Extracts common columns like USUBJID, --TESTCD, --STRESN/--STRESC, --DY.
    More specific parsing logic might be needed depending on the endpoint.
    Returns a DataFrame with common findings info or None.
    """
    if domain_df is None or domain_df.empty:
        # Don't warn for every missing findings domain, only log info if present
        # logging.info(f"{domain_name.upper()} domain is missing or empty.")
        return None

    domain_prefix = domain_name.upper()
    testcd_col = f"{domain_prefix}TESTCD"
    test_col = f"{domain_prefix}TEST"
    stresn_col = f"{domain_prefix}STRESN"  # Numeric result
    stresc_col = f"{domain_prefix}STRESC"  # Character result
    stresu_col = f"{domain_prefix}STRESU"  # Units
    dy_col = f"{domain_prefix}DY"  # Study Day

    required_cols = ["USUBJID"]
    result_cols = []
    present_cols = []

    if testcd_col in domain_df.columns:
        required_cols.append(testcd_col)
    if test_col in domain_df.columns:
        required_cols.append(test_col)

    # Check for result columns
    if stresn_col in domain_df.columns:
        result_cols.append(stresn_col)
    if stresc_col in domain_df.columns:
        result_cols.append(stresc_col)
    if not result_cols:
        logging.warning(
            f"{domain_prefix} domain missing result columns ({stresn_col}, {stresc_col})."
        )
        # Cannot proceed without results
        return None

    # Add other common columns if present
    if stresu_col in domain_df.columns:
        present_cols.append(stresu_col)
    if dy_col in domain_df.columns:
        present_cols.append(dy_col)

    all_needed_cols = required_cols + result_cols + present_cols
    findings = domain_df[all_needed_cols].copy()

    # Attempt to convert numeric result column
    if stresn_col in findings.columns:
        findings[stresn_col] = pd.to_numeric(findings[stresn_col], errors="coerce")

    # Attempt to convert day column
    if dy_col in findings.columns:
        findings[dy_col] = pd.to_numeric(findings[dy_col], errors="coerce")

    logging.info(f"Parsed {domain_prefix} domain with {findings.shape[0]} records.")
    return findings


def parse_domains(study_data: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Any]:
    """Orchestrates the parsing of different SEND domains.

    Args:
        study_data: Dictionary of loaded domain DataFrames (lowercase keys)
                    as produced by send_loader.load_send_study.

    Returns:
        A dictionary containing parsed data structures where keys are
        lowercase domain names (e.g., 'dm', 'ex', 'bw', 'ts').
    """
    parsed_data: Dict[str, Any] = {}

    logging.info("Starting domain parsing...")

    # Parse core domains
    dm_parsed = parse_demographics(study_data.get("dm"))
    if dm_parsed is not None:
        parsed_data["dm"] = dm_parsed

    ex_parsed = parse_exposure(study_data.get("ex"))
    if ex_parsed is not None:
        parsed_data["ex"] = ex_parsed

    # Trial summary is already a dict, add directly if not empty
    ts_parsed = parse_trial_summary(study_data.get("ts"))
    if ts_parsed:
        parsed_data["ts"] = ts_parsed

    # Parse common findings domains
    findings_domains = [
        "lb",
        "cl",
        "bw",
        "pc",
        "pp",
        "mi",
        "om",
    ]  # Add other relevant findings domains here
    for domain in findings_domains:
        if domain in study_data:
            parsed_df = parse_findings_domain(study_data[domain], domain)
            if parsed_df is not None:
                parsed_data[domain] = parsed_df  # Add directly with domain key

    logging.info("Finished domain parsing.")
    return parsed_data


# Example Usage (Optional - for testing)
# if __name__ == '__main__':
#     from pathlib import Path
#     from send_loader import load_send_study # Assuming send_loader.py is in the same directory
#
#     # Adjust the path to one of the downloaded PHUSE dataset folders
#     example_study_path = Path('../../data/external/phuse-scripts/data/send/CBER-POC-Pilot-Study1-Vaccine')
#     loaded_data = load_send_study(example_study_path)
#
#     if loaded_data:
#         parsed_results = parse_domains(loaded_data)
#
#         print("\n--- Parsed Demographics ---")
#         if parsed_results['dm'] is not None:
#             print(parsed_results['dm'].head())
#         else:
#             print("Not available")
#
#         print("\n--- Parsed Exposure ---")
#         if parsed_results['ex'] is not None:
#             print(parsed_results['ex'].head())
#         else:
#             print("Not available")
#
#         print("\n--- Parsed Trial Summary ---")
#         print(parsed_results['ts'])
#
#         print("\n--- Parsed Findings (LB Example) ---")
#         if 'lb' in parsed_results:
#             print(parsed_results['lb'].head())
#         else:
#             print("LB domain not available or failed parsing")
