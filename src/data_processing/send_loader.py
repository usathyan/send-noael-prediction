import pandas as pd

# import xport.v56  # No longer using xport library
import pyreadstat  # Use pyreadstat instead
from pathlib import Path
import logging
from typing import Dict, Optional, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_xpt_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Loads a single XPT file into a pandas DataFrame using pyreadstat."""
    if not file_path.exists() or not file_path.suffix.lower() == ".xpt":
        logging.warning(f"File not found or not an XPT file: {file_path}")
        return None
    try:
        # pyreadstat reads the XPT file and returns a tuple (DataFrame, metadata)
        df, meta = pyreadstat.read_xport(file_path)
        # Get dataset name from metadata if needed (less critical now)
        dataset_name = (
            meta.table_name if hasattr(meta, "table_name") else file_path.stem
        )
        logging.info(
            f"Successfully loaded {file_path} ({dataset_name}) with shape {df.shape}"
        )
        # pyreadstat usually handles string decoding, but check just in case
        # (No explicit decode needed typically)
        return df
    except Exception as e:
        logging.error(
            f"Error loading XPT file {file_path} using pyreadstat: {e}", exc_info=True
        )
        return None


def load_send_study(
    study_dir: Path,
    domains: List[str] = ["DM", "EX", "LB", "CL", "TS", "BW", "PC", "PP"],
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Loads multiple SEND domain XPT files from a study directory.
    Handles cases where the zip file might contain a single top-level folder.

    Args:
        study_dir: The directory where the study zip was extracted.
        domains: A list of uppercase domain abbreviations expected (used for reporting missing).

    Returns:
        A dictionary where keys are domain abbreviations (lowercase) and values
        are the corresponding pandas DataFrames, or None if a domain file
        is not found or fails to load.
    """
    study_data: Dict[str, Optional[pd.DataFrame]] = {}
    if not study_dir.is_dir():
        logging.error(f"Study directory not found: {study_dir}")
        return study_data

    logging.info(f"Loading SEND study from target directory: {study_dir}")

    # --- Determine the actual directory containing .xpt files ---
    search_dir = study_dir  # Default search path
    try:
        potential_data_dir_found = False
        for item in study_dir.iterdir():
            # Check if item is a directory and contains any .xpt files
            if item.is_dir():
                if list(item.glob("*.xpt")):  # Check for .xpt files inside
                    logging.info(
                        f"Found subdirectory '{item.name}' containing XPT files. Setting search path to this directory."
                    )
                    search_dir = item
                    potential_data_dir_found = True
                    break  # Assume first one found is correct

        # Log a warning if no subdirectory with .xpt files was found,
        # and the base directory also doesn't have .xpt files directly.
        if not potential_data_dir_found and not list(study_dir.glob("*.xpt")):
            logging.warning(
                f"Could not find a subdirectory containing .xpt files, and no .xpt files found directly in {study_dir}. Check zip structure."
            )
            # search_dir remains study_dir, the loop below will likely find nothing.

    except OSError as e:
        logging.error(
            f"Error listing contents of {study_dir}: {e}. Searching in base directory."
        )
        search_dir = study_dir  # Fallback

    logging.info(f"Searching for XPT files in: {search_dir}")
    loaded_domains = set()
    found_files = False

    # Search for .xpt files in the determined search directory
    for file_path in search_dir.glob("*.xpt"):
        found_files = True
        domain = file_path.stem.upper()  # Get domain name (e.g., 'DM' from 'dm.xpt')
        df = load_xpt_file(file_path)
        study_data[domain.lower()] = df
        if df is not None:
            loaded_domains.add(domain)

    if not found_files:
        logging.warning(
            f"No .xpt files found in {search_dir}. Please check the zip file structure and extraction contents."
        )

    # Report on requested domains that were missing
    missing_requested = set(d.upper() for d in domains) - loaded_domains
    if missing_requested:
        logging.warning(
            f"Did not find or load requested domains in {search_dir}: {', '.join(missing_requested)}"
        )
        # Ensure missing requested domains have None entry
        for domain_upper in missing_requested:
            if domain_upper.lower() not in study_data:
                study_data[domain_upper.lower()] = None

    logging.info(
        f"Finished loading study from {search_dir}. Loaded domains: {', '.join(sorted(d.lower() for d in loaded_domains))}"
    )
    return study_data


def validate_send_domains(
    study_data: Dict[str, Optional[pd.DataFrame]],
    required_domains: List[str] = ["DM", "EX", "TS"],
) -> bool:
    """
    Performs basic validation checks on the loaded SEND domain data.

    Args:
        study_data: Dictionary of loaded domain DataFrames (lowercase keys).
        required_domains: List of uppercase domain abbreviations considered essential.

    Returns:
        True if basic validation passes, False otherwise.
    """
    is_valid = True
    required_domains_lower = [d.lower() for d in required_domains]

    for domain_lower in required_domains_lower:
        if domain_lower not in study_data or study_data[domain_lower] is None:
            logging.error(
                f"Validation failed: Required domain '{domain_lower.upper()}' is missing or failed to load."
            )
            is_valid = False
        elif study_data[domain_lower].empty:
            logging.warning(
                f"Validation warning: Required domain '{domain_lower.upper()}' is empty."
            )
            # Depending on requirements, you might set is_valid = False here

    # Add more specific validation rules here (e.g., check for essential columns like USUBJID)
    if is_valid and "dm" in study_data and study_data["dm"] is not None:
        if "USUBJID" not in study_data["dm"].columns:
            logging.error(
                "Validation failed: DM domain missing required column 'USUBJID'."
            )
            is_valid = False

    if is_valid:
        logging.info("Basic SEND domain validation successful.")
    else:
        logging.error("Basic SEND domain validation failed.")

    return is_valid


# Example Usage updated slightly
# if __name__ == '__main__':
#     # Adjust the path to one of the downloaded PHUSE dataset folders
#     example_study_path = Path(__file__).parent.parent.parent / 'data' / 'external' / 'phuse-scripts' / 'data' / 'send' / 'CBER-POC-Pilot-Study1-Vaccine'
#     if example_study_path.exists():
#         loaded_data = load_send_study(example_study_path)
#         if loaded_data:
#             print(f"\nLoaded domains: {list(loaded_data.keys())}")
#             is_valid = validate_send_domains(loaded_data)
#             print(f"Dataset validation result: {is_valid}\n")
#             # Print shape of loaded dataframes
#             for domain, df in loaded_data.items():
#                 if df is not None:
#                     print(f"Domain {domain}: Shape {df.shape}, Columns: {list(df.columns[:5])}...")
#                 else:
#                     print(f"Domain {domain}: Not loaded")
#     else:
#          print(f"Example study path not found: {example_study_path}")
