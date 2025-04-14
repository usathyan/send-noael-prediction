import logging
import os
import re
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests
import json

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
            match = re.search(r"^(\d+(\.\d+)?)\b", str(dose_val).strip())
            if match:
                return float(match.group(1))
            else:
                return np.nan
        except Exception:
            return np.nan


def _get_dose_groups(
    dm_df: pd.DataFrame, ex_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Merges subject demographics with exposure to get dose groups."""
    if (
        dm_df is None
        or ex_df is None
        or not all(c in dm_df.columns for c in ["USUBJID"])
        or not all(c in ex_df.columns for c in ["USUBJID", "EXDOSE"])
    ):
        logger.warning(
            "Cannot determine dose groups: Missing DM/EX data or required columns."
        )
        return dm_df.copy().set_index("USUBJID"), np.nan

    # Use only necessary columns from DM
    subjects_df = (
        dm_df[["USUBJID", "SEX", "ARMCD"]]
        .drop_duplicates(subset=["USUBJID"])
        .set_index("USUBJID")
    )

    # Extract numeric dose from EX
    ex_df_copy = ex_df[["USUBJID", "EXDOSE", "EXDOSU"]].copy()
    ex_df_copy["DOSE_NUM"] = ex_df_copy["EXDOSE"].apply(_extract_numeric_dose)

    # Keep first valid numeric dose per subject (simplification)
    first_dose = (
        ex_df_copy.dropna(subset=["DOSE_NUM"])
        .drop_duplicates(subset=["USUBJID"], keep="first")
        .set_index("USUBJID")
    )

    # Merge dose info into subjects
    subjects_with_dose = subjects_df.join(
        first_dose[["DOSE_NUM", "EXDOSU"]], how="left"
    )

    # Identify control dose (0.0 or minimum)
    control_dose = np.nan
    valid_doses = subjects_with_dose["DOSE_NUM"].dropna().unique()
    if len(valid_doses) > 0:
        if 0.0 in valid_doses:
            control_dose = 0.0
        else:
            control_dose = np.min(valid_doses)
        logger.info(f"Identified control dose: {control_dose}")
    else:
        logger.warning("No valid numeric doses found to determine control group.")

    return subjects_with_dose, control_dose


def _extract_strain_info(dm_df: pd.DataFrame, ts_df: Optional[Dict[str, Any]]) -> str:
    """Extract strain information from DM or TS domains."""
    strain = "Not specified"

    # Try to get strain from DM domain if STRAIN column exists
    if dm_df is not None and not dm_df.empty and "STRAIN" in dm_df.columns:
        strains = dm_df["STRAIN"].dropna().unique()
        if len(strains) > 0:
            strain = strains[0]
            return strain

    # If not found in DM, try TS domain
    if ts_df:
        strain_keys = ["STRAIN", "STRMN", "STRSPEC"]
        for key in strain_keys:
            if key in ts_df:
                strain = ts_df[key]
                return strain

    return strain


def _extract_study_duration(
    ex_df: pd.DataFrame, ts_df: Optional[Dict[str, Any]]
) -> str:
    """Extract study duration from EX or TS domains."""
    duration = "Not specified"

    # Try to calculate from EX domain first
    if ex_df is not None and not ex_df.empty and "EXDY" in ex_df.columns:
        try:
            max_day = pd.to_numeric(ex_df["EXDY"], errors="coerce").max()
            if not pd.isna(max_day) and max_day > 0:
                duration = f"{int(max_day)} days"
                return duration
        except Exception as e:
            logger.warning(f"Could not calculate duration from EX domain: {e}")

    # If not available from EX, try TS domain
    if ts_df:
        duration_keys = ["TRTDUR", "STUDYDUR", "SDUR", "PDUR"]
        for key in duration_keys:
            if key in ts_df:
                duration = ts_df[key]
                # Add "days" if it's just a number
                if duration.isdigit():
                    duration = f"{duration} days"
                return duration

    return duration


# --- Enhanced Domain Summarization Functions ---


def summarize_bodyweight_data(
    bw_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of body weight data with statistical comparisons."""
    summary_lines = []
    if bw_df is None or bw_df.empty:
        return "No body weight data available."

    try:
        # Filter to BW records
        bw_df_clean = bw_df[bw_df["BWTESTCD"] == "BW"].copy()
        bw_df_clean["BWSTRESN"] = pd.to_numeric(
            bw_df_clean["BWSTRESN"], errors="coerce"
        )
        bw_df_clean["BWDY"] = pd.to_numeric(bw_df_clean["BWDY"], errors="coerce")
        bw_df_clean = bw_df_clean.dropna(subset=["USUBJID", "BWSTRESN", "BWDY"])

        if bw_df_clean.empty:
            return "No valid numeric body weight data available."

        # Merge with dose info
        bw_with_dose = pd.merge(
            bw_df_clean,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        bw_with_dose = bw_with_dose.dropna(subset=["DOSE_NUM"])

        # Identify timepoints to show
        days = sorted(bw_with_dose["BWDY"].unique())
        # Show baseline, intermediate points (weeks), and terminal
        timepoints = [days[0]]  # Always include baseline

        # Add intermediate time points (approximately weekly)
        for week in range(1, int(max(days) / 7) + 1):
            week_day = week * 7
            closest_day = min(days, key=lambda x: abs(x - week_day))
            if closest_day not in timepoints:
                timepoints.append(closest_day)

        # Ensure terminal day is included
        if days[-1] not in timepoints:
            timepoints.append(days[-1])

        timepoints = sorted(timepoints)

        # Calculate mean body weights by dose group at selected timepoints
        summary_lines.append("Mean Body Weights (g) by Time Point:")

        # Table header with time points
        header = (
            "| Dose Group | " + " | ".join([f"Day {day}" for day in timepoints]) + " |"
        )
        separator = (
            "|"
            + "-" * (len("Dose Group") + 2)
            + "|"
            + "".join(["-" * (len(f"Day {day}") + 2) + "|" for day in timepoints])
        )
        summary_lines.append(header)
        summary_lines.append(separator)

        for dose in sorted(bw_with_dose["DOSE_NUM"].unique()):
            group_label = (
                f"Control ({dose:.2f} {dose_unit_str})"
                if not pd.isna(control_dose) and dose == control_dose
                else f"{dose:.2f} {dose_unit_str}"
            )

            row_values = []
            for day in timepoints:
                day_data = bw_with_dose[
                    (bw_with_dose["DOSE_NUM"] == dose) & (bw_with_dose["BWDY"] == day)
                ]
                if not day_data.empty:
                    mean_bw = day_data["BWSTRESN"].mean()
                    row_values.append(f"{mean_bw:.1f}")
                else:
                    row_values.append("N/A")

            row = f"| {group_label} | " + " | ".join(row_values) + " |"
            summary_lines.append(row)

        # Calculate % change from baseline to terminal
        summary_lines.append("\nBody Weight Change (Terminal vs. Baseline):")

        # Set up for baseline and terminal comparisons
        baseline_day = timepoints[0]
        terminal_day = timepoints[-1]

        for dose in sorted(bw_with_dose["DOSE_NUM"].unique()):
            group_label = (
                f"Control ({dose:.2f} {dose_unit_str})"
                if not pd.isna(control_dose) and dose == control_dose
                else f"{dose:.2f} {dose_unit_str}"
            )

            # Get baseline weights for this dose group
            baseline_weights = bw_with_dose[
                (bw_with_dose["DOSE_NUM"] == dose)
                & (bw_with_dose["BWDY"] == baseline_day)
            ]
            terminal_weights = bw_with_dose[
                (bw_with_dose["DOSE_NUM"] == dose)
                & (bw_with_dose["BWDY"] == terminal_day)
            ]

            # Skip if data missing
            if baseline_weights.empty or terminal_weights.empty:
                summary_lines.append(
                    f"- {group_label}: Insufficient data for comparison"
                )
                continue

            # Calculate changes by matching subjects
            changes = []
            for subject in baseline_weights["USUBJID"].unique():
                if subject in terminal_weights["USUBJID"].values:
                    baseline_weight = baseline_weights.loc[
                        baseline_weights["USUBJID"] == subject, "BWSTRESN"
                    ].iloc[0]
                    terminal_weight = terminal_weights.loc[
                        terminal_weights["USUBJID"] == subject, "BWSTRESN"
                    ].iloc[0]
                    pct_change = (
                        (terminal_weight - baseline_weight) / baseline_weight
                    ) * 100
                    changes.append(pct_change)

            if changes:
                mean_change = sum(changes) / len(changes)
                summary_lines.append(
                    f"- {group_label}: Mean change: {mean_change:.1f}% (n={len(changes)})"
                )
            else:
                summary_lines.append(
                    f"- {group_label}: No matched subjects for comparison"
                )

        # Compare to control if available
        if not pd.isna(control_dose):
            summary_lines.append("\nComparison to Control (Terminal Body Weight):")

            control_data = bw_with_dose[
                (bw_with_dose["DOSE_NUM"] == control_dose)
                & (bw_with_dose["BWDY"] == terminal_day)
            ]

            if not control_data.empty:
                control_mean = control_data["BWSTRESN"].mean()

                for dose in sorted(bw_with_dose["DOSE_NUM"].unique()):
                    if dose == control_dose:
                        continue  # Skip control vs control comparison

                    group_label = f"{dose:.2f} {dose_unit_str}"
                    group_data = bw_with_dose[
                        (bw_with_dose["DOSE_NUM"] == dose)
                        & (bw_with_dose["BWDY"] == terminal_day)
                    ]

                    if not group_data.empty:
                        group_mean = group_data["BWSTRESN"].mean()
                        pct_diff = ((group_mean - control_mean) / control_mean) * 100
                        summary_lines.append(
                            f"- {group_label}: {pct_diff:.1f}% compared to control"
                        )
            else:
                summary_lines.append("- Control data not available for comparison")

    except Exception as e:
        logger.warning(f"Error summarizing body weight data: {e}", exc_info=True)
        return "Error summarizing body weight data."

    return "\n".join(summary_lines)


def summarize_clinical_observations(
    cl_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of clinical observations."""
    summary_lines = ["\n## Clinical Observations Summary"]
    if cl_df is None or cl_df.empty:
        summary_lines.append("No clinical observation data available.")
        return "\n".join(summary_lines)

    try:
        # Check for required columns
        if "CLTERM" not in cl_df.columns or "USUBJID" not in cl_df.columns:
            summary_lines.append(
                "Missing required columns for clinical observations analysis."
            )
            return "\n".join(summary_lines)

        # Merge with dose information
        cl_with_dose = pd.merge(
            cl_df,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        cl_with_dose = cl_with_dose.dropna(subset=["DOSE_NUM"])

        if cl_with_dose.empty:
            summary_lines.append("No clinical observations linked to dose groups.")
            return "\n".join(summary_lines)

        # Count subjects per dose group (for incidence calculation)
        dose_subject_counts = cl_with_dose.groupby("DOSE_NUM")["USUBJID"].nunique()

        # Filter non-normal observations
        abnormal_cl = cl_with_dose[cl_with_dose["CLSTRESC"] != "NORMAL"]

        if abnormal_cl.empty:
            summary_lines.append("No abnormal clinical observations recorded.")
            return "\n".join(summary_lines)

        # Count observations by term and dose
        term_dose_counts = (
            abnormal_cl.groupby(["DOSE_NUM", "CLTERM"])["USUBJID"]
            .nunique()
            .reset_index()
        )
        term_dose_counts.columns = ["DOSE_NUM", "CLTERM", "SUBJECT_COUNT"]

        # Get terms sorted by frequency
        term_counts = term_dose_counts.groupby("CLTERM")["SUBJECT_COUNT"].sum()
        sorted_terms = term_counts.sort_values(ascending=False).index.tolist()

        # Format the summary by term, with incidence by dose group
        summary_lines.append("\nIncidence of Clinical Observations (# affected/total):")

        for term in sorted_terms[:10]:  # Limit to top 10 findings
            summary_lines.append(f"\n{term}:")
            for dose in sorted(cl_with_dose["DOSE_NUM"].unique()):
                group_label = (
                    f"Control ({dose:.2f} {dose_unit_str})"
                    if not pd.isna(control_dose) and dose == control_dose
                    else f"{dose:.2f} {dose_unit_str}"
                )

                # Get count for this term and dose
                count_row = term_dose_counts[
                    (term_dose_counts["DOSE_NUM"] == dose)
                    & (term_dose_counts["CLTERM"] == term)
                ]

                if not count_row.empty:
                    count = count_row["SUBJECT_COUNT"].iloc[0]
                    total = dose_subject_counts[dose]
                    incidence = (count / total) * 100
                    summary_lines.append(
                        f"- {group_label}: {count}/{total} ({incidence:.1f}%)"
                    )
                else:
                    total = dose_subject_counts[dose]
                    summary_lines.append(f"- {group_label}: 0/{total} (0%)")

        # If more than 10 findings, note this
        if len(sorted_terms) > 10:
            summary_lines.append(
                f"\n[{len(sorted_terms) - 10} additional terms not shown]"
            )

    except Exception as e:
        logger.warning(f"Error summarizing clinical observations: {e}", exc_info=True)
        summary_lines.append("Error processing clinical observations data.")

    return "\n".join(summary_lines)


def summarize_laboratory_data(
    lb_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of laboratory test results with statistical comparisons."""
    summary_lines = ["\n## Laboratory Tests Summary"]
    if lb_df is None or lb_df.empty:
        summary_lines.append("No laboratory data available.")
        return "\n".join(summary_lines)

    try:
        # Check for required columns
        required_cols = ["USUBJID", "LBTESTCD", "LBSTRESN", "LBSTRESU"]
        if not all(col in lb_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in lb_df.columns]
            summary_lines.append(f"Missing required columns: {', '.join(missing)}")
            return "\n".join(summary_lines)

        # Merge with dose information
        lb_with_dose = pd.merge(
            lb_df,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        lb_with_dose = lb_with_dose.dropna(subset=["DOSE_NUM"])

        if lb_with_dose.empty:
            summary_lines.append("No laboratory data linked to dose groups.")
            return "\n".join(summary_lines)

        # Convert numeric results
        lb_with_dose["LBSTRESN"] = pd.to_numeric(
            lb_with_dose["LBSTRESN"], errors="coerce"
        )
        lb_with_dose = lb_with_dose.dropna(subset=["LBSTRESN"])

        # Define key test groups
        test_groups = {
            "Liver Function": [
                "ALT",
                "AST",
                "ALKP",
                "BILI",
                "GGT",
                "LDH",
                "ALB",
                "TP",
                "TBIL",
                "DBIL",
            ],
            "Kidney Function": ["BUN", "CREAT", "UREA", "PHOS", "CA", "CL", "K", "NA"],
            "Hematology": [
                "HCT",
                "HGB",
                "RBC",
                "MCV",
                "MCHC",
                "MCH",
                "PLT",
                "WBC",
                "NEUTRO",
                "LYMPHO",
                "MONO",
                "EOS",
                "BASO",
            ],
            "Electrolytes": ["CA", "CL", "K", "NA", "PHOS"],
            "Lipids": ["CHOL", "TRIG", "HDL", "LDL"],
        }

        # Find which test groups we have data for
        available_tests = lb_with_dose["LBTESTCD"].unique()
        tests_to_show = []

        for group, tests in test_groups.items():
            # Find intersection of available tests and this group
            group_tests = [test for test in tests if test in available_tests]
            if group_tests:
                tests_to_show.extend(group_tests)
                summary_lines.append(f"\n### {group} Tests")

                # Calculate statistics for each test in this group
                for test in group_tests:
                    test_data = lb_with_dose[lb_with_dose["LBTESTCD"] == test]
                    if test_data.empty:
                        continue

                    # Get the unit for this test
                    unit = (
                        test_data["LBSTRESU"].mode()[0]
                        if not test_data["LBSTRESU"].empty
                        else ""
                    )

                    summary_lines.append(f"\n{test} ({unit}):")

                    # Get control mean if available
                    control_mean = None
                    if not pd.isna(control_dose):
                        control_test_data = test_data[
                            test_data["DOSE_NUM"] == control_dose
                        ]
                        if not control_test_data.empty:
                            control_mean = control_test_data["LBSTRESN"].mean()

                    # Show means by dose group with percent difference from control
                    for dose in sorted(test_data["DOSE_NUM"].unique()):
                        group_label = (
                            f"Control ({dose:.2f} {dose_unit_str})"
                            if not pd.isna(control_dose) and dose == control_dose
                            else f"{dose:.2f} {dose_unit_str}"
                        )

                        dose_test_data = test_data[test_data["DOSE_NUM"] == dose]
                        if not dose_test_data.empty:
                            n = len(dose_test_data)
                            mean = dose_test_data["LBSTRESN"].mean()

                            if control_mean and dose != control_dose:
                                pct_diff = ((mean - control_mean) / control_mean) * 100
                                direction = "higher" if pct_diff > 0 else "lower"
                                summary_lines.append(
                                    f"- {group_label}: {mean:.2f} (n={n}), {abs(pct_diff):.1f}% {direction} than control"
                                )
                            else:
                                summary_lines.append(
                                    f"- {group_label}: {mean:.2f} (n={n})"
                                )

        # If no recognized tests found
        if not tests_to_show:
            summary_lines.append(
                "\nNo recognized standard clinical pathology tests found in the data."
            )

    except Exception as e:
        logger.warning(f"Error summarizing laboratory data: {e}", exc_info=True)
        summary_lines.append("Error processing laboratory data.")

    return "\n".join(summary_lines)


def summarize_microscopic_findings(
    mi_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of microscopic findings with severity analysis."""
    summary_lines = ["\n## Microscopic Findings Summary"]
    if mi_df is None or mi_df.empty:
        summary_lines.append("No microscopic findings data available.")
        return "\n".join(summary_lines)

    try:
        # Check for required columns
        required_cols = ["USUBJID", "MITERM", "MISPEC", "MISEV"]
        if not all(col in mi_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in mi_df.columns]
            summary_lines.append(f"Missing required columns: {', '.join(missing)}")
            return "\n".join(summary_lines)

        # Merge with dose information
        mi_with_dose = pd.merge(
            mi_df,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        mi_with_dose = mi_with_dose.dropna(subset=["DOSE_NUM"])

        if mi_with_dose.empty:
            summary_lines.append("No microscopic findings linked to dose groups.")
            return "\n".join(summary_lines)

        # Count subjects per dose group for incidence calculation
        dose_subject_counts = subjects_df.groupby("DOSE_NUM").size()

        # Filter abnormal findings
        abnormal_mi = mi_with_dose[mi_with_dose["MISTRESC"] != "NORMAL"]

        if abnormal_mi.empty:
            summary_lines.append("No abnormal microscopic findings recorded.")
            return "\n".join(summary_lines)

        # Organize by tissue and finding
        findings_by_tissue = (
            abnormal_mi.groupby(["MISPEC", "MITERM", "DOSE_NUM"])["USUBJID"]
            .nunique()
            .reset_index()
        )
        findings_by_tissue.columns = ["Tissue", "Finding", "Dose", "Count"]

        # Get total number of findings to limit display
        total_findings = len(
            findings_by_tissue[["Tissue", "Finding"]].drop_duplicates()
        )

        # Group by tissue to show findings for each tissue
        for tissue in sorted(findings_by_tissue["Tissue"].unique()):
            tissue_findings = findings_by_tissue[findings_by_tissue["Tissue"] == tissue]

            # Skip tissues with only normal findings
            if tissue_findings.empty:
                continue

            summary_lines.append(f"\n### {tissue}")

            # For each finding in this tissue
            for finding in sorted(tissue_findings["Finding"].unique()):
                summary_lines.append(f"\n{finding}:")

                # Show incidence and severity by dose group
                for dose in sorted(mi_with_dose["DOSE_NUM"].unique()):
                    group_label = (
                        f"Control ({dose:.2f} {dose_unit_str})"
                        if not pd.isna(control_dose) and dose == control_dose
                        else f"{dose:.2f} {dose_unit_str}"
                    )

                    # Get count for this finding/tissue/dose combo
                    count_data = tissue_findings[
                        (tissue_findings["Finding"] == finding)
                        & (tissue_findings["Dose"] == dose)
                    ]

                    if not count_data.empty:
                        count = count_data["Count"].iloc[0]
                        total = dose_subject_counts.get(dose, 0)
                        if total > 0:
                            incidence = (count / total) * 100

                            # Get severity distribution for this finding/tissue/dose
                            severity_data = abnormal_mi[
                                (abnormal_mi["MISPEC"] == tissue)
                                & (abnormal_mi["MITERM"] == finding)
                                & (abnormal_mi["DOSE_NUM"] == dose)
                            ]

                            severity_counts = severity_data.groupby("MISEV").size()
                            severity_str = ", ".join(
                                [
                                    f"{sev}: {cnt}"
                                    for sev, cnt in severity_counts.items()
                                ]
                            )

                            summary_lines.append(
                                f"- {group_label}: {count}/{total} ({incidence:.1f}%) [Severity: {severity_str}]"
                            )
                        else:
                            summary_lines.append(
                                f"- {group_label}: {count}/? (unknown %)"
                            )
                    else:
                        total = dose_subject_counts.get(dose, 0)
                        if total > 0:
                            summary_lines.append(f"- {group_label}: 0/{total} (0%)")
                        else:
                            summary_lines.append(f"- {group_label}: 0/? (0%)")

        # If too many findings, note that we're only showing a subset
        if total_findings > 15:  # Arbitrary cutoff
            summary_lines.append(
                f"\n[{total_findings - 15} additional findings not shown]"
            )

    except Exception as e:
        logger.warning(f"Error summarizing microscopic findings: {e}", exc_info=True)
        summary_lines.append("Error processing microscopic findings data.")

    return "\n".join(summary_lines)


def summarize_organ_measurements(
    om_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of organ measurements with statistical comparisons."""
    summary_lines = ["\n## Organ Measurement Summary"]
    if om_df is None or om_df.empty:
        summary_lines.append("No organ measurement data available.")
        return "\n".join(summary_lines)

    try:
        # Check for required columns
        required_cols = ["USUBJID", "OMTESTCD", "OMSTRESN", "OMSTRESU"]
        if not all(col in om_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in om_df.columns]
            summary_lines.append(f"Missing required columns: {', '.join(missing)}")
            return "\n".join(summary_lines)

        # Merge with dose information
        om_with_dose = pd.merge(
            om_df,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        om_with_dose = om_with_dose.dropna(subset=["DOSE_NUM"])

        if om_with_dose.empty:
            summary_lines.append("No organ measurements linked to dose groups.")
            return "\n".join(summary_lines)

        # Convert numeric results
        om_with_dose["OMSTRESN"] = pd.to_numeric(
            om_with_dose["OMSTRESN"], errors="coerce"
        )
        om_with_dose = om_with_dose.dropna(subset=["OMSTRESN"])

        # Get list of organs measured
        organs = om_with_dose["OMTESTCD"].unique()

        # Key organs of toxicological significance
        key_organs = [
            "LIVER",
            "KIDNEY",
            "SPLEEN",
            "HEART",
            "BRAIN",
            "ADRENAL",
            "THYMUS",
            "TESTIS",
            "OVARY",
        ]

        # Filter to key organs that are present in the data
        organs_to_show = [organ for organ in key_organs if organ in organs]

        # If no key organs, use the most common ones in the data
        if not organs_to_show and len(organs) > 0:
            organ_counts = (
                om_with_dose.groupby("OMTESTCD").size().sort_values(ascending=False)
            )
            organs_to_show = organ_counts.index[:5].tolist()  # Show top 5

        if not organs_to_show:
            summary_lines.append(
                "No relevant organ measurements identified in the data."
            )
            return "\n".join(summary_lines)

        # For each selected organ
        for organ in organs_to_show:
            organ_data = om_with_dose[om_with_dose["OMTESTCD"] == organ]

            if organ_data.empty:
                continue

            # Get the unit for this organ
            unit = (
                organ_data["OMSTRESU"].mode()[0]
                if not organ_data["OMSTRESU"].empty
                else ""
            )

            summary_lines.append(f"\n### {organ} ({unit}):")

            # Get control mean if available
            control_mean = None
            if not pd.isna(control_dose):
                control_organ_data = organ_data[organ_data["DOSE_NUM"] == control_dose]
                if not control_organ_data.empty:
                    control_mean = control_organ_data["OMSTRESN"].mean()

            # Show absolute values and % difference from control by dose group
            for dose in sorted(organ_data["DOSE_NUM"].unique()):
                group_label = (
                    f"Control ({dose:.2f} {dose_unit_str})"
                    if not pd.isna(control_dose) and dose == control_dose
                    else f"{dose:.2f} {dose_unit_str}"
                )

                dose_organ_data = organ_data[organ_data["DOSE_NUM"] == dose]
                if not dose_organ_data.empty:
                    n = len(dose_organ_data)
                    mean = dose_organ_data["OMSTRESN"].mean()

                    if control_mean and dose != control_dose:
                        pct_diff = ((mean - control_mean) / control_mean) * 100
                        direction = "higher" if pct_diff > 0 else "lower"
                        summary_lines.append(
                            f"- {group_label}: {mean:.3f} (n={n}), {abs(pct_diff):.1f}% {direction} than control"
                        )
                    else:
                        summary_lines.append(f"- {group_label}: {mean:.3f} (n={n})")

            # Now check if relative weights are available (organ/body weight ratio)
            rel_organ = f"REL{organ}"
            if rel_organ in organs:
                rel_organ_data = om_with_dose[om_with_dose["OMTESTCD"] == rel_organ]

                if not rel_organ_data.empty:
                    # Get the unit for relative weight
                    rel_unit = (
                        rel_organ_data["OMSTRESU"].mode()[0]
                        if not rel_organ_data["OMSTRESU"].empty
                        else "%"
                    )

                    summary_lines.append(f"\n{organ} (Relative, {rel_unit}):")

                    # Get control mean for relative weight if available
                    rel_control_mean = None
                    if not pd.isna(control_dose):
                        rel_control_data = rel_organ_data[
                            rel_organ_data["DOSE_NUM"] == control_dose
                        ]
                        if not rel_control_data.empty:
                            rel_control_mean = rel_control_data["OMSTRESN"].mean()

                    # Show by dose group
                    for dose in sorted(rel_organ_data["DOSE_NUM"].unique()):
                        group_label = (
                            f"Control ({dose:.2f} {dose_unit_str})"
                            if not pd.isna(control_dose) and dose == control_dose
                            else f"{dose:.2f} {dose_unit_str}"
                        )

                        rel_dose_data = rel_organ_data[
                            rel_organ_data["DOSE_NUM"] == dose
                        ]
                        if not rel_dose_data.empty:
                            n = len(rel_dose_data)
                            mean = rel_dose_data["OMSTRESN"].mean()

                            if rel_control_mean and dose != control_dose:
                                pct_diff = (
                                    (mean - rel_control_mean) / rel_control_mean
                                ) * 100
                                direction = "higher" if pct_diff > 0 else "lower"
                                summary_lines.append(
                                    f"- {group_label}: {mean:.3f} (n={n}), {abs(pct_diff):.1f}% {direction} than control"
                                )
                            else:
                                summary_lines.append(
                                    f"- {group_label}: {mean:.3f} (n={n})"
                                )

    except Exception as e:
        logger.warning(f"Error summarizing organ measurements: {e}", exc_info=True)
        summary_lines.append("Error processing organ measurement data.")

    return "\n".join(summary_lines)


def summarize_macroscopic_findings(
    ma_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    control_dose: Optional[float],
    dose_unit_str: str,
) -> str:
    """Generate enhanced summary of macroscopic (gross) findings."""
    summary_lines = ["\n## Macroscopic Findings Summary"]
    if ma_df is None or ma_df.empty:
        summary_lines.append("No macroscopic findings data available.")
        return "\n".join(summary_lines)

    try:
        # Check for required columns
        required_cols = ["USUBJID", "MATERM", "MASPEC"]
        if not all(col in ma_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in ma_df.columns]
            summary_lines.append(f"Missing required columns: {', '.join(missing)}")
            return "\n".join(summary_lines)

        # Merge with dose information
        ma_with_dose = pd.merge(
            ma_df,
            subjects_df[["DOSE_NUM"]],
            left_on="USUBJID",
            right_index=True,
            how="left",
        )
        ma_with_dose = ma_with_dose.dropna(subset=["DOSE_NUM"])

        if ma_with_dose.empty:
            summary_lines.append("No macroscopic findings linked to dose groups.")
            return "\n".join(summary_lines)

        # Count subjects per dose group for incidence calculation
        dose_subject_counts = subjects_df.groupby("DOSE_NUM").size()

        # Filter abnormal findings
        abnormal_ma = ma_with_dose[ma_with_dose["MASTRESC"] != "NORMAL"]

        if abnormal_ma.empty:
            summary_lines.append("No abnormal macroscopic findings recorded.")
            return "\n".join(summary_lines)

        # Organize by tissue and finding
        findings_by_tissue = (
            abnormal_ma.groupby(["MASPEC", "MATERM", "DOSE_NUM"])["USUBJID"]
            .nunique()
            .reset_index()
        )
        findings_by_tissue.columns = ["Tissue", "Finding", "Dose", "Count"]

        # Get total findings to limit display
        total_findings = len(
            findings_by_tissue[["Tissue", "Finding"]].drop_duplicates()
        )

        # Group by tissue to show findings for each tissue
        for tissue in sorted(findings_by_tissue["Tissue"].unique()):
            tissue_findings = findings_by_tissue[findings_by_tissue["Tissue"] == tissue]

            # Skip tissues with only normal findings
            if tissue_findings.empty:
                continue

            summary_lines.append(f"\n### {tissue}")

            # For each finding in this tissue
            for finding in sorted(tissue_findings["Finding"].unique()):
                summary_lines.append(f"\n{finding}:")

                # Show incidence by dose group
                for dose in sorted(ma_with_dose["DOSE_NUM"].unique()):
                    group_label = (
                        f"Control ({dose:.2f} {dose_unit_str})"
                        if not pd.isna(control_dose) and dose == control_dose
                        else f"{dose:.2f} {dose_unit_str}"
                    )

                    # Get count for this finding/tissue/dose combo
                    count_data = tissue_findings[
                        (tissue_findings["Finding"] == finding)
                        & (tissue_findings["Dose"] == dose)
                    ]

                    if not count_data.empty:
                        count = count_data["Count"].iloc[0]
                        total = dose_subject_counts.get(dose, 0)
                        if total > 0:
                            incidence = (count / total) * 100
                            summary_lines.append(
                                f"- {group_label}: {count}/{total} ({incidence:.1f}%)"
                            )
                        else:
                            summary_lines.append(
                                f"- {group_label}: {count}/? (unknown %)"
                            )
                    else:
                        total = dose_subject_counts.get(dose, 0)
                        if total > 0:
                            summary_lines.append(f"- {group_label}: 0/{total} (0%)")
                        else:
                            summary_lines.append(f"- {group_label}: 0/? (0%)")

        # If too many findings, note that we're only showing a subset
        if total_findings > 15:  # Arbitrary cutoff
            summary_lines.append(
                f"\n[{total_findings - 15} additional findings not shown]"
            )

    except Exception as e:
        logger.warning(f"Error summarizing macroscopic findings: {e}", exc_info=True)
        summary_lines.append("Error processing macroscopic findings data.")

    return "\n".join(summary_lines)


# --- Main Enhanced Processing Function ---


def enhanced_process_for_txgemma(parsed_data: Dict[str, Any], study_id: str) -> Dict:
    """
    Performs enhanced analysis of SEND data across multiple domains,
    generates a more comprehensive and structured prompt for the LLM,
    calls the LLM via Friendli API, and returns the result.
    """
    logger.info(f"Starting enhanced analysis for study: {study_id}")
    comprehensive_findings_summary = "Analysis not performed."
    llm_prompt = "No prompt generated."
    llm_response = "LLM not called."
    status = "Analysis Failed"
    error_message = None
    analysis_type = "Comprehensive Toxicology"

    try:
        # 1. Extract all available domains
        dm_df = parsed_data.get("dm")
        ex_df = parsed_data.get("ex")
        ts_df = parsed_data.get("ts")
        bw_df = parsed_data.get("bw")
        cl_df = parsed_data.get("cl")
        lb_df = parsed_data.get("lb")
        ma_df = parsed_data.get("ma")
        mi_df = parsed_data.get("mi")
        om_df = parsed_data.get("om")

        # Log available domains
        available_domains = [
            domain
            for domain, data in parsed_data.items()
            if data is not None
            and (not isinstance(data, pd.DataFrame) or not data.empty)
        ]
        logger.info(
            f"Available domains for enhanced analysis: {', '.join(available_domains)}"
        )

        # Check required domains
        if dm_df is None or dm_df.empty:
            raise ValueError("DM domain data is missing or empty - cannot proceed.")
        if ex_df is None or ex_df.empty:
            raise ValueError("EX domain data is missing or empty - cannot proceed.")

        # 2. Identify Dose Groups and extract key study metadata
        subjects_df, control_dose = _get_dose_groups(dm_df, ex_df)
        dose_groups = sorted(subjects_df["DOSE_NUM"].dropna().unique())
        dose_units = subjects_df["EXDOSU"].dropna().unique()
        dose_unit_str = dose_units[0] if len(dose_units) > 0 else "units"

        # Format dose groups for display
        dose_groups_with_units = ", ".join(
            [f"{dose:.2f} {dose_unit_str}" for dose in dose_groups]
        )

        # Extract study metadata
        species = (
            dm_df["SPECIES"].iloc[0]
            if "SPECIES" in dm_df.columns and not dm_df["SPECIES"].empty
            else "Not specified"
        )
        strain = _extract_strain_info(dm_df, ts_df)

        # Sexes Tested
        sexes = dm_df["SEX"].unique() if "SEX" in dm_df.columns else []
        sexes_str = ", ".join(sex for sex in sexes if pd.notna(sex))

        # Study Duration
        duration = _extract_study_duration(ex_df, ts_df)

        # Route of Administration
        route = "Not specified"
        if ex_df is not None and not ex_df.empty and "EXROUTE" in ex_df.columns:
            valid_routes = ex_df["EXROUTE"].dropna()
            if not valid_routes.empty:
                route = (
                    valid_routes.mode()[0]
                    if not valid_routes.mode().empty
                    else valid_routes.iloc[0]
                )

        # Test Article
        test_article = "Not specified"
        if ts_df:
            ta_keys = ["TSTIND", "TRT", "TSTNAM", "TEST ARTICLE"]
            for key in ta_keys:
                ta_val = ts_df.get(key)
                if ta_val is not None:
                    test_article = str(ta_val)
                    break

        # 3. Generate enhanced domain summaries
        logger.info("Generating enhanced domain summaries...")

        # Body Weight Analysis
        bw_summary = "No body weight data available."
        if bw_df is not None and not bw_df.empty:
            bw_summary = summarize_bodyweight_data(
                bw_df, subjects_df, control_dose, dose_unit_str
            )
        else:
            logger.warning(
                "Body weight data (BW domain) is missing - using placeholder message"
            )

        # Clinical Observations
        cl_summary = "No clinical observations data available."
        if cl_df is not None and not cl_df.empty:
            cl_summary = summarize_clinical_observations(
                cl_df, subjects_df, control_dose, dose_unit_str
            )

        # Laboratory Tests
        lb_summary = "No laboratory test data available."
        if lb_df is not None and not lb_df.empty:
            lb_summary = summarize_laboratory_data(
                lb_df, subjects_df, control_dose, dose_unit_str
            )

        # Macroscopic Findings
        ma_summary = "No macroscopic findings data available."
        if ma_df is not None and not ma_df.empty:
            ma_summary = summarize_macroscopic_findings(
                ma_df, subjects_df, control_dose, dose_unit_str
            )

        # Microscopic Findings
        mi_summary = "No microscopic findings data available."
        if mi_df is not None and not mi_df.empty:
            mi_summary = summarize_microscopic_findings(
                mi_df, subjects_df, control_dose, dose_unit_str
            )

        # Organ Measurements
        om_summary = "No organ measurement data available."
        if om_df is not None and not om_df.empty:
            om_summary = summarize_organ_measurements(
                om_df, subjects_df, control_dose, dose_unit_str
            )

        # 4. Construct enhanced LLM prompt
        logger.info("Building enhanced LLM prompt...")

        llm_prompt = f"""
You are a toxicologist analyzing standardized SEND (Standard for Exchange of Nonclinical Data) datasets from a preclinical toxicology study. Analyze the following study data to help determine the No Observed Adverse Effect Level (NOAEL):

## Study Metadata
- Species: {species}
- Strain: {strain}
- Sex(es): {sexes_str}
- Study Duration: {duration}
- Route of Administration: {route}
- Test Article: {test_article}
- Dose Groups: {dose_groups_with_units}

## Body Weight Analysis
{bw_summary}

{cl_summary}

{lb_summary}

{ma_summary}

{mi_summary}

{om_summary}

Based on this data:
1. Identify the key toxicological findings across all domains, organizing them by dose group.
2. Determine whether there is a dose-response relationship for each finding.
3. Assess the toxicological significance of each finding in the context of this specific study.
4. Based on your comprehensive assessment, determine the most likely NOAEL and provide your reasoning.
5. Identify any data gaps or limitations that impact your confidence in the NOAEL determination.

Please structure your response with clear section headings and include the dose units ({dose_unit_str}) in your NOAEL determination.
"""

        # Store the comprehensive summary
        comprehensive_findings_summary = f"""
## Study Metadata
- Species: {species}
- Strain: {strain}
- Sex(es): {sexes_str}
- Study Duration: {duration}
- Route of Administration: {route}
- Test Article: {test_article}
- Dose Groups: {dose_groups_with_units}

## Body Weight Analysis
{bw_summary}

{cl_summary}

{lb_summary}

{ma_summary}

{mi_summary}

{om_summary}
"""

        # 5. Call LLM API via Friendli
        logger.info("Calling LLM API via Friendli with enhanced prompt...")
        friendli_token = os.getenv("FRIENDLI_TOKEN")
        friendli_url = "https://api.friendli.ai/dedicated/v1/chat/completions"
        friendli_model_id = "2c137my37hew"  # Use the existing model ID

        if not friendli_token:
            raise ValueError("FRIENDLI_TOKEN environment variable not set.")

        try:
            headers = {
                "Authorization": "Bearer " + friendli_token,
                "Content-Type": "application/json",
            }

            payload = {
                "model": friendli_model_id,
                "messages": [{"role": "user", "content": llm_prompt}],
                "max_tokens": 4096,
                "top_p": 0.8,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            logger.info(f"Using Friendli model: {friendli_model_id}")
            response = requests.post(
                friendli_url, json=payload, headers=headers, stream=True
            )
            response.raise_for_status()

            full_response_content = ""
            usage_data = None

            # Process the streamed response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[len("data: ") :]
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    full_response_content += delta["content"]
                            if "usage" in chunk and chunk["usage"]:
                                usage_data = chunk["usage"]
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON chunk: {json_str}")
                            continue
                    else:
                        logger.warning(
                            f"Unexpected line format in stream: {decoded_line}"
                        )

            # Handle response
            if full_response_content:
                llm_response = full_response_content.strip()
                status = "Analysis Successful"
                if usage_data:
                    logger.info(f"Friendli API usage: {usage_data}")
                logger.info("Received and processed response from Friendli LLM.")
            elif response.status_code == 200:
                llm_response = (
                    "Received empty but successful response from Friendli LLM."
                )
                status = "Analysis Successful"
                logger.info("Received empty but successful response from Friendli LLM.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Friendli API: {e}", exc_info=True)
            error_message = f"Error calling LLM via Friendli: {e}"
            llm_response = "Failed to get response from Friendli LLM."
        except Exception as e:
            logger.error(f"Error processing Friendli response: {e}", exc_info=True)
            error_message = f"Error processing Friendli response: {e}"
            llm_response = "Failed to process response from Friendli LLM."

    except ValueError as ve:
        logger.error(f"Data validation or processing error: {ve}")
        error_message = str(ve)
    except Exception as e:
        logger.error(f"Unexpected error during enhanced processing: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {e}"

    # Format and return results
    results = {
        "study_id": study_id,
        "status": status,
        "analysis_type": analysis_type,
        "comprehensive_findings_summary": comprehensive_findings_summary,
        "llm_prompt": llm_prompt,
        "llm_response": llm_response,
        "error": error_message,
    }

    logger.info(f"Finished enhanced processing for study: {study_id}. Status: {status}")
    return results
