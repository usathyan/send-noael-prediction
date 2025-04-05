# SEND (Standard for Exchange of Nonclinical Data) Research

## Overview and Purpose

SEND (Standard for Exchange of Nonclinical Data) is an implementation of the CDISC Study Data Tabulation Model (SDTM) specifically designed for nonclinical studies. It provides a standardized framework for organizing, formatting, and structuring nonclinical data for regulatory submissions to agencies like the FDA.

Key points about SEND:

- It was created to align the organization, formatting, and structure of all nonclinical data across sponsors and studies
- It helps make the review process of nonclinical data more productive and efficient
- It is one of the required standards for data submission to the FDA
- It is chartered by CDISC (Clinical Data Interchange Standards Consortium), a globally recognized, not-for-profit organization

## Regulatory Requirements

The FDA requires SEND for most nonclinical studies:

- For CDER (Center for Drug Evaluation and Research):
  - Required for all studies that start after December 17, 2016 for NDAs, BLAs, and ANDAs
  - Required for all studies that start after December 17, 2017 for INDs
  
- For CBER (Center for Biologics Evaluation and Research):
  - Required for all studies that start after March 15, 2023 for NDAs, BLAs, ANDAs, and INDs

- Version requirements:
  - SEND version 3.0 for studies starting after the requirement dates but before March 15, 2019 (NDAs, BLAs, ANDAs) or March 15, 2020 (INDs)
  - SEND version 3.1 for studies starting after these dates

## SEND Implementation Guide (IG)

The SEND Implementation Guide (IG) describes how to prepare, manage, and structure data for FDA submission:

- The current version (as of 2021) is 3.1.1, published on March 30, 2021
- It includes details for the preparation of datasets for:
  - Single-dose toxicity studies
  - Repeat-dose toxicity studies
  - Carcinogenicity studies
  - Cardiovascular/respiratory safety pharmacology studies

Key updates in version 3.1.1:
- Additional examples and descriptions of variables (e.g., nominal timing)
- More examples to model relationships between domains (concentration data, dosing, PK parameters, etc.)
- Addition of Cardiovascular (CV) and Respiratory (RE) domains
- Updated requirements for specific variables
- Additional variables for consistent identifiers across multiple domains

## SEND Dataset Structure

SEND datasets are organized into domains, which are categories of data:

1. **Interventions**: Data about treatments or procedures applied to subjects
2. **Events**: Data about occurrences or incidents during the study
3. **Findings**: Data about observations, measurements, or assessments
4. **Trial Design**: Data about the study design
5. **Relationship**: Data about relationships between other data elements
6. **Special-Purpose**: Data that doesn't fit into other categories

Each domain contains variables that serve different roles:
- **Identifying variables**: Used to identify unique records
- **Qualifier variables**: Provide additional context
- **Rule variables**: Define rules or conditions
- **Timing variables**: Provide information about when events occurred

Many variables use controlled terminology (CT), which means they must use standardized values to ensure consistency and eliminate ambiguity.

## Benefits of SEND

1. Assists agency reviewers in application evaluation
2. Removes ambiguity within reported results
3. Reduces costs and timelines for FDA submission and review
4. Guides sponsors to collect and report raw data in a structured, standardized, and interpretable manner

## Challenges of SEND Implementation

1. Limited understanding of SEND requirements, especially for companies without exposure to clinical CDISC standards
2. Timing challenges due to evolving requirements
3. Collection of laboratory data in an organized and readily usable format
4. Manual data entry from handwritten records, which requires additional QC verification

## Relationship to NOAEL Determination

NOAEL (No Observed Adverse Effect Level) determination relies on the systematic analysis of toxicology study data. SEND standardization provides several advantages for NOAEL determination:

1. **Consistent Data Structure**: SEND ensures that toxicology data is organized consistently across studies, making it easier to identify adverse effects and determine NOAEL values.

2. **Standardized Terminology**: The use of controlled terminology in SEND reduces ambiguity in the interpretation of findings, leading to more reliable NOAEL determinations.

3. **Cross-Study Analysis**: SEND's standardized format facilitates the comparison of data across multiple studies, enabling more robust NOAEL determinations based on broader evidence.

4. **Automated Analysis**: The structured nature of SEND data makes it amenable to automated analysis, including machine learning approaches for NOAEL prediction.

5. **Comprehensive Data Capture**: SEND domains capture all relevant aspects of toxicology studies, ensuring that all data needed for NOAEL determination is available in a standardized format.

## Relevant SEND Domains for NOAEL Determination

Several SEND domains are particularly relevant for NOAEL determination:

1. **Trial Summary (TS)**: Contains study-level information, including species, study duration, and study design.

2. **Trial Arms (TA)**: Describes the different treatment groups in the study.

3. **Trial Sets (TX)**: Defines sets of subjects used for analysis.

4. **Demographics (DM)**: Contains subject-level information such as species, strain, sex, and age.

5. **Dosing (EX)**: Contains information about the test article administration, including dose levels.

6. **Clinical Observations (CL)**: Contains observations of clinical signs that may indicate toxicity.

7. **Body Weight (BW)**: Contains body weight measurements, which can indicate systemic toxicity.

8. **Food Consumption (FC)**: Contains food consumption data, which can indicate palatability issues or systemic toxicity.

9. **Laboratory Test Results (LB)**: Contains clinical pathology data, including hematology, clinical chemistry, and urinalysis.

10. **Microscopic Findings (MI)**: Contains histopathology findings, which are often critical for NOAEL determination.

11. **Organ Measurements (OM)**: Contains organ weight data, which can indicate target organ toxicity.

12. **Pharmacokinetics Concentrations (PC)**: Contains test article concentration data in biological specimens.

13. **Pharmacokinetics Parameters (PP)**: Contains derived pharmacokinetic parameters.

14. **Subject Characteristics (SC)**: Contains additional subject information not included in demographics.

15. **Vital Signs (VS)**: Contains vital signs measurements, which can indicate physiological effects.

These domains collectively provide the comprehensive dataset needed for thorough toxicological evaluation and NOAEL determination.
