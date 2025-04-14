# Enhanced TxGemma Prompts for SEND Data Analysis

This document contains enhanced prompt templates for TxGemma to analyze SEND datasets. These prompts have been carefully designed to extract the maximum toxicological insights from available SEND domains and are implemented in the `enhanced_processor.py` module.

This document outlines improved prompt structures for TxGemma analysis of SEND datasets, incorporating more of the available domains and organizing information for optimal toxicological assessment.

## 1. Comprehensive Toxicology Assessment Prompt

```
You are a toxicologist analyzing standardized SEND (Standard for Exchange of Nonclinical Data) datasets from a preclinical toxicology study. Analyze the following study data to help determine the No Observed Adverse Effect Level (NOAEL):

## Study Metadata
- Species: {species}
- Strain: {strain}
- Sex(es): {sexes}
- Study Duration: {duration}
- Route of Administration: {route}
- Test Article: {test_article}
- Dose Groups: {dose_groups_with_units}

## Body Weight Analysis
{body_weight_summary}

## Clinical Observations
{clinical_observations_summary}

## Laboratory Tests
{lab_tests_summary}

## Macroscopic Findings (Gross Pathology)
{macroscopic_findings_summary}

## Microscopic Findings (Histopathology)
{microscopic_findings_summary}

## Organ Measurements
{organ_measurements_summary}

Based on this data:
1. Identify the key toxicological findings across all domains, organizing them by dose group.
2. Determine whether there is a dose-response relationship for each finding.
3. Assess the toxicological significance of each finding in the context of this specific study.
4. Based on your comprehensive assessment, determine the most likely NOAEL and provide your reasoning.
5. Identify any data gaps or limitations that impact your confidence in the NOAEL determination.

Please structure your response with clear section headings and include the dose units ({dose_unit_str}) in your NOAEL determination.
```

## 2. Target Organ Prediction Prompt

```
As a toxicology expert analyzing SEND data from a preclinical study, identify potential target organs of toxicity based on the following information:

## Study Basics
- Test Article: {test_article}
- Species/Strain: {species}/{strain}
- Sex(es): {sexes}
- Route: {route}
- Duration: {duration}
- Dose Groups: {dose_groups_with_units}

## Laboratory Results (Clinical Pathology)
{key_clinical_pathology_findings}

## Organ Weights
{organ_weight_findings}

## Macroscopic Findings
{macroscopic_findings}

## Microscopic Findings
{microscopic_findings}

Based on this multidomain evidence:
1. Identify the most likely target organs affected by the test article.
2. For each target organ, provide:
   a. The supporting evidence across multiple domains
   b. The lowest dose at which effects were observed
   c. The severity and reversibility (if data available)
3. Rank the identified target organs by the weight of evidence.
4. Indicate which findings appear treatment-related versus incidental.

Provide a concise summary of your assessment, suitable for inclusion in a toxicology report.
```

## 3. Treatment-Related Effects Query Prompt

```
As a toxicologist reviewing SEND data, determine which findings are treatment-related versus incidental. Analyze the following preclinical toxicology findings:

## Study Design
- Test Article: {test_article}
- Control Group Size: {control_group_size} animals
- Treatment Groups: {treatment_groups_with_sizes}
- Duration: {duration}

## Findings Summary
{findings_across_domains_by_dose_group}

## Historical Control Data (if available)
{historical_control_ranges}

For each finding:
1. Assess whether it shows a dose-response relationship.
2. Evaluate statistical significance versus biological relevance.
3. Compare to historical control ranges (if available).
4. Consider consistency across related endpoints.
5. Determine if the finding is likely treatment-related or incidental.

For findings you determine to be treatment-related:
- Identify the lowest dose at which the effect was observed (LOAEL for that endpoint).
- Assess whether there was a NOAEL for that specific endpoint.

Present your analysis as a structured assessment with clear reasoning for your determinations.
```

## 4. Cross-Species Toxicity Risk Assessment Prompt

```
You are evaluating the cross-species relevance of toxicity findings from the following preclinical study:

## Study Information
- Test Article: {test_article}
- Study Species: {species}
- Study Findings Summary:
{key_findings_with_doses}

## Mechanism of Action (if known)
{mechanism_information}

Based on these findings:
1. Assess which findings may be species-specific versus relevant to humans.
2. For findings likely relevant to humans, estimate the human equivalent doses (if possible).
3. Identify any findings that might warrant specific monitoring in clinical trials.
4. Recommend additional studies or analyses that would strengthen the cross-species relevance assessment.

Provide a concise risk assessment focusing on the translatability of these findings to humans.
```

## 5. NOAEL Determination with Uncertainty Quantification Prompt

```
As a toxicologist determining the NOAEL (No Observed Adverse Effect Level) for a regulatory submission, analyze the following preclinical data and quantify your uncertainty:

## Study Summary
- Test Article: {test_article}
- Species/Strain: {species}/{strain}
- Sex(es): {sexes}
- Route: {route}
- Duration: {duration}
- Dose Groups: {dose_groups_with_units}

## Key Findings by Domain
{comprehensive_findings_by_domain}

Based on this dataset:
1. Provide your NOAEL determination with dose in {dose_unit_str}.
2. Assign a confidence level (High/Medium/Low) to your determination.
3. Identify the critical effect(s) that define the NOAEL.
4. Discuss alternative interpretations that regulators might consider.
5. Recommend additional analyses or studies that would increase confidence in the NOAEL.

Structure your response as a scientific assessment with clearly labeled sections addressing each point above.
```

## Implementation Recommendations

1. **Domain Mapping**: Create a comprehensive mapping between SEND domains and specific toxicological endpoints:
   - BW → Body weight changes (systemic toxicity)
   - LB → Organ function (liver enzymes, kidney markers)
   - CL → Clinical signs (CNS effects, physical condition)
   - MI → Target organ toxicity (histopathology)
   - OM → Organ weight changes (target organ effects)

2. **Data Enrichment**:
   - Calculate dose-normalized effects where appropriate
   - Include statistical significance for key findings when available
   - Add severity grading for microscopic findings
   - Provide historical control ranges when available

3. **Context Addition**:
   - Include study design parameters from TS domain
   - Add species-specific normal ranges for lab tests
   - Incorporate known information about the test article class

4. **Prompt Tuning**:
   - Start with the comprehensive assessment prompt and tailor to specific study designs
   - Adapt domain-specific sections based on data availability
   - Consider separate prompts for different study types (acute, subchronic, chronic)