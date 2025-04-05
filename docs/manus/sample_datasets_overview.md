# Sample SEND Datasets for NOAEL Prediction

This document provides an overview of the sample SEND (Standard for Exchange of Nonclinical Data) datasets that have been collected for the NOAEL prediction project.

## Dataset Source

The sample datasets were obtained from the PHUSE (Pharmaceutical Users Software Exchange) GitHub repository, which contains a collection of publicly available SEND-formatted datasets for various types of toxicology studies.

**Repository URL**: https://github.com/phuse-org/phuse-scripts

## Available Datasets

The following SEND datasets are available in the repository:

1. **CBER-POC-Pilot-Study1-Vaccine** - Vaccine toxicology study
2. **CBER-POC-Pilot-Study2-Vaccine** - Vaccine toxicology study
3. **CBER-POC-Pilot-Study3-Gene-Therapy** - Gene therapy toxicology study
4. **CBER-POC-Pilot-Study4-Vaccine** - Vaccine toxicology study
5. **CBER-POC-Pilot-Study5** - General toxicology study
6. **CDISC-Safety-Pharmacology-POC** - Safety pharmacology study
7. **CJ16050** - General toxicology study
8. **CJUGSEND00** - General toxicology study
9. **FFU-Contribution-to-FDA** - FDA submission example
10. **JSON-CBER-POC-Pilot-Study3-Gene-Therapy** - JSON format of gene therapy study
11. **Nimble** - General toxicology study
12. **PDS** - General toxicology study
13. **PointCross** - General toxicology study
14. **SENDIG3.1.1excel** - Excel format of SEND datasets
15. **instem** - General toxicology study

## Dataset Structure

Each dataset follows the SEND standard structure, which organizes data into domain-specific files. Taking CBER-POC-Pilot-Study1-Vaccine as an example, it contains the following files:

- **bg.xpt** - Body Weight Gain domain
- **bw.xpt** - Body Weight domain
- **cl.xpt** - Clinical Observations domain
- **co.xpt** - Comments domain
- **dm.xpt** - Demographics domain
- **ds.xpt** - Disposition domain
- **ex.xpt** - Exposure domain
- **is.xpt** - Immunogenicity Specimen Assessment domain
- **lb.xpt** - Laboratory Test Results domain
- **se.xpt** - Subject Elements domain
- **ta.xpt** - Trial Arms domain
- **te.xpt** - Trial Elements domain
- **ts.xpt** - Trial Summary domain
- **tx.xpt** - Trial Sets domain
- **supp*.xpt** - Supplemental Qualifier datasets for various domains

Additionally, each dataset includes supporting documentation:
- **define.xml** - Dataset definition file
- **define2-0-0.xsl** - XSL stylesheet for define.xml
- **nsdrg.pdf** - Nonclinical Study Data Reviewers Guide
- Study report PDFs and other documentation

## Relevance to NOAEL Prediction

These datasets are particularly valuable for NOAEL prediction because they contain:

1. **Dose Information**: The exposure (EX) domain contains dosing information, which is critical for establishing dose-response relationships.

2. **Toxicity Endpoints**: The laboratory results (LB) domain and clinical observations (CL) domain contain measurements of various toxicity endpoints.

3. **Study Design**: The trial summary (TS) and trial elements (TE) domains provide information about study design, which is important for contextualizing the toxicity findings.

4. **Animal Demographics**: The demographics (DM) domain provides information about the test subjects, which can be important covariates in the prediction model.

5. **Body Weight Data**: The body weight (BW) domain contains information about animal weights, which are often used as indicators of general toxicity.

## Data Format

The datasets are provided in XPT format (SAS Transport File Format), which is the standard format for SEND submissions to regulatory agencies. These files can be read using various tools:

- SAS software
- R (using the `haven` or `xport` packages)
- Python (using the `xport` package with pandas)
- Specialized SEND viewers and analysis tools

## Usage for NOAEL Prediction

For the NOAEL prediction project, these datasets will be used to:

1. **Train the TxGemma Model**: The datasets will be processed and used to fine-tune the TxGemma model for NOAEL prediction.

2. **Validate Prediction Accuracy**: A portion of the datasets will be reserved for validation to assess the accuracy of the NOAEL predictions.

3. **Develop Data Processing Pipelines**: The datasets will help in developing robust data processing pipelines that can handle the complex structure of SEND data.

4. **Test the Frontend Interface**: The datasets will provide real-world examples to test the visualization capabilities of the frontend interface.

## Technical Considerations

When working with these datasets, several technical considerations should be kept in mind:

1. **Data Compatibility**: The datasets follow different versions of the SEND Implementation Guide (SENDIG), which may require version-specific handling.

2. **Missing Data**: Like most real-world datasets, these may contain missing values that need to be handled appropriately.

3. **Data Integration**: For comprehensive NOAEL prediction, data from multiple domains need to be integrated, which requires careful mapping and merging.

4. **Controlled Terminology**: SEND uses controlled terminology for many fields, which needs to be properly interpreted for accurate analysis.

## Conclusion

The sample SEND datasets obtained from the PHUSE GitHub repository provide a comprehensive resource for developing and testing the NOAEL prediction tool. They cover various types of toxicology studies and contain all the necessary domains for NOAEL determination. These datasets will be instrumental in training the TxGemma model and developing a robust prediction system.
