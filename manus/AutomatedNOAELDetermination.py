"""
Use Case 1: Automated NOAEL Determination from Multiple Endpoints

This script demonstrates how to use TxGemma to automatically determine NOAEL values
by analyzing multiple toxicological endpoints simultaneously from SEND datasets.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

# Path to SEND dataset
SEND_DATA_PATH = "/home/ubuntu/noael_project/sample_datasets/phuse-scripts/data/send/CBER-POC-Pilot-Study1-Vaccine"

# Function to load and process SEND datasets
def load_send_domains(base_path):
    """
    Load relevant SEND domains from XPT files
    """
    print("Loading SEND domains...")
    
    # Dictionary to store dataframes
    domains = {}
    
    try:
        # Import necessary libraries for XPT file reading
        from xport import read_xport
        
        # List of domains to load
        domain_files = {
            'dm': 'dm.xpt',    # Demographics
            'ex': 'ex.xpt',    # Exposure (dosing)
            'lb': 'lb.xpt',    # Laboratory test results
            'bw': 'bw.xpt',    # Body weights
            'cl': 'cl.xpt',    # Clinical observations
            'mi': 'mi.xpt',    # Microscopic findings (if available)
            'ma': 'ma.xpt',    # Macroscopic findings (if available)
            'om': 'om.xpt',    # Organ measurements (if available)
            'ts': 'ts.xpt'     # Trial summary
        }
        
        # Load each domain if file exists
        for domain, filename in domain_files.items():
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    domains[domain] = read_xport(f)
                print(f"Loaded {domain} domain with {len(domains[domain])} records")
            else:
                print(f"Warning: {filename} not found")
        
        return domains
    
    except Exception as e:
        print(f"Error loading SEND domains: {e}")
        # For demonstration purposes, create mock data if loading fails
        print("Creating mock data for demonstration...")
        return create_mock_send_data()

# Function to create mock SEND data for demonstration
def create_mock_send_data():
    """
    Create mock SEND data for demonstration purposes
    """
    domains = {}
    
    # Create mock demographics (DM) domain
    domains['dm'] = pd.DataFrame({
        'USUBJID': [f'SUBJ-{i:03d}' for i in range(1, 41)],
        'SEX': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'] * 5,
        'ARMCD': ['C', 'C', 'LD', 'LD', 'MD', 'MD', 'HD', 'HD'] * 5,
        'ARM': ['Control', 'Control', 'Low Dose', 'Low Dose', 
                'Mid Dose', 'Mid Dose', 'High Dose', 'High Dose'] * 5,
        'SPECIES': ['RAT'] * 40,
        'STRAIN': ['WISTAR'] * 40
    })
    
    # Create mock exposure (EX) domain
    dose_levels = {'C': 0, 'LD': 10, 'MD': 50, 'HD': 200}
    ex_data = []
    for subj in domains['dm']['USUBJID']:
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        ex_data.append({
            'USUBJID': subj,
            'EXDOSE': dose_levels[arm],
            'EXDOSU': 'mg/kg/day',
            'EXROUTE': 'ORAL'
        })
    domains['ex'] = pd.DataFrame(ex_data)
    
    # Create mock laboratory results (LB) domain with dose-dependent effects
    lb_data = []
    endpoints = ['ALT', 'AST', 'BUN', 'CREAT', 'WBC', 'RBC', 'HGB', 'PLT']
    baseline_values = {'ALT': 45, 'AST': 80, 'BUN': 15, 'CREAT': 0.6, 
                      'WBC': 10, 'RBC': 8, 'HGB': 15, 'PLT': 800}
    
    # Effect multipliers for each dose group (control, low, mid, high)
    effect_multipliers = {
        'ALT': [1.0, 1.1, 1.3, 1.8],    # Clear dose response
        'AST': [1.0, 1.1, 1.2, 1.6],    # Clear dose response
        'BUN': [1.0, 1.0, 1.1, 1.3],    # Mild effect at high dose
        'CREAT': [1.0, 1.0, 1.0, 1.2],  # Mild effect at high dose
        'WBC': [1.0, 1.0, 1.0, 0.9],    # Mild decrease at high dose
        'RBC': [1.0, 1.0, 1.0, 1.0],    # No effect
        'HGB': [1.0, 1.0, 1.0, 1.0],    # No effect
        'PLT': [1.0, 0.95, 0.9, 0.8]    # Dose-dependent decrease
    }
    
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        
        for endpoint in endpoints:
            # Add biological variability
            base_value = baseline_values[endpoint]
            multiplier = effect_multipliers[endpoint][arm_idx[arm]]
            
            # Add random variation (10% CV)
            value = base_value * multiplier * np.random.normal(1, 0.1)
            
            lb_data.append({
                'USUBJID': subj,
                'LBTEST': endpoint,
                'LBSTRESN': value,
                'LBSTRESU': 'U/L' if endpoint in ['ALT', 'AST'] else 
                            'mg/dL' if endpoint in ['BUN', 'CREAT'] else
                            '10^9/L' if endpoint == 'PLT' else
                            '10^12/L' if endpoint == 'RBC' else
                            'g/dL' if endpoint == 'HGB' else
                            '10^9/L'
            })
    
    domains['lb'] = pd.DataFrame(lb_data)
    
    # Create mock body weight (BW) domain with dose-dependent effects
    bw_data = []
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        sex = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'SEX'].values[0]
        
        # Base weight depends on sex
        base_weight = 250 if sex == 'M' else 200
        
        # Weight effect by dose (control, low, mid, high)
        weight_effect = [1.0, 0.98, 0.95, 0.85]
        
        # Add random variation (5% CV)
        weight = base_weight * weight_effect[arm_idx[arm]] * np.random.normal(1, 0.05)
        
        bw_data.append({
            'USUBJID': subj,
            'BWSTRESN': weight,
            'BWSTRESU': 'g',
            'VISITDY': 28  # Terminal body weight
        })
    
    domains['bw'] = pd.DataFrame(bw_data)
    
    # Create mock trial summary (TS) domain
    domains['ts'] = pd.DataFrame({
        'TSGRPID': ['STUDY-001'],
        'TSPARMCD': ['SPECIES'],
        'TSPARM': ['Species'],
        'TSVAL': ['Rat']
    })
    
    print("Created mock SEND data with the following domains:")
    for domain, df in domains.items():
        print(f"- {domain}: {len(df)} records")
    
    return domains

# Function to extract features from SEND domains
def extract_features(domains):
    """
    Extract relevant features from SEND domains for NOAEL determination
    """
    print("Extracting features from SEND domains...")
    
    features = {}
    
    # Get dose groups
    if 'dm' in domains and 'ex' in domains:
        # Merge DM and EX to get dose information for each subject
        subjects = domains['dm'][['USUBJID', 'ARMCD', 'ARM', 'SEX']]
        if 'EXDOSE' in domains['ex'].columns:
            doses = domains['ex'][['USUBJID', 'EXDOSE']].drop_duplicates()
            subjects = subjects.merge(doses, on='USUBJID', how='left')
            
            # Group subjects by dose
            dose_groups = subjects.groupby('EXDOSE')['USUBJID'].apply(list).to_dict()
            features['dose_groups'] = dose_groups
            features['doses'] = sorted(dose_groups.keys())
            
            print(f"Identified {len(features['doses'])} dose groups: {features['doses']}")
        else:
            print("Warning: EXDOSE column not found in EX domain")
            # Use ARM as a proxy for dose groups
            arm_groups = subjects.groupby('ARMCD')['USUBJID'].apply(list).to_dict()
            features['dose_groups'] = arm_groups
            features['doses'] = sorted(arm_groups.keys())
    
    # Extract laboratory test results
    if 'lb' in domains:
        lb_data = domains['lb']
        
        # Group by test and calculate statistics for each dose group
        if 'LBTEST' in lb_data.columns and 'LBSTRESN' in lb_data.columns:
            lb_features = {}
            
            for test in lb_data['LBTEST'].unique():
                test_data = lb_data[lb_data['LBTEST'] == test]
                
                # Calculate statistics for each dose group
                test_stats = {}
                for dose, subjects in features['dose_groups'].items():
                    dose_data = test_data[test_data['USUBJID'].isin(subjects)]
                    
                    if not dose_data.empty:
                        values = dose_data['LBSTRESN'].values
                        test_stats[dose] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'n': len(values),
                            'values': values.tolist()
                        }
                
                lb_features[test] = test_stats
            
            features['laboratory_tests'] = lb_features
            print(f"Extracted data for {len(lb_features)} laboratory tests")
    
    # Extract body weight data
    if 'bw' in domains:
        bw_data = domains['bw']
        
        if 'BWSTRESN' in bw_data.columns:
            bw_features = {}
            
            # Calculate statistics for each dose group
            for dose, subjects in features['dose_groups'].items():
                dose_data = bw_data[bw_data['USUBJID'].isin(subjects)]
                
                if not dose_data.empty:
                    values = dose_data['BWSTRESN'].values
                    bw_features[dose] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'n': len(values),
                        'values': values.tolist()
                    }
            
            features['body_weights'] = bw_features
            print(f"Extracted body weight data for {len(bw_features)} dose groups")
    
    return features

# Function to analyze dose-response relationships
def analyze_dose_response(features):
    """
    Analyze dose-response relationships for each endpoint
    """
    print("Analyzing dose-response relationships...")
    
    results = {}
    
    # Analyze laboratory test results
    if 'laboratory_tests' in features:
        lb_results = {}
        
        for test, test_data in features['laboratory_tests'].items():
            # Skip if less than 2 dose groups
            if len(test_data) < 2:
                continue
            
            # Get control group data
            control_dose = min(features['doses'])
            control_data = test_data.get(control_dose, {}).get('values', [])
            
            if not control_data:
                continue
            
            # Analyze each dose group compared to control
            dose_effects = {}
            for dose in features['doses']:
                if dose == control_dose:
                    continue
                
                dose_data = test_data.get(dose, {}).get('values', [])
                
                if not dose_data:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_data, dose_data, equal_var=False)
                
                # Calculate percent change from control
                control_mean = np.mean(control_data)
                dose_mean = np.mean(dose_data)
                percent_change = ((dose_mean - control_mean) / control_mean) * 100
                
                # Determine if effect is adverse
                # For demonstration, we'll consider >20% change with p<0.05 as adverse
                is_adverse = (abs(percent_change) > 20) and (p_value < 0.05)
                
                dose_effects[dose] = {
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'is_adverse': is_adverse
                }
            
            lb_results[test] = dose_effects
        
        results['laboratory_tests'] = lb_results
    
    # Analyze body weight data
    if 'body_weights' in features:
        bw_results = {}
        
        # Get control group data
        control_dose = min(features['doses'])
        control_data = features['body_weights'].get(control_dose, {}).get('values', [])
        
        if control_data:
            # Analyze each dose group compared to control
            for dose in features['doses']:
                if dose == control_dose:
                    continue
                
                dose_data = features['body_weights'].get(dose, {}).get('values', [])
                
                if not dose_data:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_data, dose_data, equal_var=False)
                
                # Calculate percent change from control
                control_mean = np.mean(control_data)
                dose_mean = np.mean(dose_data)
                percent_change = ((dose_mean - control_mean) / control_mean) * 100
                
                # Determine if effect is adverse
                # For body weight, a decrease >10% with p<0.05 is typically considered adverse
                is_adverse = (percent_change < -10) and (p_value < 0.05)
                
                bw_results[dose] = {
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'is_adverse': is_adverse
                }
        
        results['body_weights'] = bw_results
    
    return results

# Function to determine NOAEL based on analysis results
def determine_noael(features, analysis_results):
    """
    Determine NOAEL based on analysis results
    """
    print("Determining NOAEL...")
    
    # Get sorted dose levels
    doses = sorted(features['doses'])
    
    # Track adverse effects by dose
    adverse_effects = {dose: [] for dose in doses}
    
    # Check laboratory test results
    if 'laboratory_tests' in analysis_results:
        for test, test_results in analysis_results['laboratory_tests'].items():
            for dose, result in test_results.items():
                if result['is_adverse']:
                    adverse_effects[dose].append({
                        'endpoint': test,
                        'percent_change': result['percent_change'],
                        'p_value': result['p_value']
                    })
    
    # Check body weight results
    if 'body_weights' in analysis_results:
        for dose, result in analysis_results['body_weights'].items():
            if result['is_adverse']:
                adverse_effects[dose].append({
                    'endpoint': 'Body Weight',
                    'percent_change': result['percent_change'],
                    'p_value': result['p_value']
                })
    
    # Determine NOAEL as highest dose with no adverse effects
    noael = None
    for dose in doses:
        if not adverse_effects[dose]:
            noael = dose
        else:
            break
    
    # Prepare results
    noael_results = {
        'noael': noael,
        'adverse_effects': adverse_effects,
        'doses': doses
    }
    
    return noael_results

# Function to prepare TxGemma input
def prepare_txgemma_input(features, analysis_results, noael_results):
    """
    Prepare input for TxGemma model
    """
    print("Preparing TxGemma input...")
    
    # Create a summary of the study and findings
    study_summary = {
        'study_design': {
            'species': 'Rat',  # Assuming rat for demonstration
            'duration': '28 days',  # Assuming 28-day study for demonstration
            'dose_groups': features['doses'],
            'endpoints_evaluated': list(features.get('laboratory_tests', {}).keys()) + ['Body Weight']
        },
        'findings': {
            'laboratory_tests': {},
            'body_weight': {}
        },
        'preliminary_noael': noael_results['noael']
    }
    
    # Add laboratory test findings
    if 'laboratory_tests' in analysis_results:
        for test, test_results in analysis_results['laboratory_tests'].items():
            study_summary['findings']['laboratory_tests'][test] = {}
            for dose, result in test_results.items():
                study_summary['findings']['laboratory_tests'][test][dose] = {
                    'percent_change': round(result['percent_change'], 1),
                    'p_value': round(result['p_value'], 3),
                    'is_adverse': result['is_adverse']
                }
    
    # Add body weight findings
    if 'body_weights' in analysis_results:
        for dose, result in analysis_results['body_weights'].items():
            study_summary['findings']['body_weight'][dose] = {
                'percent_change': round(result['percent_change'], 1),
                'p_value': round(result['p_value'], 3),
                'is_adverse': result['is_adverse']
            }
    
    # Convert to JSON string
    study_summary_json = json.dumps(study_summary, indent=2)
    
    # Create prompt for TxGemma
    prompt = f"""
You are a toxicology expert analyzing data from a preclinical safety study to determine the No Observed Adverse Effect Level (NOAEL).

Below is a summary of the study design and findings:

{study_summary_json}

Based on this data, please:
1. Determine the NOAEL for this study
2. Explain your reasoning, including which endpoints were most important in your determination
3. Assess your confidence in this NOAEL determination (high, medium, or low)
4. Suggest any additional analyses that might strengthen the NOAEL determination

Your response should be structured as a JSON object with the following fields:
- noael: the determined NOAEL value
- reasoning: explanation of your determination
- key_endpoints: list of endpoints that were most important
- confidence: confidence level (high, medium, low)
- additional_analyses: suggestions for additional analyses
"""
    
    return prompt

# Function to simulate TxGemma response
def simulate_txgemma_response(prompt):
    """
    Simulate TxGemma response for demonstration purposes
    """
    print("Simulating TxGemma response...")
    
    # Extract study summary from prompt
    try:
        start_idx = prompt.find('{')
        end_idx = prompt.find('\n\nBased on this data')
        study_summary_json = prompt[start_idx:end_idx]
        study_summary = json.loads(study_summary_json)
    except:
        print("Error parsing study summary from prompt")
        study_summary = {}
    
    # Extract preliminary NOAEL
    preliminary_noael = study_summary.get('preliminary_noael')
    
    # Extract findings
    findings = study_summary.get('findings', {})
    lab_tests = findings.get('laboratory_tests', {})
    body_weight = findings.get('body_weight', {})
    
    # Identify key adverse findings
    adverse_findings = []
    
    for test, test_results in lab_tests.items():
        for dose, result in test_results.items():
            if result.get('is_adverse', False):
                adverse_findings.append({
                    'endpoint': test,
                    'dose': dose,
                    'percent_change': result.get('percent_change'),
                    'p_value': result.get('p_value')
                })
    
    for dose, result in body_weight.items():
        if result.get('is_adverse', False):
            adverse_findings.append({
                'endpoint': 'Body Weight',
                'dose': dose,
                'percent_change': result.get('percent_change'),
                'p_value': result.get('p_value')
            })
    
    # Sort adverse findings by dose
    adverse_findings.sort(key=lambda x: float(x['dose']))
    
    # Determine key endpoints
    key_endpoints = list(set([finding['endpoint'] for finding in adverse_findings]))
    
    # Generate reasoning based on findings
    if adverse_findings:
        lowest_adverse_dose = adverse_findings[0]['dose']
        lowest_adverse_endpoints = [f['endpoint'] for f in adverse_findings if f['dose'] == lowest_adverse_dose]
        
        reasoning = f"The NOAEL is determined to be {preliminary_noael} based on adverse findings at the next higher dose level ({lowest_adverse_dose}). "
        reasoning += f"At {lowest_adverse_dose}, adverse effects were observed in the following endpoints: {', '.join(lowest_adverse_endpoints)}. "
        
        # Add details about specific findings
        for endpoint in lowest_adverse_endpoints:
            findings_for_endpoint = [f for f in adverse_findings if f['endpoint'] == endpoint and f['dose'] == lowest_adverse_dose]
            if findings_for_endpoint:
                finding = findings_for_endpoint[0]
                reasoning += f"For {endpoint}, a {abs(finding['percent_change']):.1f}% {'increase' if finding['percent_change'] > 0 else 'decrease'} "
                reasoning += f"was observed (p={finding['p_value']:.3f}), which is considered toxicologically significant. "
    else:
        # No adverse findings
        highest_dose = max([float(d) for d in study_summary.get('study_design', {}).get('dose_groups', [0])])
        reasoning = f"No adverse effects were observed at any dose level, including the highest tested dose of {highest_dose}. "
        reasoning += "Therefore, the NOAEL is determined to be the highest tested dose."
    
    # Determine confidence level
    if not adverse_findings:
        confidence = "medium"  # Medium confidence when no adverse effects (might need higher doses)
        confidence_explanation = "Medium confidence because no adverse effects were observed at any dose level, suggesting that higher doses might be needed to establish a complete toxicity profile."
    elif len(key_endpoints) >= 3:
        confidence = "high"  # High confidence when multiple endpoints show effects
        confidence_explanation = "High confidence due to consistent adverse effects observed across multiple endpoints, providing a clear dose-response relationship."
    elif any(f['p_value'] < 0.01 for f in adverse_findings):
        confidence = "high"  # High confidence when strong statistical significance
        confidence_explanation = "High confidence due to statistically significant findings with p-values below 0.01, indicating clear treatment-related effects."
    else:
        confidence = "medium"  # Medium confidence in other cases
        confidence_explanation = "Medium confidence based on the limited number of affected endpoints and moderate statistical significance of findings."
    
    # Suggest additional analyses
    additional_analyses = [
        "Conduct histopathological examination of affected organs to confirm and characterize observed effects",
        "Evaluate dose-response relationships using benchmark dose modeling",
        "Analyze potential sex differences in response to treatment",
        "Consider time-course analysis to determine if effects are progressive or adaptive"
    ]
    
    # Create simulated response
    response = {
        "noael": preliminary_noael,
        "reasoning": reasoning + confidence_explanation,
        "key_endpoints": key_endpoints,
        "confidence": confidence,
        "additional_analyses": additional_analyses
    }
    
    return json.dumps(response, indent=2)

# Function to visualize results
def visualize_results(features, analysis_results, noael_results, txgemma_response):
    """
    Create visualizations of the results
    """
    print("Creating visualizations...")
    
    # Parse TxGemma response
    try:
        txgemma_data = json.loads(txgemma_response)
    except:
        print("Error parsing TxGemma response")
        txgemma_data = {}
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NOAEL Determination Results', fontsize=16)
    
    # Plot 1: Laboratory test results
    ax = axs[0, 0]
    
    if 'laboratory_tests' in features:
        # Select a few key tests to display
        key_tests = list(features['laboratory_tests'].keys())[:4]  # Limit to 4 tests for clarity
        
        for i, test in enumerate(key_tests):
            test_data = features['laboratory_tests'][test]
            
            # Extract means and standard errors
            doses = []
            means = []
            sems = []
            
            for dose in sorted(test_data.keys()):
                doses.append(float(dose))
                means.append(test_data[dose]['mean'])
                sems.append(test_data[dose]['std'] / np.sqrt(test_data[dose]['n']))
            
            # Plot with error bars
            ax.errorbar(doses, means, yerr=sems, marker='o', label=test)
        
        ax.set_xlabel('Dose')
        ax.set_ylabel('Value')
        ax.set_title('Laboratory Test Results')
        ax.legend()
        
        # Add NOAEL line
        if noael_results['noael'] is not None:
            ax.axvline(x=float(noael_results['noael']), color='green', linestyle='--', 
                      label=f"NOAEL = {noael_results['noael']}")
            ax.legend()
    
    # Plot 2: Body weight results
    ax = axs[0, 1]
    
    if 'body_weights' in features:
        # Extract means and standard errors
        doses = []
        means = []
        sems = []
        
        for dose in sorted(features['body_weights'].keys()):
            doses.append(float(dose))
            means.append(features['body_weights'][dose]['mean'])
            sems.append(features['body_weights'][dose]['std'] / np.sqrt(features['body_weights'][dose]['n']))
        
        # Plot with error bars
        ax.errorbar(doses, means, yerr=sems, marker='o', color='blue')
        
        ax.set_xlabel('Dose')
        ax.set_ylabel('Body Weight (g)')
        ax.set_title('Body Weight Results')
        
        # Add NOAEL line
        if noael_results['noael'] is not None:
            ax.axvline(x=float(noael_results['noael']), color='green', linestyle='--', 
                      label=f"NOAEL = {noael_results['noael']}")
            ax.legend()
    
    # Plot 3: Adverse effects summary
    ax = axs[1, 0]
    
    # Count adverse effects by dose
    doses = sorted(noael_results['adverse_effects'].keys())
    adverse_counts = [len(noael_results['adverse_effects'][dose]) for dose in doses]
    
    # Create bar chart
    bars = ax.bar([str(dose) for dose in doses], adverse_counts, color='lightcoral')
    
    # Add NOAEL marker
    if noael_results['noael'] is not None:
        noael_idx = doses.index(noael_results['noael'])
        bars[noael_idx].set_color('lightgreen')
        
        # Add text label
        ax.text(noael_idx, adverse_counts[noael_idx] + 0.1, 'NOAEL', 
               ha='center', va='bottom', color='green', fontweight='bold')
    
    ax.set_xlabel('Dose')
    ax.set_ylabel('Number of Adverse Effects')
    ax.set_title('Adverse Effects by Dose')
    
    # Plot 4: TxGemma assessment
    ax = axs[1, 1]
    ax.axis('off')  # Turn off axis
    
    # Create text summary from TxGemma response
    if txgemma_data:
        text = "TxGemma Assessment:\n\n"
        text += f"NOAEL: {txgemma_data.get('noael', 'Not determined')}\n\n"
        text += f"Confidence: {txgemma_data.get('confidence', 'Not specified')}\n\n"
        text += "Key Endpoints:\n"
        
        for endpoint in txgemma_data.get('key_endpoints', []):
            text += f"- {endpoint}\n"
        
        text += f"\nReasoning:\n{txgemma_data.get('reasoning', 'Not provided')}"
        
        ax.text(0, 1, text, va='top', ha='left', wrap=True, fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('/home/ubuntu/noael_project/demo_code/use_case1_results.png')
    print("Visualization saved to /home/ubuntu/noael_project/demo_code/use_case1_results.png")
    
    return '/home/ubuntu/noael_project/demo_code/use_case1_results.png'

# Main function
def main():
    """
    Main function to demonstrate TxGemma for NOAEL determination
    """
    print("Starting NOAEL determination demonstration using TxGemma...")
    
    # Step 1: Load SEND domains
    domains = load_send_domains(SEND_DATA_PATH)
    
    # Step 2: Extract features from SEND domains
    features = extract_features(domains)
    
    # Step 3: Analyze dose-response relationships
    analysis_results = analyze_dose_response(features)
    
    # Step 4: Determine preliminary NOAEL
    noael_results = determine_noael(features, analysis_results)
    print(f"Preliminary NOAEL determination: {noael_results['noael']}")
    
    # Step 5: Prepare input for TxGemma
    txgemma_prompt = prepare_txgemma_input(features, analysis_results, noael_results)
    
    # Step 6: Get TxGemma response (simulated for demonstration)
    txgemma_response = simulate_txgemma_response(txgemma_prompt)
    print("\nTxGemma Response:")
    print(txgemma_response)
    
    # Step 7: Visualize results
    visualization_path = visualize_results(features, analysis_results, noael_results, txgemma_response)
    
    print("\nDemonstration completed successfully!")
    print(f"Results visualization saved to: {visualization_path}")
    
    return {
        'features': features,
        'analysis_results': analysis_results,
        'noael_results': noael_results,
        'txgemma_response': json.loads(txgemma_response),
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    main()
