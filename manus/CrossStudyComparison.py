"""
Use Case 3: Cross-Study NOAEL Comparison and Consistency Analysis

This script demonstrates how to use TxGemma to compare NOAEL determinations across
multiple studies of similar compounds or across different studies of the same compound.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

# Paths to multiple SEND datasets
SEND_DATA_PATHS = [
    "/home/ubuntu/noael_project/sample_datasets/phuse-scripts/data/send/CBER-POC-Pilot-Study1-Vaccine",
    "/home/ubuntu/noael_project/sample_datasets/phuse-scripts/data/send/CBER-POC-Pilot-Study2-Vaccine",
    "/home/ubuntu/noael_project/sample_datasets/phuse-scripts/data/send/CBER-POC-Pilot-Study4-Vaccine"
]

# Function to load and process SEND datasets
def load_send_domains(base_path, study_id=None):
    """
    Load relevant SEND domains from XPT files
    
    Parameters:
    -----------
    base_path : str
        Path to SEND dataset
    study_id : str, optional
        Study identifier
    """
    print(f"Loading SEND domains for study {study_id or 'Unknown'}...")
    
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
        
        # Add study_id to each domain if provided
        if study_id:
            for domain in domains:
                domains[domain]['STUDYID'] = study_id
        
        return domains
    
    except Exception as e:
        print(f"Error loading SEND domains: {e}")
        # For demonstration purposes, create mock data if loading fails
        print("Creating mock data for demonstration...")
        return create_mock_send_data(study_id)

# Function to create mock SEND data for demonstration
def create_mock_send_data(study_id=None):
    """
    Create mock SEND data for demonstration purposes
    
    Parameters:
    -----------
    study_id : str, optional
        Study identifier
    """
    domains = {}
    
    # Set study parameters based on study_id
    if not study_id:
        study_id = "STUDY-001"
    
    # Vary parameters based on study_id to simulate different studies
    if study_id == "STUDY-001":
        species = "RAT"
        strain = "WISTAR"
        duration = 28
        dose_levels = {'C': 0, 'LD': 10, 'MD': 50, 'HD': 200}
        effect_strength = 1.0  # Baseline effect strength
    elif study_id == "STUDY-002":
        species = "RAT"
        strain = "SPRAGUE-DAWLEY"
        duration = 28
        dose_levels = {'C': 0, 'LD': 5, 'MD': 25, 'HD': 100}
        effect_strength = 0.8  # Slightly weaker effects
    elif study_id == "STUDY-003":
        species = "RAT"
        strain = "WISTAR"
        duration = 90
        dose_levels = {'C': 0, 'LD': 5, 'MD': 25, 'HD': 100}
        effect_strength = 1.2  # Slightly stronger effects due to longer duration
    else:
        species = "RAT"
        strain = "WISTAR"
        duration = 28
        dose_levels = {'C': 0, 'LD': 10, 'MD': 50, 'HD': 200}
        effect_strength = 1.0
    
    # Create mock demographics (DM) domain
    domains['dm'] = pd.DataFrame({
        'USUBJID': [f'{study_id}-{i:03d}' for i in range(1, 41)],
        'SEX': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'] * 5,
        'ARMCD': ['C', 'C', 'LD', 'LD', 'MD', 'MD', 'HD', 'HD'] * 5,
        'ARM': ['Control', 'Control', 'Low Dose', 'Low Dose', 
                'Mid Dose', 'Mid Dose', 'High Dose', 'High Dose'] * 5,
        'SPECIES': [species] * 40,
        'STRAIN': [strain] * 40,
        'STUDYID': [study_id] * 40
    })
    
    # Create mock exposure (EX) domain
    ex_data = []
    for subj in domains['dm']['USUBJID']:
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        ex_data.append({
            'USUBJID': subj,
            'EXDOSE': dose_levels[arm],
            'EXDOSU': 'mg/kg/day',
            'EXROUTE': 'ORAL',
            'STUDYID': study_id
        })
    domains['ex'] = pd.DataFrame(ex_data)
    
    # Create mock laboratory results (LB) domain with dose-dependent effects
    lb_data = []
    endpoints = ['ALT', 'AST', 'BUN', 'CREAT', 'WBC', 'RBC', 'HGB', 'PLT']
    baseline_values = {'ALT': 45, 'AST': 80, 'BUN': 15, 'CREAT': 0.6, 
                      'WBC': 10, 'RBC': 8, 'HGB': 15, 'PLT': 800}
    
    # Effect multipliers for each dose group (control, low, mid, high)
    # Adjust by effect_strength to simulate different study sensitivities
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
    
    # Adjust effect multipliers by effect_strength
    for endpoint in effect_multipliers:
        for i in range(1, 4):  # Skip control group
            # Adjust effect size while preserving direction
            if effect_multipliers[endpoint][i] > 1.0:
                effect_multipliers[endpoint][i] = 1.0 + (effect_multipliers[endpoint][i] - 1.0) * effect_strength
            elif effect_multipliers[endpoint][i] < 1.0:
                effect_multipliers[endpoint][i] = 1.0 - (1.0 - effect_multipliers[endpoint][i]) * effect_strength
    
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
                            '10^9/L',
                'STUDYID': study_id
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
        
        # Adjust effect by effect_strength
        for i in range(1, 4):  # Skip control group
            weight_effect[i] = 1.0 - (1.0 - weight_effect[i]) * effect_strength
        
        # Add random variation (5% CV)
        weight = base_weight * weight_effect[arm_idx[arm]] * np.random.normal(1, 0.05)
        
        bw_data.append({
            'USUBJID': subj,
            'BWSTRESN': weight,
            'BWSTRESU': 'g',
            'VISITDY': duration,  # Terminal body weight
            'STUDYID': study_id
        })
    
    domains['bw'] = pd.DataFrame(bw_data)
    
    # Create mock trial summary (TS) domain
    domains['ts'] = pd.DataFrame({
        'TSGRPID': [study_id],
        'TSPARMCD': ['SPECIES', 'STRAIN', 'DURATION'],
        'TSPARM': ['Species', 'Strain', 'Study Duration'],
        'TSVAL': [species, strain, str(duration)],
        'STUDYID': [study_id, study_id, study_id]
    })
    
    print(f"Created mock SEND data for study {study_id} with the following domains:")
    for domain, df in domains.items():
        print(f"- {domain}: {len(df)} records")
    
    return domains

# Function to extract features from SEND domains
def extract_features(domains, study_id=None):
    """
    Extract relevant features from SEND domains for NOAEL determination
    
    Parameters:
    -----------
    domains : dict
        Dictionary of SEND domain dataframes
    study_id : str, optional
        Study identifier
    """
    print(f"Extracting features for study {study_id or 'Unknown'}...")
    
    features = {}
    
    # Add study_id to features
    if study_id:
        features['study_id'] = study_id
    elif 'dm' in domains and 'STUDYID' in domains['dm'].columns:
        features['study_id'] = domains['dm']['STUDYID'].iloc[0]
    else:
        features['study_id'] = 'Unknown'
    
    # Extract study design information
    if 'ts' in domains:
        ts_data = domains['ts']
        
        study_design = {}
        
        # Extract key parameters
        param_mapping = {
            'SPECIES': 'species',
            'STRAIN': 'strain',
            'DURATION': 'duration',
            'ROUTE': 'route',
            'DOSFRQ': 'dosing_frequency'
        }
        
        for param, feature in param_mapping.items():
            if 'TSPARMCD' in ts_data.columns and param in ts_data['TSPARMCD'].values:
                value = ts_data.loc[ts_data['TSPARMCD'] == param, 'TSVAL'].iloc[0]
                study_design[feature] = value
        
        features['study_design'] = study_design
    
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
    
    Parameters:
    -----------
    features : dict
        Features extracted from SEND domains
    """
    print(f"Analyzing dose-response relationships for study {features.get('study_id', 'Unknown')}...")
    
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
    
    Parameters:
    -----------
    features : dict
        Features extracted from SEND domains
    analysis_results : dict
        Results of dose-response analysis
    """
    print(f"Determining NOAEL for study {features.get('study_id', 'Unknown')}...")
    
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
        'study_id': features.get('study_id', 'Unknown'),
        'noael': noael,
        'adverse_effects': adverse_effects,
        'doses': doses,
        'study_design': features.get('study_design', {})
    }
    
    return noael_results

# Function to normalize NOAEL values across studies
def normalize_noael_values(noael_results_list):
    """
    Normalize NOAEL values across studies to account for differences in study design
    
    Parameters:
    -----------
    noael_results_list : list
        List of NOAEL results from multiple studies
    """
    print("Normalizing NOAEL values across studies...")
    
    normalized_results = []
    
    # Extract study designs and NOAEL values
    study_designs = []
    noael_values = []
    
    for result in noael_results_list:
        study_designs.append(result.get('study_design', {}))
        noael_values.append(result.get('noael'))
    
    # Check if normalization is needed
    need_normalization = False
    
    # Check for differences in species/strain
    species_set = set(design.get('species') for design in study_designs if 'species' in design)
    strain_set = set(design.get('strain') for design in study_designs if 'strain' in design)
    
    if len(species_set) > 1 or len(strain_set) > 1:
        need_normalization = True
    
    # Check for differences in duration
    duration_set = set(design.get('duration') for design in study_designs if 'duration' in design)
    
    if len(duration_set) > 1:
        need_normalization = True
    
    # Apply normalization if needed
    if need_normalization:
        for i, result in enumerate(noael_results_list):
            normalized_result = result.copy()
            
            # Apply normalization factors
            normalization_factor = 1.0
            
            # Adjust for species/strain differences
            # This is a simplified approach - in reality, more complex allometric scaling would be used
            if 'species' in study_designs[i]:
                species = study_designs[i]['species']
                if species == 'MOUSE':
                    normalization_factor *= 0.2  # Approximate scaling factor for mouse to rat
            
            # Adjust for duration differences
            if 'duration' in study_designs[i]:
                try:
                    duration = int(study_designs[i]['duration'])
                    if duration == 90:
                        normalization_factor *= 0.5  # Approximate scaling factor for 90-day to 28-day study
                    elif duration == 180:
                        normalization_factor *= 0.3  # Approximate scaling factor for 180-day to 28-day study
                except:
                    pass
            
            # Apply normalization to NOAEL
            if normalized_result['noael'] is not None:
                normalized_result['normalized_noael'] = normalized_result['noael'] * normalization_factor
            else:
                normalized_result['normalized_noael'] = None
            
            normalized_result['normalization_factor'] = normalization_factor
            normalized_results.append(normalized_result)
    else:
        # No normalization needed
        for result in noael_results_list:
            normalized_result = result.copy()
            normalized_result['normalized_noael'] = normalized_result['noael']
            normalized_result['normalization_factor'] = 1.0
            normalized_results.append(normalized_result)
    
    return normalized_results

# Function to compare NOAEL determinations across studies
def compare_noael_determinations(normalized_results):
    """
    Compare NOAEL determinations across studies
    
    Parameters:
    -----------
    normalized_results : list
        List of normalized NOAEL results from multiple studies
    """
    print("Comparing NOAEL determinations across studies...")
    
    comparison_results = {
        'study_ids': [result['study_id'] for result in normalized_results],
        'noael_values': [result['noael'] for result in normalized_results],
        'normalized_noael_values': [result['normalized_noael'] for result in normalized_results],
        'normalization_factors': [result['normalization_factor'] for result in normalized_results],
        'study_designs': [result.get('study_design', {}) for result in normalized_results]
    }
    
    # Calculate summary statistics
    valid_normalized_values = [v for v in comparison_results['normalized_noael_values'] if v is not None]
    
    if valid_normalized_values:
        comparison_results['mean_normalized_noael'] = np.mean(valid_normalized_values)
        comparison_results['median_normalized_noael'] = np.median(valid_normalized_values)
        comparison_results['std_normalized_noael'] = np.std(valid_normalized_values)
        comparison_results['cv_normalized_noael'] = (comparison_results['std_normalized_noael'] / 
                                                    comparison_results['mean_normalized_noael'] * 100)
    
    # Identify outliers (studies with NOAEL values > 2 SD from mean)
    if len(valid_normalized_values) > 2 and 'mean_normalized_noael' in comparison_results:
        mean = comparison_results['mean_normalized_noael']
        std = comparison_results['std_normalized_noael']
        
        outliers = []
        for i, value in enumerate(comparison_results['normalized_noael_values']):
            if value is not None and (abs(value - mean) > 2 * std):
                outliers.append({
                    'study_id': comparison_results['study_ids'][i],
                    'normalized_noael': value,
                    'z_score': (value - mean) / std
                })
        
        comparison_results['outliers'] = outliers
    
    # Analyze consistency of adverse effects across studies
    endpoint_consistency = {}
    
    for result in normalized_results:
        for dose in result['adverse_effects']:
            for effect in result['adverse_effects'][dose]:
                endpoint = effect['endpoint']
                
                if endpoint not in endpoint_consistency:
                    endpoint_consistency[endpoint] = {
                        'studies_with_adverse_effect': 0,
                        'total_studies': len(normalized_results),
                        'percent_change_values': []
                    }
                
                if effect['is_adverse'] if 'is_adverse' in effect else True:
                    endpoint_consistency[endpoint]['studies_with_adverse_effect'] += 1
                    endpoint_consistency[endpoint]['percent_change_values'].append(effect['percent_change'])
    
    # Calculate consistency metrics
    for endpoint in endpoint_consistency:
        consistency = endpoint_consistency[endpoint]
        consistency['consistency_percentage'] = (consistency['studies_with_adverse_effect'] / 
                                               consistency['total_studies'] * 100)
        
        if consistency['percent_change_values']:
            consistency['mean_percent_change'] = np.mean(consistency['percent_change_values'])
            consistency['std_percent_change'] = np.std(consistency['percent_change_values'])
    
    comparison_results['endpoint_consistency'] = endpoint_consistency
    
    return comparison_results

# Function to prepare TxGemma input
def prepare_txgemma_input(normalized_results, comparison_results):
    """
    Prepare input for TxGemma model
    
    Parameters:
    -----------
    normalized_results : list
        List of normalized NOAEL results from multiple studies
    comparison_results : dict
        Results of NOAEL comparison across studies
    """
    print("Preparing TxGemma input...")
    
    # Create a summary of the studies and comparison results
    study_summary = {
        'studies': [],
        'comparison': {
            'mean_normalized_noael': comparison_results.get('mean_normalized_noael'),
            'median_normalized_noael': comparison_results.get('median_normalized_noael'),
            'cv_normalized_noael': comparison_results.get('cv_normalized_noael'),
            'outliers': comparison_results.get('outliers', [])
        },
        'endpoint_consistency': {}
    }
    
    # Add study information
    for result in normalized_results:
        study_info = {
            'study_id': result['study_id'],
            'noael': result['noael'],
            'normalized_noael': result['normalized_noael'],
            'normalization_factor': result['normalization_factor'],
            'study_design': result.get('study_design', {}),
            'key_adverse_effects': []
        }
        
        # Add key adverse effects
        for dose in result['adverse_effects']:
            for effect in result['adverse_effects'][dose]:
                if effect.get('is_adverse', True):
                    study_info['key_adverse_effects'].append({
                        'endpoint': effect['endpoint'],
                        'dose': dose,
                        'percent_change': round(effect['percent_change'], 1),
                        'p_value': round(effect['p_value'], 3) if 'p_value' in effect else None
                    })
        
        study_summary['studies'].append(study_info)
    
    # Add endpoint consistency information
    for endpoint, consistency in comparison_results.get('endpoint_consistency', {}).items():
        study_summary['endpoint_consistency'][endpoint] = {
            'consistency_percentage': round(consistency.get('consistency_percentage', 0), 1),
            'mean_percent_change': round(consistency.get('mean_percent_change', 0), 1),
            'std_percent_change': round(consistency.get('std_percent_change', 0), 1) if 'std_percent_change' in consistency else None
        }
    
    # Convert to JSON string
    study_summary_json = json.dumps(study_summary, indent=2)
    
    # Create prompt for TxGemma
    prompt = f"""
You are a toxicology expert analyzing data from multiple preclinical safety studies to assess the consistency of NOAEL determinations across studies.

Below is a summary of the studies and comparison results:

{study_summary_json}

Based on this data, please:
1. Assess the overall consistency of NOAEL determinations across the studies
2. Identify potential reasons for any inconsistencies observed
3. Determine which endpoints show the most consistent adverse effects across studies
4. Recommend a weight-of-evidence NOAEL based on the cross-study analysis
5. Suggest additional analyses or data that could improve the cross-study comparison

Your response should be structured as a JSON object with the following fields:
- consistency_assessment: assessment of overall NOAEL consistency (high, medium, or low)
- inconsistency_factors: list of factors contributing to inconsistencies
- most_consistent_endpoints: list of endpoints with consistent effects across studies
- weight_of_evidence_noael: recommended NOAEL based on all studies
- confidence: confidence level in the weight-of-evidence NOAEL (high, medium, or low)
- additional_analyses: suggestions for additional analyses
"""
    
    return prompt

# Function to simulate TxGemma response
def simulate_txgemma_response(prompt):
    """
    Simulate TxGemma response for demonstration purposes
    
    Parameters:
    -----------
    prompt : str
        Prompt for TxGemma model
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
    
    # Extract study information
    studies = study_summary.get('studies', [])
    comparison = study_summary.get('comparison', {})
    endpoint_consistency = study_summary.get('endpoint_consistency', {})
    
    # Assess consistency of NOAEL determinations
    if 'cv_normalized_noael' in comparison:
        cv = comparison['cv_normalized_noael']
        if cv < 20:
            consistency_assessment = "high"
        elif cv < 50:
            consistency_assessment = "medium"
        else:
            consistency_assessment = "low"
    else:
        consistency_assessment = "medium"
    
    # Identify factors contributing to inconsistencies
    inconsistency_factors = []
    
    # Check for outliers
    if 'outliers' in comparison and comparison['outliers']:
        inconsistency_factors.append("Presence of outlier studies with significantly different NOAEL values")
    
    # Check for differences in study design
    species_set = set()
    strain_set = set()
    duration_set = set()
    
    for study in studies:
        design = study.get('study_design', {})
        if 'species' in design:
            species_set.add(design['species'])
        if 'strain' in design:
            strain_set.add(design['strain'])
        if 'duration' in design:
            duration_set.add(design['duration'])
    
    if len(species_set) > 1:
        inconsistency_factors.append(f"Different species used across studies ({', '.join(species_set)})")
    
    if len(strain_set) > 1:
        inconsistency_factors.append(f"Different strains used across studies ({', '.join(strain_set)})")
    
    if len(duration_set) > 1:
        inconsistency_factors.append(f"Different study durations ({', '.join(duration_set)})")
    
    # If no specific factors identified, add general factors
    if not inconsistency_factors:
        inconsistency_factors = [
            "Biological variability between studies",
            "Differences in study conduct and laboratory procedures",
            "Variations in test article formulation or stability"
        ]
    
    # Identify most consistent endpoints
    most_consistent_endpoints = []
    
    for endpoint, consistency in endpoint_consistency.items():
        if consistency.get('consistency_percentage', 0) >= 75:
            most_consistent_endpoints.append({
                'endpoint': endpoint,
                'consistency_percentage': consistency.get('consistency_percentage'),
                'mean_percent_change': consistency.get('mean_percent_change')
            })
    
    # Sort by consistency percentage
    most_consistent_endpoints.sort(key=lambda x: x.get('consistency_percentage', 0), reverse=True)
    
    # Extract just the endpoint names for the response
    most_consistent_endpoint_names = [item['endpoint'] for item in most_consistent_endpoints]
    
    # Determine weight-of-evidence NOAEL
    if 'median_normalized_noael' in comparison:
        weight_of_evidence_noael = comparison['median_normalized_noael']
    elif 'mean_normalized_noael' in comparison:
        weight_of_evidence_noael = comparison['mean_normalized_noael']
    else:
        # If no summary statistics, use the most common NOAEL value
        noael_values = [study.get('normalized_noael') for study in studies if study.get('normalized_noael') is not None]
        if noael_values:
            weight_of_evidence_noael = max(set(noael_values), key=noael_values.count)
        else:
            weight_of_evidence_noael = None
    
    # Determine confidence in weight-of-evidence NOAEL
    if consistency_assessment == "high" and len(studies) >= 3:
        confidence = "high"
    elif consistency_assessment == "medium" or (consistency_assessment == "high" and len(studies) < 3):
        confidence = "medium"
    else:
        confidence = "low"
    
    # Suggest additional analyses
    additional_analyses = [
        "Conduct benchmark dose modeling across all studies to derive a more robust point of departure",
        "Perform meta-analysis of dose-response data for key endpoints",
        "Analyze historical control data to better contextualize findings",
        "Investigate potential sources of inter-study variability through sensitivity analysis",
        "Consider toxicokinetic data to normalize doses based on internal exposure rather than administered dose"
    ]
    
    # Create simulated response
    response = {
        "consistency_assessment": consistency_assessment,
        "inconsistency_factors": inconsistency_factors,
        "most_consistent_endpoints": most_consistent_endpoint_names,
        "weight_of_evidence_noael": weight_of_evidence_noael,
        "confidence": confidence,
        "additional_analyses": additional_analyses
    }
    
    return json.dumps(response, indent=2)

# Function to visualize comparison results
def visualize_comparison_results(normalized_results, comparison_results, txgemma_response):
    """
    Create visualizations of the comparison results
    
    Parameters:
    -----------
    normalized_results : list
        List of normalized NOAEL results from multiple studies
    comparison_results : dict
        Results of NOAEL comparison across studies
    txgemma_response : str
        TxGemma response as JSON string
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
    fig.suptitle('Cross-Study NOAEL Comparison and Consistency Analysis', fontsize=16)
    
    # Plot 1: NOAEL values across studies
    ax = axs[0, 0]
    
    study_ids = comparison_results['study_ids']
    noael_values = comparison_results['noael_values']
    normalized_noael_values = comparison_results['normalized_noael_values']
    
    # Create bar chart
    x = np.arange(len(study_ids))
    width = 0.35
    
    ax.bar(x - width/2, noael_values, width, label='Original NOAEL')
    ax.bar(x + width/2, normalized_noael_values, width, label='Normalized NOAEL')
    
    # Add mean/median line if available
    if 'mean_normalized_noael' in comparison_results:
        ax.axhline(y=comparison_results['mean_normalized_noael'], color='red', linestyle='--', 
                  label=f"Mean: {comparison_results['mean_normalized_noael']:.1f}")
    
    if 'median_normalized_noael' in comparison_results:
        ax.axhline(y=comparison_results['median_normalized_noael'], color='green', linestyle=':', 
                  label=f"Median: {comparison_results['median_normalized_noael']:.1f}")
    
    # Add weight-of-evidence NOAEL if available
    if txgemma_data and 'weight_of_evidence_noael' in txgemma_data:
        ax.axhline(y=txgemma_data['weight_of_evidence_noael'], color='blue', linestyle='-.',
                  label=f"WoE NOAEL: {txgemma_data['weight_of_evidence_noael']:.1f}")
    
    ax.set_xlabel('Study')
    ax.set_ylabel('NOAEL (mg/kg/day)')
    ax.set_title('NOAEL Values Across Studies')
    ax.set_xticks(x)
    ax.set_xticklabels(study_ids)
    ax.legend()
    
    # Plot 2: Endpoint consistency heatmap
    ax = axs[0, 1]
    
    # Create matrix of adverse effects by endpoint and study
    endpoints = list(comparison_results.get('endpoint_consistency', {}).keys())
    
    if endpoints:
        # Initialize matrix with zeros
        consistency_matrix = np.zeros((len(endpoints), len(study_ids)))
        
        # Fill matrix with percent change values
        for i, endpoint in enumerate(endpoints):
            for j, study_id in enumerate(study_ids):
                # Find adverse effects for this endpoint in this study
                for result in normalized_results:
                    if result['study_id'] == study_id:
                        for dose in result['adverse_effects']:
                            for effect in result['adverse_effects'][dose]:
                                if effect['endpoint'] == endpoint and effect.get('is_adverse', True):
                                    consistency_matrix[i, j] = effect['percent_change']
        
        # Create heatmap
        im = ax.imshow(consistency_matrix, cmap='RdBu_r', aspect='auto', vmin=-50, vmax=50)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Percent Change from Control')
        
        # Set labels
        ax.set_yticks(np.arange(len(endpoints)))
        ax.set_yticklabels(endpoints)
        ax.set_xticks(np.arange(len(study_ids)))
        ax.set_xticklabels(study_ids)
        
        ax.set_title('Adverse Effect Consistency Across Studies')
        ax.set_xlabel('Study')
        ax.set_ylabel('Endpoint')
    else:
        ax.text(0.5, 0.5, 'No consistent endpoints found', ha='center', va='center')
        ax.set_title('Endpoint Consistency')
    
    # Plot 3: Study design comparison
    ax = axs[1, 0]
    
    # Extract study design parameters
    design_params = ['species', 'strain', 'duration', 'route']
    design_values = {param: [] for param in design_params}
    
    for i, study_id in enumerate(study_ids):
        design = comparison_results['study_designs'][i]
        for param in design_params:
            design_values[param].append(design.get(param, 'Unknown'))
    
    # Create grouped bar chart for categorical parameters
    param_positions = np.arange(len(design_params))
    bar_width = 0.8 / len(study_ids)
    
    for i, study_id in enumerate(study_ids):
        values = []
        for param in design_params:
            # Convert to numeric if possible
            try:
                values.append(float(design_values[param][i]))
            except:
                values.append(0)  # Use 0 for non-numeric values
        
        # Only plot numeric values
        numeric_params = []
        numeric_values = []
        numeric_positions = []
        
        for j, param in enumerate(design_params):
            try:
                value = float(design_values[param][i])
                numeric_params.append(param)
                numeric_values.append(value)
                numeric_positions.append(param_positions[j])
            except:
                pass
        
        if numeric_values:
            ax.bar(np.array(numeric_positions) + i*bar_width - 0.4, numeric_values, bar_width, label=study_id)
    
    ax.set_xlabel('Study Design Parameter')
    ax.set_ylabel('Value')
    ax.set_title('Study Design Comparison')
    ax.set_xticks(param_positions)
    ax.set_xticklabels(design_params)
    ax.legend()
    
    # Plot 4: TxGemma assessment
    ax = axs[1, 1]
    ax.axis('off')  # Turn off axis
    
    # Create text summary from TxGemma response
    if txgemma_data:
        text = "TxGemma Assessment:\n\n"
        text += f"Consistency: {txgemma_data.get('consistency_assessment', 'Not specified').capitalize()}\n\n"
        text += f"Weight-of-Evidence NOAEL: {txgemma_data.get('weight_of_evidence_noael', 'Not determined')}\n"
        text += f"Confidence: {txgemma_data.get('confidence', 'Not specified').capitalize()}\n\n"
        
        text += "Inconsistency Factors:\n"
        for factor in txgemma_data.get('inconsistency_factors', []):
            text += f"- {factor}\n"
        
        text += "\nMost Consistent Endpoints:\n"
        for endpoint in txgemma_data.get('most_consistent_endpoints', []):
            text += f"- {endpoint}\n"
        
        ax.text(0, 1, text, va='top', ha='left', wrap=True, fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('/home/ubuntu/noael_project/demo_code/use_case3_results.png')
    print("Visualization saved to /home/ubuntu/noael_project/demo_code/use_case3_results.png")
    
    return '/home/ubuntu/noael_project/demo_code/use_case3_results.png'

# Main function
def main():
    """
    Main function to demonstrate TxGemma for cross-study NOAEL comparison
    """
    print("Starting cross-study NOAEL comparison demonstration using TxGemma...")
    
    # Step 1: Load SEND domains for multiple studies
    all_domains = {}
    for i, path in enumerate(SEND_DATA_PATHS):
        study_id = f"STUDY-{i+1:03d}"
        all_domains[study_id] = load_send_domains(path, study_id)
    
    # Step 2: Extract features for each study
    all_features = {}
    for study_id, domains in all_domains.items():
        all_features[study_id] = extract_features(domains, study_id)
    
    # Step 3: Analyze dose-response relationships for each study
    all_analysis_results = {}
    for study_id, features in all_features.items():
        all_analysis_results[study_id] = analyze_dose_response(features)
    
    # Step 4: Determine NOAEL for each study
    all_noael_results = []
    for study_id, features in all_features.items():
        noael_result = determine_noael(features, all_analysis_results[study_id])
        all_noael_results.append(noael_result)
        print(f"NOAEL for study {study_id}: {noael_result['noael']}")
    
    # Step 5: Normalize NOAEL values across studies
    normalized_results = normalize_noael_values(all_noael_results)
    print("\nNormalized NOAEL values:")
    for result in normalized_results:
        print(f"Study {result['study_id']}: {result['noael']} (normalized: {result['normalized_noael']})")
    
    # Step 6: Compare NOAEL determinations across studies
    comparison_results = compare_noael_determinations(normalized_results)
    
    # Step 7: Prepare input for TxGemma
    txgemma_prompt = prepare_txgemma_input(normalized_results, comparison_results)
    
    # Step 8: Get TxGemma response (simulated for demonstration)
    txgemma_response = simulate_txgemma_response(txgemma_prompt)
    print("\nTxGemma Response:")
    print(txgemma_response)
    
    # Step 9: Visualize comparison results
    visualization_path = visualize_comparison_results(normalized_results, comparison_results, txgemma_response)
    
    print("\nDemonstration completed successfully!")
    print(f"Results visualization saved to: {visualization_path}")
    
    return {
        'normalized_results': normalized_results,
        'comparison_results': comparison_results,
        'txgemma_response': json.loads(txgemma_response),
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    main()
