"""
Use Case 5: Uncertainty Quantification and Confidence Assessment for NOAEL Determination

This script demonstrates how to use TxGemma to quantify uncertainty in NOAEL determinations,
providing confidence scores and identifying key factors contributing to uncertainty.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
import scipy.optimize as optimize

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
    Create mock SEND data for demonstration purposes with varying levels of uncertainty
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
    
    # Create mock laboratory results (LB) domain with dose-dependent effects and varying uncertainty
    lb_data = []
    
    # Define endpoints with different uncertainty characteristics
    endpoints = {
        # High certainty endpoints - clear dose response, low variability
        'ALT': {'baseline': 45, 'effect': [1.0, 1.1, 1.5, 2.5], 'cv': 0.08},
        'AST': {'baseline': 80, 'effect': [1.0, 1.1, 1.4, 2.2], 'cv': 0.08},
        
        # Medium certainty endpoints - moderate dose response, medium variability
        'BUN': {'baseline': 15, 'effect': [1.0, 1.0, 1.2, 1.5], 'cv': 0.15},
        'CREAT': {'baseline': 0.6, 'effect': [1.0, 1.0, 1.1, 1.3], 'cv': 0.15},
        
        # Low certainty endpoints - subtle dose response, high variability
        'WBC': {'baseline': 10, 'effect': [1.0, 0.95, 0.9, 0.85], 'cv': 0.25},
        'PLT': {'baseline': 800, 'effect': [1.0, 0.98, 0.95, 0.9], 'cv': 0.25},
        
        # Conflicting endpoints - non-monotonic dose response
        'ALP': {'baseline': 120, 'effect': [1.0, 0.9, 1.1, 1.3], 'cv': 0.15},
        'CHOL': {'baseline': 80, 'effect': [1.0, 1.2, 1.1, 1.3], 'cv': 0.15},
        
        # No effect endpoints
        'RBC': {'baseline': 8, 'effect': [1.0, 1.0, 1.0, 1.0], 'cv': 0.08},
        'HGB': {'baseline': 15, 'effect': [1.0, 1.0, 1.0, 1.0], 'cv': 0.08}
    }
    
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        
        for endpoint, params in endpoints.items():
            # Add biological variability
            base_value = params['baseline']
            effect_multiplier = params['effect'][arm_idx[arm]]
            cv = params['cv']
            
            # Add random variation
            value = base_value * effect_multiplier * np.random.normal(1, cv)
            
            # Determine units based on endpoint
            if endpoint in ['ALT', 'AST', 'ALP']:
                units = 'U/L'
            elif endpoint in ['BUN', 'CREAT']:
                units = 'mg/dL'
            elif endpoint in ['WBC', 'PLT']:
                units = '10^9/L'
            elif endpoint in ['RBC']:
                units = '10^12/L'
            elif endpoint in ['HGB']:
                units = 'g/dL'
            elif endpoint in ['CHOL']:
                units = 'mg/dL'
            else:
                units = ''
            
            lb_data.append({
                'USUBJID': subj,
                'LBTEST': endpoint,
                'LBSTRESN': value,
                'LBSTRESU': units
            })
    
    domains['lb'] = pd.DataFrame(lb_data)
    
    # Create mock body weight (BW) domain with dose-dependent effects
    bw_data = []
    
    # Define body weight effects with medium certainty
    bw_effect = [1.0, 0.98, 0.95, 0.9]  # Effect multipliers by dose
    bw_cv = 0.08  # Coefficient of variation
    
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        sex = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'SEX'].values[0]
        
        # Base weight depends on sex
        base_weight = 250 if sex == 'M' else 200
        
        # Apply effect multiplier
        effect_multiplier = bw_effect[arm_idx[arm]]
        
        # Add random variation
        weight = base_weight * effect_multiplier * np.random.normal(1, bw_cv)
        
        bw_data.append({
            'USUBJID': subj,
            'BWSTRESN': weight,
            'BWSTRESU': 'g',
            'VISITDY': 28  # Terminal body weight
        })
    
    domains['bw'] = pd.DataFrame(bw_data)
    
    # Create mock histopathology findings (MI) domain with varying incidence
    mi_data = []
    
    # Define histopathology findings with different uncertainty characteristics
    findings = {
        # High certainty finding - clear dose response
        'LIVER': {
            'Hepatocellular hypertrophy': {'incidence': [0, 0.2, 0.6, 0.9], 'severity': [0, 1, 2, 3]}
        },
        # Medium certainty finding - moderate dose response
        'KIDNEY': {
            'Tubular degeneration': {'incidence': [0, 0, 0.3, 0.7], 'severity': [0, 0, 1, 2]}
        },
        # Low certainty finding - subtle dose response, low incidence
        'LUNG': {
            'Inflammation': {'incidence': [0.1, 0.1, 0.2, 0.3], 'severity': [1, 1, 1, 2]}
        },
        # Conflicting finding - non-monotonic dose response
        'SPLEEN': {
            'Extramedullary hematopoiesis': {'incidence': [0.2, 0.1, 0.3, 0.2], 'severity': [1, 1, 2, 1]}
        }
    }
    
    # Define severity grades
    severity_grades = {0: 'Normal', 1: 'Minimal', 2: 'Mild', 3: 'Moderate', 4: 'Marked'}
    
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        
        for organ, organ_findings in findings.items():
            for finding, params in organ_findings.items():
                # Determine if this subject has the finding based on incidence
                incidence = params['incidence'][arm_idx[arm]]
                if np.random.random() < incidence:
                    # Determine severity
                    severity = params['severity'][arm_idx[arm]]
                    
                    # Add finding
                    mi_data.append({
                        'USUBJID': subj,
                        'MISPEC': organ,
                        'MISTRESC': finding,
                        'MISEV': severity,
                        'MISEVTXT': severity_grades[severity]
                    })
    
    domains['mi'] = pd.DataFrame(mi_data)
    
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
                            'values': values.tolist(),
                            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
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
                        'values': values.tolist(),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    }
            
            features['body_weights'] = bw_features
            print(f"Extracted body weight data for {len(bw_features)} dose groups")
    
    # Extract histopathology findings
    if 'mi' in domains:
        mi_data = domains['mi']
        
        if 'MISPEC' in mi_data.columns and 'MISTRESC' in mi_data.columns:
            mi_features = {}
            
            # Group by organ and finding
            for organ in mi_data['MISPEC'].unique():
                organ_data = mi_data[mi_data['MISPEC'] == organ]
                
                organ_findings = {}
                for finding in organ_data['MISTRESC'].unique():
                    finding_data = organ_data[organ_data['MISTRESC'] == finding]
                    
                    # Calculate incidence and severity for each dose group
                    finding_stats = {}
                    for dose, subjects in features['dose_groups'].items():
                        dose_data = finding_data[finding_data['USUBJID'].isin(subjects)]
                        
                        # Calculate incidence
                        incidence = len(dose_data) / len(subjects) if subjects else 0
                        
                        # Calculate mean severity if severity data is available
                        mean_severity = dose_data['MISEV'].mean() if 'MISEV' in dose_data.columns and not dose_data.empty else 0
                        
                        finding_stats[dose] = {
                            'incidence': incidence,
                            'mean_severity': mean_severity,
                            'n_affected': len(dose_data),
                            'n_total': len(subjects)
                        }
                    
                    organ_findings[finding] = finding_stats
                
                mi_features[organ] = organ_findings
            
            features['histopathology'] = mi_features
            print(f"Extracted histopathology data for {len(mi_features)} organs")
    
    return features

# Function to analyze dose-response relationships with uncertainty
def analyze_dose_response_with_uncertainty(features):
    """
    Analyze dose-response relationships for each endpoint with uncertainty quantification
    """
    print("Analyzing dose-response relationships with uncertainty...")
    
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
            control_data = test_data.get(control_dose, {})
            
            if not control_data or 'values' not in control_data:
                continue
            
            # Analyze each dose group compared to control
            dose_effects = {}
            for dose in features['doses']:
                if dose == control_dose:
                    continue
                
                dose_data = test_data.get(dose, {})
                
                if not dose_data or 'values' not in dose_data:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_data['values'], dose_data['values'], equal_var=False)
                
                # Calculate percent change from control
                control_mean = control_data['mean']
                dose_mean = dose_data['mean']
                percent_change = ((dose_mean - control_mean) / control_mean) * 100
                
                # Calculate confidence interval for percent change
                # Using bootstrap method for demonstration
                n_bootstrap = 1000
                bootstrap_changes = []
                
                for _ in range(n_bootstrap):
                    # Resample with replacement
                    control_sample = np.random.choice(control_data['values'], size=len(control_data['values']), replace=True)
                    dose_sample = np.random.choice(dose_data['values'], size=len(dose_data['values']), replace=True)
                    
                    # Calculate percent change
                    control_mean_boot = np.mean(control_sample)
                    dose_mean_boot = np.mean(dose_sample)
                    percent_change_boot = ((dose_mean_boot - control_mean_boot) / control_mean_boot) * 100
                    
                    bootstrap_changes.append(percent_change_boot)
                
                # Calculate 95% confidence interval
                ci_lower = np.percentile(bootstrap_changes, 2.5)
                ci_upper = np.percentile(bootstrap_changes, 97.5)
                
                # Determine if effect is adverse with uncertainty
                # For demonstration, we'll consider >20% change with p<0.05 as adverse
                is_adverse = (abs(percent_change) > 20) and (p_value < 0.05)
                
                # Calculate uncertainty metrics
                uncertainty = {
                    'statistical': {
                        'p_value': p_value,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_upper - ci_lower
                    },
                    'biological': {
                        'control_cv': control_data.get('cv', 0),
                        'dose_cv': dose_data.get('cv', 0),
                        'sample_size': min(control_data.get('n', 0), dose_data.get('n', 0))
                    }
                }
                
                # Calculate overall uncertainty score (0-100, higher = more uncertain)
                # Factors: p-value, CI width, CV, sample size
                p_value_factor = min(p_value * 100, 50)  # 0-50 points based on p-value
                ci_width_factor = min(abs(ci_upper - ci_lower) / 20, 25)  # 0-25 points based on CI width
                cv_factor = min((control_data.get('cv', 0) + dose_data.get('cv', 0)) * 50, 15)  # 0-15 points based on CV
                sample_size_factor = max(0, 10 - min(control_data.get('n', 0), dose_data.get('n', 0)))  # 0-10 points based on sample size
                
                uncertainty_score = p_value_factor + ci_width_factor + cv_factor + sample_size_factor
                uncertainty_score = min(uncertainty_score, 100)  # Cap at 100
                
                # Determine confidence level based on uncertainty score
                if uncertainty_score < 25:
                    confidence = "high"
                elif uncertainty_score < 50:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                dose_effects[dose] = {
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'is_adverse': is_adverse,
                    'uncertainty': uncertainty,
                    'uncertainty_score': uncertainty_score,
                    'confidence': confidence
                }
            
            # Analyze dose-response trend
            if len(dose_effects) >= 2:
                doses = sorted([float(d) for d in dose_effects.keys()])
                changes = [dose_effects[d]['percent_change'] for d in doses]
                
                # Check for monotonicity
                is_monotonic = all(changes[i] <= changes[i+1] for i in range(len(changes)-1)) or \
                              all(changes[i] >= changes[i+1] for i in range(len(changes)-1))
                
                # Fit dose-response model if enough data points
                if len(doses) >= 3:
                    try:
                        # Simple linear regression for demonstration
                        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, changes)
                        
                        dose_response = {
                            'model': 'linear',
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'is_monotonic': is_monotonic
                        }
                    except:
                        dose_response = {
                            'model': 'none',
                            'is_monotonic': is_monotonic
                        }
                else:
                    dose_response = {
                        'model': 'none',
                        'is_monotonic': is_monotonic
                    }
                
                # Add dose-response information to results
                lb_results[test] = {
                    'dose_effects': dose_effects,
                    'dose_response': dose_response
                }
            else:
                lb_results[test] = {
                    'dose_effects': dose_effects,
                    'dose_response': {'model': 'none', 'is_monotonic': True}
                }
        
        results['laboratory_tests'] = lb_results
    
    # Analyze body weight data
    if 'body_weights' in features:
        bw_results = {}
        
        # Get control group data
        control_dose = min(features['doses'])
        control_data = features['body_weights'].get(control_dose, {})
        
        if control_data and 'values' in control_data:
            # Analyze each dose group compared to control
            dose_effects = {}
            for dose in features['doses']:
                if dose == control_dose:
                    continue
                
                dose_data = features['body_weights'].get(dose, {})
                
                if not dose_data or 'values' not in dose_data:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_data['values'], dose_data['values'], equal_var=False)
                
                # Calculate percent change from control
                control_mean = control_data['mean']
                dose_mean = dose_data['mean']
                percent_change = ((dose_mean - control_mean) / control_mean) * 100
                
                # Calculate confidence interval for percent change
                # Using bootstrap method for demonstration
                n_bootstrap = 1000
                bootstrap_changes = []
                
                for _ in range(n_bootstrap):
                    # Resample with replacement
                    control_sample = np.random.choice(control_data['values'], size=len(control_data['values']), replace=True)
                    dose_sample = np.random.choice(dose_data['values'], size=len(dose_data['values']), replace=True)
                    
                    # Calculate percent change
                    control_mean_boot = np.mean(control_sample)
                    dose_mean_boot = np.mean(dose_sample)
                    percent_change_boot = ((dose_mean_boot - control_mean_boot) / control_mean_boot) * 100
                    
                    bootstrap_changes.append(percent_change_boot)
                
                # Calculate 95% confidence interval
                ci_lower = np.percentile(bootstrap_changes, 2.5)
                ci_upper = np.percentile(bootstrap_changes, 97.5)
                
                # Determine if effect is adverse with uncertainty
                # For body weight, a decrease >10% with p<0.05 is typically considered adverse
                is_adverse = (percent_change < -10) and (p_value < 0.05)
                
                # Calculate uncertainty metrics
                uncertainty = {
                    'statistical': {
                        'p_value': p_value,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_upper - ci_lower
                    },
                    'biological': {
                        'control_cv': control_data.get('cv', 0),
                        'dose_cv': dose_data.get('cv', 0),
                        'sample_size': min(control_data.get('n', 0), dose_data.get('n', 0))
                    }
                }
                
                # Calculate overall uncertainty score (0-100, higher = more uncertain)
                p_value_factor = min(p_value * 100, 50)
                ci_width_factor = min(abs(ci_upper - ci_lower) / 10, 25)  # Body weight typically has narrower CIs
                cv_factor = min((control_data.get('cv', 0) + dose_data.get('cv', 0)) * 50, 15)
                sample_size_factor = max(0, 10 - min(control_data.get('n', 0), dose_data.get('n', 0)))
                
                uncertainty_score = p_value_factor + ci_width_factor + cv_factor + sample_size_factor
                uncertainty_score = min(uncertainty_score, 100)
                
                # Determine confidence level based on uncertainty score
                if uncertainty_score < 25:
                    confidence = "high"
                elif uncertainty_score < 50:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                dose_effects[dose] = {
                    'p_value': p_value,
                    'percent_change': percent_change,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'is_adverse': is_adverse,
                    'uncertainty': uncertainty,
                    'uncertainty_score': uncertainty_score,
                    'confidence': confidence
                }
            
            # Analyze dose-response trend
            if len(dose_effects) >= 2:
                doses = sorted([float(d) for d in dose_effects.keys()])
                changes = [dose_effects[d]['percent_change'] for d in doses]
                
                # Check for monotonicity
                is_monotonic = all(changes[i] <= changes[i+1] for i in range(len(changes)-1)) or \
                              all(changes[i] >= changes[i+1] for i in range(len(changes)-1))
                
                # Fit dose-response model if enough data points
                if len(doses) >= 3:
                    try:
                        # Simple linear regression for demonstration
                        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, changes)
                        
                        dose_response = {
                            'model': 'linear',
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'is_monotonic': is_monotonic
                        }
                    except:
                        dose_response = {
                            'model': 'none',
                            'is_monotonic': is_monotonic
                        }
                else:
                    dose_response = {
                        'model': 'none',
                        'is_monotonic': is_monotonic
                    }
                
                # Add dose-response information to results
                bw_results = {
                    'dose_effects': dose_effects,
                    'dose_response': dose_response
                }
            else:
                bw_results = {
                    'dose_effects': dose_effects,
                    'dose_response': {'model': 'none', 'is_monotonic': True}
                }
        
        results['body_weights'] = bw_results
    
    # Analyze histopathology findings
    if 'histopathology' in features:
        mi_results = {}
        
        for organ, organ_findings in features['histopathology'].items():
            organ_results = {}
            
            for finding, finding_data in organ_findings.items():
                # Analyze dose-response for incidence
                doses = sorted([float(d) for d in finding_data.keys()])
                incidences = [finding_data[d]['incidence'] for d in doses]
                
                # Check for monotonicity
                is_monotonic = all(incidences[i] <= incidences[i+1] for i in range(len(incidences)-1)) or \
                              all(incidences[i] >= incidences[i+1] for i in range(len(incidences)-1))
                
                # Calculate uncertainty for each dose
                dose_effects = {}
                for dose in doses:
                    # Skip control group
                    if dose == min(doses):
                        continue
                    
                    # Get data for this dose
                    dose_data = finding_data[dose]
                    
                    # Calculate confidence interval for incidence using Wilson score interval
                    # This is appropriate for small sample sizes and proportions near 0 or 1
                    n = dose_data['n_total']
                    p = dose_data['incidence']
                    z = 1.96  # 95% confidence
                    
                    denominator = 1 + z**2/n
                    center = (p + z**2/(2*n)) / denominator
                    halfwidth = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
                    
                    ci_lower = max(0, center - halfwidth)
                    ci_upper = min(1, center + halfwidth)
                    
                    # Determine if effect is adverse
                    # For histopathology, any statistically significant increase in incidence is adverse
                    # Using Fisher's exact test for small sample sizes
                    control_data = finding_data[min(doses)]
                    
                    # Create contingency table
                    table = np.array([
                        [dose_data['n_affected'], dose_data['n_total'] - dose_data['n_affected']],
                        [control_data['n_affected'], control_data['n_total'] - control_data['n_affected']]
                    ])
                    
                    # Perform Fisher's exact test
                    odds_ratio, p_value = stats.fisher_exact(table)
                    
                    is_adverse = (dose_data['incidence'] > control_data['incidence']) and (p_value < 0.05)
                    
                    # Calculate uncertainty metrics
                    uncertainty = {
                        'statistical': {
                            'p_value': p_value,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'ci_width': ci_upper - ci_lower
                        },
                        'biological': {
                            'sample_size': dose_data['n_total'],
                            'background_incidence': control_data['incidence']
                        }
                    }
                    
                    # Calculate overall uncertainty score (0-100, higher = more uncertain)
                    p_value_factor = min(p_value * 100, 50)
                    ci_width_factor = min((ci_upper - ci_lower) * 100, 25)
                    sample_size_factor = max(0, 15 - dose_data['n_total'] / 2)  # 0-15 points based on sample size
                    background_factor = min(control_data['incidence'] * 50, 10)  # Higher background incidence increases uncertainty
                    
                    uncertainty_score = p_value_factor + ci_width_factor + sample_size_factor + background_factor
                    uncertainty_score = min(uncertainty_score, 100)
                    
                    # Determine confidence level based on uncertainty score
                    if uncertainty_score < 25:
                        confidence = "high"
                    elif uncertainty_score < 50:
                        confidence = "medium"
                    else:
                        confidence = "low"
                    
                    dose_effects[dose] = {
                        'p_value': p_value,
                        'incidence': dose_data['incidence'],
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'is_adverse': is_adverse,
                        'uncertainty': uncertainty,
                        'uncertainty_score': uncertainty_score,
                        'confidence': confidence
                    }
                
                # Fit dose-response model if enough data points
                if len(doses) >= 3:
                    try:
                        # Simple linear regression for demonstration
                        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, incidences)
                        
                        dose_response = {
                            'model': 'linear',
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'is_monotonic': is_monotonic
                        }
                    except:
                        dose_response = {
                            'model': 'none',
                            'is_monotonic': is_monotonic
                        }
                else:
                    dose_response = {
                        'model': 'none',
                        'is_monotonic': is_monotonic
                    }
                
                organ_results[finding] = {
                    'dose_effects': dose_effects,
                    'dose_response': dose_response
                }
            
            mi_results[organ] = organ_results
        
        results['histopathology'] = mi_results
    
    return results

# Function to determine NOAEL with uncertainty quantification
def determine_noael_with_uncertainty(features, analysis_results):
    """
    Determine NOAEL with uncertainty quantification
    """
    print("Determining NOAEL with uncertainty quantification...")
    
    # Get sorted dose levels
    doses = sorted(features['doses'])
    
    # Track adverse effects by dose
    adverse_effects = {dose: [] for dose in doses}
    
    # Track uncertainty factors by dose
    uncertainty_factors = {dose: [] for dose in doses}
    
    # Check laboratory test results
    if 'laboratory_tests' in analysis_results:
        for test, test_results in analysis_results['laboratory_tests'].items():
            for dose, result in test_results.get('dose_effects', {}).items():
                if result['is_adverse']:
                    adverse_effects[dose].append({
                        'endpoint': test,
                        'percent_change': result['percent_change'],
                        'p_value': result['p_value'],
                        'confidence': result['confidence']
                    })
                
                # Add uncertainty factors
                if result['uncertainty_score'] > 50:  # Only add high uncertainty factors
                    uncertainty_factors[dose].append({
                        'endpoint': test,
                        'factor': 'High variability',
                        'uncertainty_score': result['uncertainty_score'],
                        'details': f"CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}], p={result['p_value']:.3f}"
                    })
                
            # Check for non-monotonic dose response
            if not test_results.get('dose_response', {}).get('is_monotonic', True):
                for dose in doses[1:]:  # Skip control
                    uncertainty_factors[dose].append({
                        'endpoint': test,
                        'factor': 'Non-monotonic dose response',
                        'uncertainty_score': 60,
                        'details': f"Non-monotonic dose response for {test}"
                    })
    
    # Check body weight results
    if 'body_weights' in analysis_results:
        for dose, result in analysis_results['body_weights'].get('dose_effects', {}).items():
            if result['is_adverse']:
                adverse_effects[dose].append({
                    'endpoint': 'Body Weight',
                    'percent_change': result['percent_change'],
                    'p_value': result['p_value'],
                    'confidence': result['confidence']
                })
            
            # Add uncertainty factors
            if result['uncertainty_score'] > 50:
                uncertainty_factors[dose].append({
                    'endpoint': 'Body Weight',
                    'factor': 'High variability',
                    'uncertainty_score': result['uncertainty_score'],
                    'details': f"CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}], p={result['p_value']:.3f}"
                })
    
    # Check histopathology results
    if 'histopathology' in analysis_results:
        for organ, organ_results in analysis_results['histopathology'].items():
            for finding, finding_results in organ_results.items():
                for dose, result in finding_results.get('dose_effects', {}).items():
                    if result['is_adverse']:
                        adverse_effects[dose].append({
                            'endpoint': f"{organ} - {finding}",
                            'incidence': result['incidence'] * 100,  # Convert to percentage
                            'p_value': result['p_value'],
                            'confidence': result['confidence']
                        })
                    
                    # Add uncertainty factors
                    if result['uncertainty_score'] > 50:
                        uncertainty_factors[dose].append({
                            'endpoint': f"{organ} - {finding}",
                            'factor': 'Low incidence or small sample size',
                            'uncertainty_score': result['uncertainty_score'],
                            'details': f"Incidence: {result['incidence']*100:.1f}%, CI: [{result['ci_lower']*100:.1f}, {result['ci_upper']*100:.1f}], p={result['p_value']:.3f}"
                        })
                
                # Check for non-monotonic dose response
                if not finding_results.get('dose_response', {}).get('is_monotonic', True):
                    for dose in doses[1:]:  # Skip control
                        uncertainty_factors[dose].append({
                            'endpoint': f"{organ} - {finding}",
                            'factor': 'Non-monotonic dose response',
                            'uncertainty_score': 60,
                            'details': f"Non-monotonic dose response for {organ} - {finding}"
                        })
    
    # Determine NOAEL as highest dose with no adverse effects
    noael = None
    for dose in doses:
        if not adverse_effects[dose]:
            noael = dose
        else:
            break
    
    # Calculate confidence in NOAEL determination
    noael_confidence = "high"  # Default confidence
    confidence_factors = []
    
    # Check if there are adverse effects at the next dose level
    if noael is not None and noael < max(doses):
        next_dose_idx = doses.index(noael) + 1
        if next_dose_idx < len(doses):
            next_dose = doses[next_dose_idx]
            
            # Check confidence of adverse effects at next dose
            effect_confidences = [effect['confidence'] for effect in adverse_effects[next_dose]]
            
            if 'low' in effect_confidences:
                noael_confidence = "medium"
                confidence_factors.append("Low confidence in adverse effects at next dose level")
            
            # Check if there are uncertainty factors at NOAEL
            if uncertainty_factors[noael]:
                noael_confidence = "medium"
                confidence_factors.append("Uncertainty factors present at NOAEL")
            
            # Check if there are uncertainty factors at next dose
            if uncertainty_factors[next_dose]:
                noael_confidence = "medium"
                confidence_factors.append("Uncertainty factors present at next dose level")
            
            # Check dose spacing
            dose_ratio = next_dose / noael if noael > 0 else float('inf')
            if dose_ratio > 5:
                noael_confidence = "medium"
                confidence_factors.append(f"Large dose spacing (ratio: {dose_ratio:.1f})")
            
            # Check number of adverse effects at next dose
            if len(adverse_effects[next_dose]) == 1:
                noael_confidence = "medium"
                confidence_factors.append("Only one adverse effect at next dose level")
    else:
        # If NOAEL is the highest dose or no NOAEL could be determined
        noael_confidence = "low"
        confidence_factors.append("NOAEL is at highest tested dose or could not be determined")
    
    # Prepare results
    noael_results = {
        'noael': noael,
        'confidence': noael_confidence,
        'confidence_factors': confidence_factors,
        'adverse_effects': adverse_effects,
        'uncertainty_factors': uncertainty_factors,
        'doses': doses
    }
    
    return noael_results

# Function to perform benchmark dose modeling
def perform_benchmark_dose_modeling(features, analysis_results):
    """
    Perform benchmark dose modeling for continuous endpoints
    """
    print("Performing benchmark dose modeling...")
    
    bmd_results = {}
    
    # Define benchmark response (BMR) levels
    bmr_levels = {
        'default': 0.1,  # 10% change from control
        'body_weight': 0.05,  # 5% change for body weight
        'enzyme': 0.2,  # 20% change for enzymes (ALT, AST, etc.)
        'histopathology': 0.1  # 10% extra risk for histopathology
    }
    
    # Analyze laboratory test results
    if 'laboratory_tests' in analysis_results:
        lb_bmd = {}
        
        for test, test_results in analysis_results['laboratory_tests'].items():
            # Skip if no dose-response data
            if 'dose_response' not in test_results or test_results['dose_response']['model'] == 'none':
                continue
            
            # Get dose-response data
            doses = sorted([float(d) for d in test_results['dose_effects'].keys()])
            control_dose = min(features['doses'])
            
            # Skip if control dose is not in the data
            if control_dose not in features['laboratory_tests'][test]:
                continue
            
            control_mean = features['laboratory_tests'][test][control_dose]['mean']
            
            # Get response values (percent change from control)
            responses = [test_results['dose_effects'][d]['percent_change'] for d in doses if d != control_dose]
            doses_no_control = [d for d in doses if d != control_dose]
            
            # Skip if not enough data points
            if len(doses_no_control) < 2:
                continue
            
            # Determine BMR based on endpoint
            if test in ['ALT', 'AST', 'ALP']:
                bmr = bmr_levels['enzyme']
            else:
                bmr = bmr_levels['default']
            
            # Convert BMR to percent change
            bmr_percent = bmr * 100
            
            try:
                # Fit Hill model for demonstration
                # Hill model: y = control + (max - control) * x^n / (k^n + x^n)
                def hill_model(x, max_effect, k, n):
                    return max_effect * x**n / (k**n + x**n)
                
                # Initial parameter estimates
                max_effect_init = max(responses) if max(responses) > 0 else min(responses)
                k_init = np.median(doses_no_control)
                n_init = 1.0
                
                # Fit model
                params, _ = optimize.curve_fit(
                    hill_model, doses_no_control, responses,
                    p0=[max_effect_init, k_init, n_init],
                    bounds=([max_effect_init*0.1, 0.1, 0.1], [max_effect_init*10, max(doses_no_control)*10, 10]),
                    maxfev=10000
                )
                
                max_effect, k, n = params
                
                # Calculate BMD
                # Solve for dose where response = BMR
                if abs(max_effect) > abs(bmr_percent):
                    bmd = k * ((bmr_percent / max_effect) / (1 - bmr_percent / max_effect))**(1/n)
                else:
                    bmd = None
                
                # Calculate BMDL (lower confidence limit)
                # For demonstration, use a simple approximation
                # In practice, would use profile likelihood or bootstrap methods
                bmdl = bmd * 0.8 if bmd is not None else None
                
                # Calculate model fit statistics
                predicted = [hill_model(d, max_effect, k, n) for d in doses_no_control]
                residuals = np.array(responses) - np.array(predicted)
                ss_residual = np.sum(residuals**2)
                ss_total = np.sum((np.array(responses) - np.mean(responses))**2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Store results
                lb_bmd[test] = {
                    'model': 'Hill',
                    'parameters': {
                        'max_effect': max_effect,
                        'k': k,
                        'n': n
                    },
                    'bmr': bmr,
                    'bmd': bmd,
                    'bmdl': bmdl,
                    'model_fit': {
                        'r_squared': r_squared,
                        'residual_sum_squares': ss_residual
                    },
                    'doses': doses_no_control,
                    'responses': responses,
                    'predicted': predicted
                }
            except Exception as e:
                print(f"Error fitting model for {test}: {e}")
        
        bmd_results['laboratory_tests'] = lb_bmd
    
    # Analyze body weight
    if 'body_weights' in analysis_results:
        bw_results = analysis_results['body_weights']
        
        # Skip if no dose-response data
        if 'dose_response' in bw_results and bw_results['dose_response']['model'] != 'none':
            # Get dose-response data
            doses = sorted([float(d) for d in bw_results['dose_effects'].keys()])
            control_dose = min(features['doses'])
            
            # Skip if control dose is not in the data
            if control_dose not in features['body_weights']:
                return bmd_results
            
            control_mean = features['body_weights'][control_dose]['mean']
            
            # Get response values (percent change from control)
            responses = [bw_results['dose_effects'][d]['percent_change'] for d in doses if d != control_dose]
            doses_no_control = [d for d in doses if d != control_dose]
            
            # Skip if not enough data points
            if len(doses_no_control) < 2:
                return bmd_results
            
            # Use body weight specific BMR
            bmr = bmr_levels['body_weight']
            bmr_percent = -bmr * 100  # Negative because decrease in body weight is adverse
            
            try:
                # Fit exponential model for demonstration
                # Exponential model: y = a * exp(b * x)
                def exp_model(x, a, b):
                    return a * np.exp(b * x)
                
                # Initial parameter estimates
                a_init = responses[0]
                b_init = -0.001
                
                # Fit model
                params, _ = optimize.curve_fit(
                    exp_model, doses_no_control, responses,
                    p0=[a_init, b_init],
                    maxfev=10000
                )
                
                a, b = params
                
                # Calculate BMD
                # Solve for dose where response = BMR
                if b < 0:  # Only if slope is negative (decreasing body weight)
                    bmd = np.log(bmr_percent / a) / b
                else:
                    bmd = None
                
                # Calculate BMDL (lower confidence limit)
                bmdl = bmd * 0.8 if bmd is not None else None
                
                # Calculate model fit statistics
                predicted = [exp_model(d, a, b) for d in doses_no_control]
                residuals = np.array(responses) - np.array(predicted)
                ss_residual = np.sum(residuals**2)
                ss_total = np.sum((np.array(responses) - np.mean(responses))**2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Store results
                bmd_results['body_weights'] = {
                    'model': 'Exponential',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'bmr': bmr,
                    'bmd': bmd,
                    'bmdl': bmdl,
                    'model_fit': {
                        'r_squared': r_squared,
                        'residual_sum_squares': ss_residual
                    },
                    'doses': doses_no_control,
                    'responses': responses,
                    'predicted': predicted
                }
            except Exception as e:
                print(f"Error fitting model for body weight: {e}")
    
    # Analyze histopathology findings
    if 'histopathology' in analysis_results:
        histo_bmd = {}
        
        for organ, organ_results in analysis_results['histopathology'].items():
            organ_bmd = {}
            
            for finding, finding_results in organ_results.items():
                # Skip if no dose-response data
                if 'dose_response' not in finding_results or finding_results['dose_response']['model'] == 'none':
                    continue
                
                # Get dose-response data
                doses = sorted([float(d) for d in finding_results['dose_effects'].keys()])
                control_dose = min(features['doses'])
                
                # Get incidence values
                incidences = []
                doses_for_model = []
                
                for dose in sorted(features['doses']):
                    if dose == control_dose:
                        # Get background incidence
                        if dose in features['histopathology'][organ][finding]:
                            background = features['histopathology'][organ][finding][dose]['incidence']
                        else:
                            background = 0
                    elif dose in finding_results['dose_effects']:
                        incidences.append(finding_results['dose_effects'][dose]['incidence'])
                        doses_for_model.append(dose)
                
                # Skip if not enough data points
                if len(doses_for_model) < 2:
                    continue
                
                # Use histopathology specific BMR
                bmr = bmr_levels['histopathology']
                
                try:
                    # Fit logistic model for demonstration
                    # Logistic model: p = background + (1 - background) / (1 + exp(-a - b*log(x)))
                    def logistic_model(x, a, b):
                        return background + (1 - background) / (1 + np.exp(-a - b*np.log(x)))
                    
                    # Initial parameter estimates
                    a_init = -5
                    b_init = 1
                    
                    # Fit model
                    params, _ = optimize.curve_fit(
                        logistic_model, doses_for_model, incidences,
                        p0=[a_init, b_init],
                        bounds=([-10, 0], [0, 10]),
                        maxfev=10000
                    )
                    
                    a, b = params
                    
                    # Calculate BMD
                    # Solve for dose where extra risk = BMR
                    # Extra risk = (p - background) / (1 - background)
                    target = background + bmr * (1 - background)
                    
                    # Solve numerically
                    def objective(x):
                        return logistic_model(x, a, b) - target
                    
                    try:
                        bmd = optimize.brentq(objective, 0.1, max(doses_for_model) * 2)
                    except:
                        bmd = None
                    
                    # Calculate BMDL (lower confidence limit)
                    bmdl = bmd * 0.7 if bmd is not None else None
                    
                    # Calculate model fit statistics
                    predicted = [logistic_model(d, a, b) for d in doses_for_model]
                    residuals = np.array(incidences) - np.array(predicted)
                    ss_residual = np.sum(residuals**2)
                    ss_total = np.sum((np.array(incidences) - np.mean(incidences))**2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Store results
                    organ_bmd[finding] = {
                        'model': 'Logistic',
                        'parameters': {
                            'a': a,
                            'b': b,
                            'background': background
                        },
                        'bmr': bmr,
                        'bmd': bmd,
                        'bmdl': bmdl,
                        'model_fit': {
                            'r_squared': r_squared,
                            'residual_sum_squares': ss_residual
                        },
                        'doses': doses_for_model,
                        'incidences': incidences,
                        'predicted': predicted
                    }
                except Exception as e:
                    print(f"Error fitting model for {organ} - {finding}: {e}")
            
            if organ_bmd:
                histo_bmd[organ] = organ_bmd
        
        if histo_bmd:
            bmd_results['histopathology'] = histo_bmd
    
    return bmd_results

# Function to perform probabilistic NOAEL determination
def perform_probabilistic_noael(features, analysis_results, bmd_results):
    """
    Perform probabilistic NOAEL determination using Monte Carlo simulation
    """
    print("Performing probabilistic NOAEL determination...")
    
    # Get sorted dose levels
    doses = sorted(features['doses'])
    
    # Define number of simulations
    n_simulations = 1000
    
    # Initialize probability distributions for NOAEL
    noael_probabilities = {dose: 0 for dose in doses}
    
    # Perform Monte Carlo simulation
    for _ in range(n_simulations):
        # Track adverse effects by dose for this simulation
        sim_adverse_effects = {dose: [] for dose in doses}
        
        # Simulate laboratory test results
        if 'laboratory_tests' in analysis_results:
            for test, test_results in analysis_results['laboratory_tests'].items():
                for dose, result in test_results.get('dose_effects', {}).items():
                    # Skip control dose
                    if dose == min(doses):
                        continue
                    
                    # Get confidence interval
                    ci_lower = result.get('ci_lower', result['percent_change'] * 0.8)
                    ci_upper = result.get('ci_upper', result['percent_change'] * 1.2)
                    
                    # Sample from distribution
                    sampled_change = np.random.uniform(ci_lower, ci_upper)
                    
                    # Sample p-value from beta distribution
                    # This simulates uncertainty in statistical significance
                    alpha = 1 + (1 - result['p_value']) * 10
                    beta = 1 + result['p_value'] * 10
                    sampled_p = 1 - np.random.beta(alpha, beta)
                    
                    # Determine if effect is adverse in this simulation
                    is_adverse = (abs(sampled_change) > 20) and (sampled_p < 0.05)
                    
                    if is_adverse:
                        sim_adverse_effects[dose].append(test)
        
        # Simulate body weight results
        if 'body_weights' in analysis_results:
            for dose, result in analysis_results['body_weights'].get('dose_effects', {}).items():
                # Skip control dose
                if dose == min(doses):
                    continue
                
                # Get confidence interval
                ci_lower = result.get('ci_lower', result['percent_change'] * 0.8)
                ci_upper = result.get('ci_upper', result['percent_change'] * 1.2)
                
                # Sample from distribution
                sampled_change = np.random.uniform(ci_lower, ci_upper)
                
                # Sample p-value from beta distribution
                alpha = 1 + (1 - result['p_value']) * 10
                beta = 1 + result['p_value'] * 10
                sampled_p = 1 - np.random.beta(alpha, beta)
                
                # Determine if effect is adverse in this simulation
                is_adverse = (sampled_change < -10) and (sampled_p < 0.05)
                
                if is_adverse:
                    sim_adverse_effects[dose].append('Body Weight')
        
        # Simulate histopathology results
        if 'histopathology' in analysis_results:
            for organ, organ_results in analysis_results['histopathology'].items():
                for finding, finding_results in organ_results.items():
                    for dose, result in finding_results.get('dose_effects', {}).items():
                        # Skip control dose
                        if dose == min(doses):
                            continue
                        
                        # Get confidence interval
                        ci_lower = result.get('ci_lower', max(0, result['incidence'] * 0.5))
                        ci_upper = result.get('ci_upper', min(1, result['incidence'] * 1.5))
                        
                        # Sample from distribution
                        sampled_incidence = np.random.uniform(ci_lower, ci_upper)
                        
                        # Sample p-value from beta distribution
                        alpha = 1 + (1 - result['p_value']) * 10
                        beta = 1 + result['p_value'] * 10
                        sampled_p = 1 - np.random.beta(alpha, beta)
                        
                        # Determine if effect is adverse in this simulation
                        # For histopathology, any statistically significant increase is adverse
                        control_incidence = 0
                        for d in finding_results.get('dose_effects', {}):
                            if d == min(doses):
                                control_incidence = finding_results['dose_effects'][d]['incidence']
                                break
                        
                        is_adverse = (sampled_incidence > control_incidence) and (sampled_p < 0.05)
                        
                        if is_adverse:
                            sim_adverse_effects[dose].append(f"{organ} - {finding}")
        
        # Determine NOAEL for this simulation
        sim_noael = None
        for dose in doses:
            if not sim_adverse_effects[dose]:
                sim_noael = dose
            else:
                break
        
        # Update probability distribution
        if sim_noael is not None:
            noael_probabilities[sim_noael] += 1
    
    # Normalize probabilities
    total = sum(noael_probabilities.values())
    if total > 0:
        for dose in noael_probabilities:
            noael_probabilities[dose] /= total
    
    # Calculate cumulative probabilities
    cumulative_probabilities = {}
    cumulative = 0
    for dose in sorted(doses):
        cumulative += noael_probabilities[dose]
        cumulative_probabilities[dose] = cumulative
    
    # Determine most likely NOAEL
    most_likely_noael = max(noael_probabilities.items(), key=lambda x: x[1])[0]
    
    # Calculate credible intervals
    lower_bound = None
    upper_bound = None
    
    for dose in sorted(doses):
        if cumulative_probabilities[dose] >= 0.025 and lower_bound is None:
            lower_bound = dose
        if cumulative_probabilities[dose] >= 0.975 and upper_bound is None:
            upper_bound = dose
    
    # If upper bound is still None, use highest dose
    if upper_bound is None:
        upper_bound = max(doses)
    
    # Prepare results
    probabilistic_results = {
        'most_likely_noael': most_likely_noael,
        'credible_interval': [lower_bound, upper_bound],
        'noael_probabilities': noael_probabilities,
        'cumulative_probabilities': cumulative_probabilities
    }
    
    # Integrate BMD results if available
    if bmd_results:
        # Collect all BMDLs
        bmdls = []
        
        if 'laboratory_tests' in bmd_results:
            for test, result in bmd_results['laboratory_tests'].items():
                if result.get('bmdl') is not None:
                    bmdls.append({
                        'endpoint': test,
                        'bmdl': result['bmdl'],
                        'bmr': result['bmr']
                    })
        
        if 'body_weights' in bmd_results and bmd_results['body_weights'].get('bmdl') is not None:
            bmdls.append({
                'endpoint': 'Body Weight',
                'bmdl': bmd_results['body_weights']['bmdl'],
                'bmr': bmd_results['body_weights']['bmr']
            })
        
        if 'histopathology' in bmd_results:
            for organ, organ_results in bmd_results['histopathology'].items():
                for finding, result in organ_results.items():
                    if result.get('bmdl') is not None:
                        bmdls.append({
                            'endpoint': f"{organ} - {finding}",
                            'bmdl': result['bmdl'],
                            'bmr': result['bmr']
                        })
        
        # Determine point of departure (POD)
        if bmdls:
            # Sort by BMDL
            bmdls.sort(key=lambda x: x['bmdl'])
            
            # Lowest BMDL is the POD
            pod = bmdls[0]
            
            probabilistic_results['point_of_departure'] = pod
            
            # Compare POD with NOAEL
            if pod['bmdl'] < most_likely_noael:
                probabilistic_results['pod_vs_noael'] = "POD is lower than NOAEL, suggesting NOAEL may not be protective enough"
            else:
                probabilistic_results['pod_vs_noael'] = "POD is higher than or equal to NOAEL, supporting the NOAEL determination"
    
    return probabilistic_results

# Function to prepare TxGemma input
def prepare_txgemma_input(features, analysis_results, noael_results, bmd_results, probabilistic_results):
    """
    Prepare input for TxGemma model
    """
    print("Preparing TxGemma input...")
    
    # Create a summary of the uncertainty analysis
    uncertainty_summary = {
        'study_design': {
            'species': 'Rat',  # Assuming rat for demonstration
            'dose_groups': [str(dose) for dose in features['doses']]
        },
        'noael_determination': {
            'noael': noael_results['noael'],
            'confidence': noael_results['confidence'],
            'confidence_factors': noael_results['confidence_factors']
        },
        'uncertainty_factors': {},
        'probabilistic_analysis': {
            'most_likely_noael': probabilistic_results['most_likely_noael'],
            'credible_interval': probabilistic_results['credible_interval'],
            'noael_probabilities': {str(k): v for k, v in probabilistic_results['noael_probabilities'].items()}
        },
        'benchmark_dose_analysis': {}
    }
    
    # Add uncertainty factors by dose
    for dose in features['doses']:
        if dose in noael_results['uncertainty_factors'] and noael_results['uncertainty_factors'][dose]:
            uncertainty_summary['uncertainty_factors'][str(dose)] = [
                {
                    'endpoint': factor['endpoint'],
                    'factor': factor['factor'],
                    'details': factor['details']
                }
                for factor in noael_results['uncertainty_factors'][dose]
            ]
    
    # Add BMD analysis results
    if 'point_of_departure' in probabilistic_results:
        uncertainty_summary['benchmark_dose_analysis']['point_of_departure'] = {
            'endpoint': probabilistic_results['point_of_departure']['endpoint'],
            'bmdl': probabilistic_results['point_of_departure']['bmdl'],
            'bmr': probabilistic_results['point_of_departure']['bmr']
        }
        uncertainty_summary['benchmark_dose_analysis']['pod_vs_noael'] = probabilistic_results['pod_vs_noael']
    
    # Add key endpoints with high uncertainty
    high_uncertainty_endpoints = []
    
    for dose in features['doses']:
        for factor in noael_results['uncertainty_factors'].get(dose, []):
            if factor['uncertainty_score'] > 70 and factor['endpoint'] not in [e['endpoint'] for e in high_uncertainty_endpoints]:
                high_uncertainty_endpoints.append({
                    'endpoint': factor['endpoint'],
                    'factor': factor['factor'],
                    'uncertainty_score': factor['uncertainty_score']
                })
    
    uncertainty_summary['high_uncertainty_endpoints'] = high_uncertainty_endpoints
    
    # Convert to JSON string
    uncertainty_summary_json = json.dumps(uncertainty_summary, indent=2)
    
    # Create prompt for TxGemma
    prompt = f"""
You are a toxicology expert analyzing data from a preclinical safety study to quantify uncertainty in NOAEL determination.

Below is a summary of the uncertainty analysis:

{uncertainty_summary_json}

Based on this analysis, please:
1. Assess the overall confidence in the NOAEL determination
2. Identify the key factors contributing to uncertainty
3. Recommend the most appropriate point of departure (NOAEL, BMDL, or probabilistic estimate)
4. Suggest additional analyses or data that could reduce uncertainty
5. Provide a weight-of-evidence interpretation considering all available information

Your response should be structured as a JSON object with the following fields:
- overall_confidence: assessment of overall confidence in NOAEL (high, medium, or low)
- key_uncertainty_factors: list of factors contributing most to uncertainty
- recommended_pod: recommended point of departure with justification
- additional_analyses: suggestions for additional analyses to reduce uncertainty
- weight_of_evidence_interpretation: integrated interpretation of all available information
"""
    
    return prompt

# Function to simulate TxGemma response
def simulate_txgemma_response(prompt):
    """
    Simulate TxGemma response for demonstration purposes
    """
    print("Simulating TxGemma response...")
    
    # Extract uncertainty summary from prompt
    try:
        start_idx = prompt.find('{')
        end_idx = prompt.find('\n\nBased on this analysis')
        uncertainty_summary_json = prompt[start_idx:end_idx]
        uncertainty_summary = json.loads(uncertainty_summary_json)
    except:
        print("Error parsing uncertainty summary from prompt")
        uncertainty_summary = {}
    
    # Extract key information
    noael_determination = uncertainty_summary.get('noael_determination', {})
    uncertainty_factors = uncertainty_summary.get('uncertainty_factors', {})
    probabilistic_analysis = uncertainty_summary.get('probabilistic_analysis', {})
    benchmark_dose_analysis = uncertainty_summary.get('benchmark_dose_analysis', {})
    high_uncertainty_endpoints = uncertainty_summary.get('high_uncertainty_endpoints', [])
    
    # Assess overall confidence
    noael_confidence = noael_determination.get('confidence', 'medium')
    confidence_factors = noael_determination.get('confidence_factors', [])
    
    # Count high uncertainty endpoints
    n_high_uncertainty = len(high_uncertainty_endpoints)
    
    # Check credible interval width
    credible_interval = probabilistic_analysis.get('credible_interval', [0, 0])
    if credible_interval[0] is not None and credible_interval[1] is not None:
        interval_ratio = credible_interval[1] / credible_interval[0] if credible_interval[0] > 0 else float('inf')
    else:
        interval_ratio = float('inf')
    
    # Determine overall confidence
    if noael_confidence == 'high' and n_high_uncertainty == 0 and interval_ratio < 3:
        overall_confidence = "high"
    elif noael_confidence == 'low' or n_high_uncertainty >= 3 or interval_ratio > 10:
        overall_confidence = "low"
    else:
        overall_confidence = "medium"
    
    # Identify key uncertainty factors
    key_uncertainty_factors = []
    
    # Add confidence factors
    for factor in confidence_factors:
        if factor not in key_uncertainty_factors:
            key_uncertainty_factors.append(factor)
    
    # Add high uncertainty endpoints
    for endpoint in high_uncertainty_endpoints:
        factor = f"{endpoint['endpoint']}: {endpoint['factor']}"
        if factor not in key_uncertainty_factors:
            key_uncertainty_factors.append(factor)
    
    # Check credible interval
    if interval_ratio > 5:
        key_uncertainty_factors.append(f"Wide credible interval for NOAEL: [{credible_interval[0]}, {credible_interval[1]}]")
    
    # Check POD vs NOAEL
    if 'pod_vs_noael' in benchmark_dose_analysis and "not be protective enough" in benchmark_dose_analysis['pod_vs_noael']:
        key_uncertainty_factors.append("BMD analysis suggests NOAEL may not be protective enough")
    
    # Recommend point of departure
    if overall_confidence == "high":
        recommended_pod = {
            "value": noael_determination.get('noael'),
            "type": "NOAEL",
            "justification": "High confidence in NOAEL determination with minimal uncertainty factors"
        }
    elif 'point_of_departure' in benchmark_dose_analysis:
        recommended_pod = {
            "value": benchmark_dose_analysis['point_of_departure']['bmdl'],
            "type": "BMDL",
            "justification": "BMDL provides a more robust point of departure given the uncertainty in NOAEL determination"
        }
    else:
        recommended_pod = {
            "value": probabilistic_analysis.get('most_likely_noael'),
            "type": "Probabilistic NOAEL",
            "justification": "Probabilistic approach accounts for uncertainty in the determination"
        }
    
    # Suggest additional analyses
    additional_analyses = [
        {
            "analysis": "Additional dose groups",
            "rationale": "Reduce uncertainty due to large dose spacing",
            "expected_impact": "Better characterization of dose-response relationship"
        },
        {
            "analysis": "Increased sample size",
            "rationale": "Reduce statistical uncertainty in key endpoints",
            "expected_impact": "Narrower confidence intervals and more robust statistical analysis"
        },
        {
            "analysis": "Toxicokinetic analysis",
            "rationale": "Understand internal exposure and variability",
            "expected_impact": "Better correlation between dose and effect, potentially reducing uncertainty"
        },
        {
            "analysis": "Mechanistic biomarkers",
            "rationale": "Provide additional evidence for effects seen in high uncertainty endpoints",
            "expected_impact": "Strengthen weight-of-evidence and reduce uncertainty in key findings"
        },
        {
            "analysis": "Benchmark dose modeling with multiple models",
            "rationale": "Compare different dose-response models",
            "expected_impact": "More robust BMD estimates and better characterization of model uncertainty"
        }
    ]
    
    # Provide weight-of-evidence interpretation
    weight_of_evidence_interpretation = ""
    
    if overall_confidence == "high":
        weight_of_evidence_interpretation = f"The weight of evidence strongly supports a NOAEL of {noael_determination.get('noael')} mg/kg/day. The determination is based on clear dose-response relationships with minimal uncertainty. Both traditional NOAEL approach and probabilistic analysis yield consistent results, and benchmark dose modeling provides supporting evidence."
    elif overall_confidence == "medium":
        weight_of_evidence_interpretation = f"The weight of evidence supports a NOAEL of {noael_determination.get('noael')} mg/kg/day, but with moderate uncertainty. The probabilistic analysis suggests a most likely NOAEL of {probabilistic_analysis.get('most_likely_noael')} mg/kg/day with a credible interval of [{credible_interval[0]}, {credible_interval[1]}]. Benchmark dose modeling provides a complementary approach, with a BMDL of {benchmark_dose_analysis.get('point_of_departure', {}).get('bmdl')} mg/kg/day. Given the uncertainty factors identified, a conservative approach using the lower bound of the credible interval or the BMDL as the point of departure may be warranted."
    else:
        weight_of_evidence_interpretation = f"The weight of evidence suggests substantial uncertainty in the NOAEL determination. The traditional NOAEL approach yields {noael_determination.get('noael')} mg/kg/day, but with low confidence. The probabilistic analysis indicates a wide credible interval of [{credible_interval[0]}, {credible_interval[1]}], reflecting the high uncertainty. Benchmark dose modeling provides a BMDL of {benchmark_dose_analysis.get('point_of_departure', {}).get('bmdl')} mg/kg/day, which may be more appropriate as a point of departure given the limitations of the dataset. Additional studies or analyses are strongly recommended to reduce uncertainty before finalizing risk assessment decisions."
    
    # Create simulated response
    response = {
        "overall_confidence": overall_confidence,
        "key_uncertainty_factors": key_uncertainty_factors,
        "recommended_pod": recommended_pod,
        "additional_analyses": additional_analyses,
        "weight_of_evidence_interpretation": weight_of_evidence_interpretation
    }
    
    return json.dumps(response, indent=2)

# Function to visualize uncertainty analysis
def visualize_uncertainty_analysis(features, analysis_results, noael_results, bmd_results, probabilistic_results, txgemma_response):
    """
    Create visualizations of the uncertainty analysis
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
    fig.suptitle('Uncertainty Quantification and Confidence Assessment for NOAEL Determination', fontsize=16)
    
    # Plot 1: Uncertainty scores by endpoint
    ax = axs[0, 0]
    
    # Collect uncertainty scores
    endpoint_uncertainty = {}
    
    # From laboratory tests
    if 'laboratory_tests' in analysis_results:
        for test, test_results in analysis_results['laboratory_tests'].items():
            for dose, result in test_results.get('dose_effects', {}).items():
                if 'uncertainty_score' in result:
                    endpoint_uncertainty[test] = max(endpoint_uncertainty.get(test, 0), result['uncertainty_score'])
    
    # From body weights
    if 'body_weights' in analysis_results:
        for dose, result in analysis_results['body_weights'].get('dose_effects', {}).items():
            if 'uncertainty_score' in result:
                endpoint_uncertainty['Body Weight'] = max(endpoint_uncertainty.get('Body Weight', 0), result['uncertainty_score'])
    
    # From histopathology
    if 'histopathology' in analysis_results:
        for organ, organ_results in analysis_results['histopathology'].items():
            for finding, finding_results in organ_results.items():
                for dose, result in finding_results.get('dose_effects', {}).items():
                    if 'uncertainty_score' in result:
                        endpoint = f"{organ} - {finding}"
                        endpoint_uncertainty[endpoint] = max(endpoint_uncertainty.get(endpoint, 0), result['uncertainty_score'])
    
    # Sort by uncertainty score
    sorted_endpoints = sorted(endpoint_uncertainty.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 10 endpoints
    top_endpoints = sorted_endpoints[:10]
    
    if top_endpoints:
        endpoints = [e[0] for e in top_endpoints]
        scores = [e[1] for e in top_endpoints]
        
        # Create horizontal bar chart
        bars = ax.barh(endpoints, scores, color='skyblue')
        
        # Add threshold lines
        ax.axvline(x=25, color='green', linestyle='--', label='High Confidence')
        ax.axvline(x=50, color='orange', linestyle='--', label='Medium Confidence')
        ax.axvline(x=75, color='red', linestyle='--', label='Low Confidence')
        
        # Color bars based on confidence
        for i, score in enumerate(scores):
            if score < 25:
                bars[i].set_color('green')
            elif score < 50:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('red')
        
        ax.set_xlabel('Uncertainty Score')
        ax.set_title('Endpoint Uncertainty Scores')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No uncertainty data available', ha='center', va='center')
    
    # Plot 2: Probabilistic NOAEL distribution
    ax = axs[0, 1]
    
    if 'noael_probabilities' in probabilistic_results:
        # Get probabilities
        doses = sorted(probabilistic_results['noael_probabilities'].keys())
        probs = [probabilistic_results['noael_probabilities'][d] for d in doses]
        
        # Create bar chart
        bars = ax.bar(doses, probs, color='skyblue')
        
        # Highlight most likely NOAEL
        most_likely_idx = doses.index(probabilistic_results['most_likely_noael'])
        bars[most_likely_idx].set_color('green')
        
        # Add credible interval
        ci_lower, ci_upper = probabilistic_results['credible_interval']
        ax.axvline(x=ci_lower, color='red', linestyle='--', label=f'95% Credible Interval: [{ci_lower}, {ci_upper}]')
        ax.axvline(x=ci_upper, color='red', linestyle='--')
        
        # Add NOAEL from traditional approach
        if noael_results['noael'] is not None:
            ax.axvline(x=noael_results['noael'], color='blue', linestyle='-', label=f"Traditional NOAEL: {noael_results['noael']}")
        
        ax.set_xlabel('Dose (mg/kg/day)')
        ax.set_ylabel('Probability')
        ax.set_title('Probabilistic NOAEL Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No probabilistic data available', ha='center', va='center')
    
    # Plot 3: Benchmark dose modeling
    ax = axs[1, 0]
    
    # Select an endpoint for demonstration
    selected_endpoint = None
    selected_bmd_data = None
    
    if 'laboratory_tests' in bmd_results and bmd_results['laboratory_tests']:
        selected_endpoint = list(bmd_results['laboratory_tests'].keys())[0]
        selected_bmd_data = bmd_results['laboratory_tests'][selected_endpoint]
    elif 'body_weights' in bmd_results:
        selected_endpoint = 'Body Weight'
        selected_bmd_data = bmd_results['body_weights']
    elif 'histopathology' in bmd_results and bmd_results['histopathology']:
        organ = list(bmd_results['histopathology'].keys())[0]
        finding = list(bmd_results['histopathology'][organ].keys())[0]
        selected_endpoint = f"{organ} - {finding}"
        selected_bmd_data = bmd_results['histopathology'][organ][finding]
    
    if selected_bmd_data:
        # Get dose-response data
        doses = selected_bmd_data['doses']
        responses = selected_bmd_data['responses']
        predicted = selected_bmd_data['predicted']
        
        # Plot observed data
        ax.scatter(doses, responses, color='blue', label='Observed')
        
        # Plot model fit
        dose_range = np.linspace(min(doses), max(doses), 100)
        
        if selected_bmd_data['model'] == 'Hill':
            max_effect = selected_bmd_data['parameters']['max_effect']
            k = selected_bmd_data['parameters']['k']
            n = selected_bmd_data['parameters']['n']
            
            predicted_curve = [max_effect * x**n / (k**n + x**n) for x in dose_range]
            ax.plot(dose_range, predicted_curve, color='red', label='Hill Model')
        elif selected_bmd_data['model'] == 'Exponential':
            a = selected_bmd_data['parameters']['a']
            b = selected_bmd_data['parameters']['b']
            
            predicted_curve = [a * np.exp(b * x) for x in dose_range]
            ax.plot(dose_range, predicted_curve, color='red', label='Exponential Model')
        elif selected_bmd_data['model'] == 'Logistic':
            a = selected_bmd_data['parameters']['a']
            b = selected_bmd_data['parameters']['b']
            background = selected_bmd_data['parameters']['background']
            
            predicted_curve = [background + (1 - background) / (1 + np.exp(-a - b*np.log(x))) for x in dose_range]
            ax.plot(dose_range, predicted_curve, color='red', label='Logistic Model')
        
        # Add BMD and BMDL
        if selected_bmd_data.get('bmd') is not None:
            ax.axvline(x=selected_bmd_data['bmd'], color='green', linestyle='--', 
                      label=f"BMD: {selected_bmd_data['bmd']:.1f}")
        
        if selected_bmd_data.get('bmdl') is not None:
            ax.axvline(x=selected_bmd_data['bmdl'], color='orange', linestyle='--', 
                      label=f"BMDL: {selected_bmd_data['bmdl']:.1f}")
        
        # Add BMR line
        if 'responses' in selected_bmd_data and selected_bmd_data['responses']:
            if selected_endpoint == 'Body Weight':
                # For body weight, BMR is negative
                bmr_value = selected_bmd_data['bmr'] * -100  # Convert to percent
                ax.axhline(y=bmr_value, color='red', linestyle=':', 
                          label=f"BMR: {bmr_value:.1f}%")
            else:
                # For other endpoints, BMR is positive
                bmr_value = selected_bmd_data['bmr'] * 100  # Convert to percent
                ax.axhline(y=bmr_value, color='red', linestyle=':', 
                          label=f"BMR: {bmr_value:.1f}%")
        
        ax.set_xlabel('Dose (mg/kg/day)')
        ax.set_ylabel('Response')
        ax.set_title(f'Benchmark Dose Modeling: {selected_endpoint}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No benchmark dose data available', ha='center', va='center')
    
    # Plot 4: TxGemma assessment
    ax = axs[1, 1]
    ax.axis('off')  # Turn off axis
    
    # Create text summary from TxGemma response
    if txgemma_data:
        text = "TxGemma Assessment:\n\n"
        text += f"Overall Confidence: {txgemma_data.get('overall_confidence', 'Not specified').capitalize()}\n\n"
        
        text += "Recommended Point of Departure:\n"
        pod = txgemma_data.get('recommended_pod', {})
        text += f"- Type: {pod.get('type', 'Not specified')}\n"
        text += f"- Value: {pod.get('value', 'Not specified')} mg/kg/day\n\n"
        
        text += "Key Uncertainty Factors:\n"
        for factor in txgemma_data.get('key_uncertainty_factors', [])[:5]:  # Show top 5
            text += f"- {factor}\n"
        
        text += "\nWeight-of-Evidence Interpretation:\n"
        text += txgemma_data.get('weight_of_evidence_interpretation', 'Not provided')
        
        ax.text(0, 1, text, va='top', ha='left', wrap=True, fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('/home/ubuntu/noael_project/demo_code/use_case5_results.png')
    print("Visualization saved to /home/ubuntu/noael_project/demo_code/use_case5_results.png")
    
    return '/home/ubuntu/noael_project/demo_code/use_case5_results.png'

# Main function
def main():
    """
    Main function to demonstrate TxGemma for uncertainty quantification in NOAEL determination
    """
    print("Starting uncertainty quantification demonstration using TxGemma...")
    
    # Step 1: Load SEND domains
    domains = load_send_domains(SEND_DATA_PATH)
    
    # Step 2: Extract features from SEND domains
    features = extract_features(domains)
    
    # Step 3: Analyze dose-response relationships with uncertainty
    analysis_results = analyze_dose_response_with_uncertainty(features)
    
    # Step 4: Determine NOAEL with uncertainty quantification
    noael_results = determine_noael_with_uncertainty(features, analysis_results)
    print(f"NOAEL determination: {noael_results['noael']} (confidence: {noael_results['confidence']})")
    
    # Step 5: Perform benchmark dose modeling
    bmd_results = perform_benchmark_dose_modeling(features, analysis_results)
    
    # Step 6: Perform probabilistic NOAEL determination
    probabilistic_results = perform_probabilistic_noael(features, analysis_results, bmd_results)
    print(f"Probabilistic NOAEL: {probabilistic_results['most_likely_noael']} (95% CI: {probabilistic_results['credible_interval']})")
    
    # Step 7: Prepare input for TxGemma
    txgemma_prompt = prepare_txgemma_input(features, analysis_results, noael_results, bmd_results, probabilistic_results)
    
    # Step 8: Get TxGemma response (simulated for demonstration)
    txgemma_response = simulate_txgemma_response(txgemma_prompt)
    print("\nTxGemma Response:")
    print(txgemma_response)
    
    # Step 9: Visualize uncertainty analysis
    visualization_path = visualize_uncertainty_analysis(features, analysis_results, noael_results, bmd_results, probabilistic_results, txgemma_response)
    
    print("\nDemonstration completed successfully!")
    print(f"Results visualization saved to: {visualization_path}")
    
    return {
        'features': features,
        'analysis_results': analysis_results,
        'noael_results': noael_results,
        'bmd_results': bmd_results,
        'probabilistic_results': probabilistic_results,
        'txgemma_response': json.loads(txgemma_response),
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    main()
