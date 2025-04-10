"""
Use Case 4: Adverse Effect Pattern Recognition and Classification

This script demonstrates how to use TxGemma to recognize and classify patterns of adverse effects
in SEND datasets across multiple domains to identify specific toxicity signatures.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
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
    Create mock SEND data for demonstration purposes with specific toxicity patterns
    """
    domains = {}
    
    # Create mock demographics (DM) domain
    domains['dm'] = pd.DataFrame({
        'USUBJID': [f'SUBJ-{i:03d}' for i in range(1, 61)],
        'SEX': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'] * 7 + ['M', 'M', 'F', 'F'],
        'ARMCD': ['C', 'C', 'LD', 'LD', 'MD', 'MD', 'HD', 'HD'] * 7 + ['C', 'HD', 'C', 'HD'],
        'ARM': ['Control', 'Control', 'Low Dose', 'Low Dose', 
                'Mid Dose', 'Mid Dose', 'High Dose', 'High Dose'] * 7 + 
               ['Control', 'High Dose', 'Control', 'High Dose'],
        'SPECIES': ['RAT'] * 60,
        'STRAIN': ['WISTAR'] * 60
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
    
    # Define toxicity patterns
    toxicity_patterns = {
        'hepatotoxicity': {
            'biomarkers': {
                'ALT': {'direction': 'increase', 'magnitude': 2.5},
                'AST': {'direction': 'increase', 'magnitude': 2.0},
                'ALP': {'direction': 'increase', 'magnitude': 1.5},
                'BILI': {'direction': 'increase', 'magnitude': 1.3},
                'ALB': {'direction': 'decrease', 'magnitude': 0.8}
            },
            'histopathology': {
                'LIVER': ['Hepatocellular necrosis', 'Hepatocellular hypertrophy', 'Inflammation']
            },
            'organ_weights': {
                'LIVER': {'direction': 'increase', 'magnitude': 1.2}
            },
            'dose_response': {
                'C': 0.0,
                'LD': 0.1,
                'MD': 0.5,
                'HD': 1.0
            }
        },
        'nephrotoxicity': {
            'biomarkers': {
                'BUN': {'direction': 'increase', 'magnitude': 1.8},
                'CREAT': {'direction': 'increase', 'magnitude': 1.5},
                'K': {'direction': 'increase', 'magnitude': 1.2},
                'NA': {'direction': 'decrease', 'magnitude': 0.9},
                'CL': {'direction': 'decrease', 'magnitude': 0.9}
            },
            'histopathology': {
                'KIDNEY': ['Tubular necrosis', 'Tubular degeneration', 'Inflammation']
            },
            'organ_weights': {
                'KIDNEY': {'direction': 'increase', 'magnitude': 1.1}
            },
            'dose_response': {
                'C': 0.0,
                'LD': 0.0,
                'MD': 0.3,
                'HD': 0.8
            }
        },
        'hematotoxicity': {
            'biomarkers': {
                'WBC': {'direction': 'decrease', 'magnitude': 0.7},
                'RBC': {'direction': 'decrease', 'magnitude': 0.8},
                'HGB': {'direction': 'decrease', 'magnitude': 0.8},
                'HCT': {'direction': 'decrease', 'magnitude': 0.8},
                'PLT': {'direction': 'decrease', 'magnitude': 0.6}
            },
            'histopathology': {
                'BONE MARROW': ['Hypocellularity', 'Decreased erythropoiesis', 'Decreased myelopoiesis'],
                'SPLEEN': ['Lymphoid depletion']
            },
            'organ_weights': {
                'SPLEEN': {'direction': 'decrease', 'magnitude': 0.9}
            },
            'dose_response': {
                'C': 0.0,
                'LD': 0.0,
                'MD': 0.2,
                'HD': 0.7
            }
        },
        'cardiotoxicity': {
            'biomarkers': {
                'CK': {'direction': 'increase', 'magnitude': 1.7},
                'LDH': {'direction': 'increase', 'magnitude': 1.5},
                'TROP': {'direction': 'increase', 'magnitude': 2.0}
            },
            'histopathology': {
                'HEART': ['Myocardial degeneration', 'Myocardial necrosis', 'Inflammation']
            },
            'organ_weights': {
                'HEART': {'direction': 'increase', 'magnitude': 1.1}
            },
            'dose_response': {
                'C': 0.0,
                'LD': 0.0,
                'MD': 0.1,
                'HD': 0.5
            }
        }
    }
    
    # Assign toxicity patterns to subjects
    # For this demo, we'll make high dose animals show multiple toxicity patterns
    subject_patterns = {}
    
    for subj in domains['dm']['USUBJID']:
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        
        # Initialize with empty list
        subject_patterns[subj] = []
        
        # Assign patterns based on dose group
        if arm == 'HD':
            # High dose animals show multiple toxicity patterns
            # Hepatotoxicity in all high dose animals
            subject_patterns[subj].append('hepatotoxicity')
            
            # Nephrotoxicity in 80% of high dose animals
            if np.random.random() < 0.8:
                subject_patterns[subj].append('nephrotoxicity')
            
            # Hematotoxicity in 70% of high dose animals
            if np.random.random() < 0.7:
                subject_patterns[subj].append('hematotoxicity')
            
            # Cardiotoxicity in 50% of high dose animals
            if np.random.random() < 0.5:
                subject_patterns[subj].append('cardiotoxicity')
        
        elif arm == 'MD':
            # Mid dose animals show fewer toxicity patterns
            # Hepatotoxicity in 80% of mid dose animals
            if np.random.random() < 0.8:
                subject_patterns[subj].append('hepatotoxicity')
            
            # Nephrotoxicity in 40% of mid dose animals
            if np.random.random() < 0.4:
                subject_patterns[subj].append('nephrotoxicity')
            
            # Hematotoxicity in 20% of mid dose animals
            if np.random.random() < 0.2:
                subject_patterns[subj].append('hematotoxicity')
        
        elif arm == 'LD':
            # Low dose animals show minimal toxicity
            # Hepatotoxicity in 20% of low dose animals
            if np.random.random() < 0.2:
                subject_patterns[subj].append('hepatotoxicity')
    
    # Create mock laboratory results (LB) domain with toxicity patterns
    lb_data = []
    
    # Define baseline values for all biomarkers
    baseline_values = {
        # Liver biomarkers
        'ALT': 45, 'AST': 80, 'ALP': 120, 'BILI': 0.3, 'ALB': 4.0,
        # Kidney biomarkers
        'BUN': 15, 'CREAT': 0.6, 'K': 4.5, 'NA': 140, 'CL': 105,
        # Hematology biomarkers
        'WBC': 10, 'RBC': 8, 'HGB': 15, 'HCT': 45, 'PLT': 800,
        # Cardiac biomarkers
        'CK': 200, 'LDH': 300, 'TROP': 0.01
    }
    
    # Generate laboratory data based on toxicity patterns
    for subj in domains['dm']['USUBJID']:
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        patterns = subject_patterns[subj]
        
        for biomarker, base_value in baseline_values.items():
            # Initialize with no effect
            effect_multiplier = 1.0
            
            # Apply effects from each toxicity pattern
            for pattern in patterns:
                if biomarker in toxicity_patterns[pattern]['biomarkers']:
                    # Get effect details
                    effect = toxicity_patterns[pattern]['biomarkers'][biomarker]
                    dose_factor = toxicity_patterns[pattern]['dose_response'][arm]
                    
                    # Calculate effect magnitude based on dose
                    if effect['direction'] == 'increase':
                        # For increases, multiplier > 1
                        pattern_multiplier = 1.0 + (effect['magnitude'] - 1.0) * dose_factor
                    else:
                        # For decreases, multiplier < 1
                        pattern_multiplier = 1.0 - (1.0 - effect['magnitude']) * dose_factor
                    
                    # Apply the strongest effect if multiple patterns affect the same biomarker
                    if (effect['direction'] == 'increase' and pattern_multiplier > effect_multiplier) or \
                       (effect['direction'] == 'decrease' and pattern_multiplier < effect_multiplier):
                        effect_multiplier = pattern_multiplier
            
            # Add random variation (10% CV)
            value = base_value * effect_multiplier * np.random.normal(1, 0.1)
            
            # Determine units based on biomarker
            if biomarker in ['ALT', 'AST', 'ALP', 'CK', 'LDH']:
                units = 'U/L'
            elif biomarker in ['BILI', 'CREAT']:
                units = 'mg/dL'
            elif biomarker in ['BUN', 'K', 'NA', 'CL']:
                units = 'mmol/L'
            elif biomarker in ['ALB']:
                units = 'g/dL'
            elif biomarker in ['WBC', 'PLT']:
                units = '10^9/L'
            elif biomarker in ['RBC']:
                units = '10^12/L'
            elif biomarker in ['HGB']:
                units = 'g/dL'
            elif biomarker in ['HCT']:
                units = '%'
            elif biomarker in ['TROP']:
                units = 'ng/mL'
            else:
                units = ''
            
            lb_data.append({
                'USUBJID': subj,
                'LBTEST': biomarker,
                'LBSTRESN': value,
                'LBSTRESU': units
            })
    
    domains['lb'] = pd.DataFrame(lb_data)
    
    # Create mock histopathology findings (MI) domain
    mi_data = []
    
    # Define severity grades
    severity_grades = {0: 'Normal', 1: 'Minimal', 2: 'Mild', 3: 'Moderate', 4: 'Marked'}
    
    # Generate histopathology data based on toxicity patterns
    for subj in domains['dm']['USUBJID']:
        patterns = subject_patterns[subj]
        
        for pattern in patterns:
            for organ, findings in toxicity_patterns[pattern]['histopathology'].items():
                # Determine severity based on dose
                arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
                dose_factor = toxicity_patterns[pattern]['dose_response'][arm]
                
                # Skip if no effect at this dose
                if dose_factor == 0:
                    continue
                
                # Calculate severity grade (0-4)
                # Higher doses get higher severity
                if dose_factor < 0.2:
                    severity = 1  # Minimal
                elif dose_factor < 0.5:
                    severity = 2  # Mild
                elif dose_factor < 0.8:
                    severity = 3  # Moderate
                else:
                    severity = 4  # Marked
                
                # Add findings with some variability
                for finding in findings:
                    # Add some variability - not all animals show all findings
                    if np.random.random() < (0.5 + 0.5 * dose_factor):
                        # Add some variability to severity
                        actual_severity = max(1, min(4, severity + np.random.randint(-1, 2)))
                        
                        mi_data.append({
                            'USUBJID': subj,
                            'MISPEC': organ,
                            'MISTRESC': finding,
                            'MISEV': actual_severity,
                            'MISEVTXT': severity_grades[actual_severity]
                        })
    
    domains['mi'] = pd.DataFrame(mi_data)
    
    # Create mock organ measurements (OM) domain
    om_data = []
    
    # Generate organ weight data based on toxicity patterns
    for subj in domains['dm']['USUBJID']:
        patterns = subject_patterns[subj]
        sex = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'SEX'].values[0]
        
        # Define baseline organ weights based on sex
        baseline_weights = {
            'LIVER': 10.0 if sex == 'M' else 8.0,
            'KIDNEY': 1.2 if sex == 'M' else 1.0,
            'HEART': 1.0 if sex == 'M' else 0.8,
            'SPLEEN': 0.8 if sex == 'M' else 0.7
        }
        
        for organ, base_weight in baseline_weights.items():
            # Initialize with no effect
            effect_multiplier = 1.0
            
            # Apply effects from each toxicity pattern
            for pattern in patterns:
                if organ in toxicity_patterns[pattern].get('organ_weights', {}):
                    # Get effect details
                    effect = toxicity_patterns[pattern]['organ_weights'][organ]
                    arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
                    dose_factor = toxicity_patterns[pattern]['dose_response'][arm]
                    
                    # Calculate effect magnitude based on dose
                    if effect['direction'] == 'increase':
                        # For increases, multiplier > 1
                        pattern_multiplier = 1.0 + (effect['magnitude'] - 1.0) * dose_factor
                    else:
                        # For decreases, multiplier < 1
                        pattern_multiplier = 1.0 - (1.0 - effect['magnitude']) * dose_factor
                    
                    # Apply the strongest effect if multiple patterns affect the same organ
                    if (effect['direction'] == 'increase' and pattern_multiplier > effect_multiplier) or \
                       (effect['direction'] == 'decrease' and pattern_multiplier < effect_multiplier):
                        effect_multiplier = pattern_multiplier
            
            # Add random variation (5% CV)
            weight = base_weight * effect_multiplier * np.random.normal(1, 0.05)
            
            om_data.append({
                'USUBJID': subj,
                'OMSPEC': organ,
                'OMSTRESN': weight,
                'OMSTRESU': 'g'
            })
    
    domains['om'] = pd.DataFrame(om_data)
    
    # Create mock clinical observations (CL) domain
    cl_data = []
    
    # Define clinical observations associated with toxicity patterns
    clinical_observations = {
        'hepatotoxicity': ['Jaundice', 'Lethargy'],
        'nephrotoxicity': ['Polyuria', 'Dehydration'],
        'hematotoxicity': ['Pallor', 'Lethargy'],
        'cardiotoxicity': ['Dyspnea', 'Lethargy']
    }
    
    # Generate clinical observations based on toxicity patterns
    for subj in domains['dm']['USUBJID']:
        patterns = subject_patterns[subj]
        
        for pattern in patterns:
            # Only add clinical observations for severe cases
            arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
            dose_factor = toxicity_patterns[pattern]['dose_response'][arm]
            
            # Skip if no effect at this dose or effect is mild
            if dose_factor < 0.5:
                continue
            
            # Add observations with some variability
            for observation in clinical_observations.get(pattern, []):
                # Add some variability - not all animals show all observations
                if np.random.random() < (0.3 + 0.7 * dose_factor):
                    cl_data.append({
                        'USUBJID': subj,
                        'CLTERM': observation,
                        'CLSEV': 'Moderate' if dose_factor < 0.8 else 'Severe'
                    })
    
    domains['cl'] = pd.DataFrame(cl_data)
    
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
    
    # Store the toxicity pattern assignments for validation
    domains['_toxicity_patterns'] = subject_patterns
    
    return domains

# Function to extract features from multiple domains
def extract_multidomain_features(domains):
    """
    Extract features from multiple SEND domains for pattern recognition
    """
    print("Extracting features from multiple domains...")
    
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
    
    # Create subject-level feature matrix
    subject_features = {}
    
    # Extract laboratory test results
    if 'lb' in domains:
        lb_data = domains['lb']
        
        if 'LBTEST' in lb_data.columns and 'LBSTRESN' in lb_data.columns:
            # Pivot to get one row per subject, one column per biomarker
            lb_pivot = lb_data.pivot_table(
                index='USUBJID', 
                columns='LBTEST', 
                values='LBSTRESN', 
                aggfunc='mean'
            )
            
            # Add prefix to column names
            lb_pivot.columns = ['LB_' + col for col in lb_pivot.columns]
            
            # Add to subject features
            for subj, row in lb_pivot.iterrows():
                if subj not in subject_features:
                    subject_features[subj] = {}
                
                for col, value in row.items():
                    subject_features[subj][col] = value
    
    # Extract histopathology findings
    if 'mi' in domains:
        mi_data = domains['mi']
        
        if 'MISPEC' in mi_data.columns and 'MISTRESC' in mi_data.columns:
            # Create features for each organ-finding combination
            for _, row in mi_data.iterrows():
                subj = row['USUBJID']
                organ = row['MISPEC']
                finding = row['MISTRESC']
                severity = row['MISEV'] if 'MISEV' in row else 1
                
                if subj not in subject_features:
                    subject_features[subj] = {}
                
                # Create feature name
                feature_name = f"MI_{organ}_{finding}"
                
                # Store severity as feature value
                subject_features[subj][feature_name] = severity
    
    # Extract organ measurements
    if 'om' in domains:
        om_data = domains['om']
        
        if 'OMSPEC' in om_data.columns and 'OMSTRESN' in om_data.columns:
            # Pivot to get one row per subject, one column per organ
            om_pivot = om_data.pivot_table(
                index='USUBJID', 
                columns='OMSPEC', 
                values='OMSTRESN', 
                aggfunc='mean'
            )
            
            # Add prefix to column names
            om_pivot.columns = ['OM_' + col for col in om_pivot.columns]
            
            # Add to subject features
            for subj, row in om_pivot.iterrows():
                if subj not in subject_features:
                    subject_features[subj] = {}
                
                for col, value in row.items():
                    subject_features[subj][col] = value
    
    # Extract clinical observations
    if 'cl' in domains:
        cl_data = domains['cl']
        
        if 'CLTERM' in cl_data.columns:
            # Create features for each observation
            for _, row in cl_data.iterrows():
                subj = row['USUBJID']
                term = row['CLTERM']
                severity = 1  # Default severity if not specified
                
                if 'CLSEV' in row:
                    if row['CLSEV'] == 'Mild':
                        severity = 1
                    elif row['CLSEV'] == 'Moderate':
                        severity = 2
                    elif row['CLSEV'] == 'Severe':
                        severity = 3
                
                if subj not in subject_features:
                    subject_features[subj] = {}
                
                # Create feature name
                feature_name = f"CL_{term}"
                
                # Store severity as feature value
                subject_features[subj][feature_name] = severity
    
    # Convert to dataframe
    subject_df = pd.DataFrame.from_dict(subject_features, orient='index')
    
    # Add subject metadata
    if 'dm' in domains:
        dm_data = domains['dm'][['USUBJID', 'SEX', 'ARMCD']]
        subject_df = subject_df.reset_index().rename(columns={'index': 'USUBJID'})
        subject_df = subject_df.merge(dm_data, on='USUBJID', how='left')
        subject_df = subject_df.set_index('USUBJID')
    
    # Add dose information
    if 'ex' in domains and 'EXDOSE' in domains['ex'].columns:
        ex_data = domains['ex'][['USUBJID', 'EXDOSE']].drop_duplicates()
        subject_df = subject_df.reset_index()
        subject_df = subject_df.merge(ex_data, on='USUBJID', how='left')
        subject_df = subject_df.set_index('USUBJID')
    
    features['subject_features'] = subject_df
    print(f"Created feature matrix with {subject_df.shape[0]} subjects and {subject_df.shape[1]} features")
    
    return features

# Function to identify toxicity patterns
def identify_toxicity_patterns(features):
    """
    Identify toxicity patterns in the feature matrix
    """
    print("Identifying toxicity patterns...")
    
    results = {}
    
    # Get feature matrix
    subject_df = features['subject_features']
    
    # Define organ systems and their associated biomarkers/findings
    organ_systems = {
        'Liver': {
            'biomarkers': ['LB_ALT', 'LB_AST', 'LB_ALP', 'LB_BILI', 'LB_ALB'],
            'histopathology': ['MI_LIVER_'],
            'organ_weights': ['OM_LIVER'],
            'clinical_signs': ['CL_Jaundice']
        },
        'Kidney': {
            'biomarkers': ['LB_BUN', 'LB_CREAT', 'LB_K', 'LB_NA', 'LB_CL'],
            'histopathology': ['MI_KIDNEY_'],
            'organ_weights': ['OM_KIDNEY'],
            'clinical_signs': ['CL_Polyuria', 'CL_Dehydration']
        },
        'Hematological': {
            'biomarkers': ['LB_WBC', 'LB_RBC', 'LB_HGB', 'LB_HCT', 'LB_PLT'],
            'histopathology': ['MI_BONE MARROW_', 'MI_SPLEEN_'],
            'organ_weights': ['OM_SPLEEN'],
            'clinical_signs': ['CL_Pallor']
        },
        'Cardiovascular': {
            'biomarkers': ['LB_CK', 'LB_LDH', 'LB_TROP'],
            'histopathology': ['MI_HEART_'],
            'organ_weights': ['OM_HEART'],
            'clinical_signs': ['CL_Dyspnea']
        }
    }
    
    # Calculate organ-specific toxicity scores
    toxicity_scores = pd.DataFrame(index=subject_df.index)
    
    for organ, features_dict in organ_systems.items():
        # Initialize scores for this organ system
        organ_scores = pd.Series(0.0, index=subject_df.index)
        
        # Add biomarker contributions
        biomarker_cols = [col for col in subject_df.columns if any(col.startswith(b) for b in features_dict['biomarkers'])]
        if biomarker_cols:
            # Standardize biomarker values
            biomarker_data = subject_df[biomarker_cols].copy()
            
            # Fill missing values with median
            for col in biomarker_data.columns:
                biomarker_data[col] = biomarker_data[col].fillna(biomarker_data[col].median())
            
            # Get control group data for normalization
            control_mask = subject_df['ARMCD'] == 'C'
            if control_mask.any():
                control_means = biomarker_data[control_mask].mean()
                control_stds = biomarker_data[control_mask].std()
                
                # Calculate z-scores relative to control group
                for col in biomarker_data.columns:
                    # Handle special cases for biomarkers where decreases are adverse
                    if col in ['LB_ALB', 'LB_WBC', 'LB_RBC', 'LB_HGB', 'LB_HCT', 'LB_PLT']:
                        # For these biomarkers, decreases are adverse
                        biomarker_data[col] = (control_means[col] - biomarker_data[col]) / control_stds[col]
                    else:
                        # For most biomarkers, increases are adverse
                        biomarker_data[col] = (biomarker_data[col] - control_means[col]) / control_stds[col]
                
                # Sum absolute z-scores for biomarker score
                biomarker_scores = biomarker_data.abs().sum(axis=1)
                
                # Add to organ scores with weight
                organ_scores += biomarker_scores * 0.4  # 40% weight to biomarkers
        
        # Add histopathology contributions
        histo_cols = [col for col in subject_df.columns if any(col.startswith(h) for h in features_dict['histopathology'])]
        if histo_cols:
            # Sum severity scores for histopathology
            histo_data = subject_df[histo_cols].copy().fillna(0)
            histo_scores = histo_data.sum(axis=1)
            
            # Add to organ scores with weight
            organ_scores += histo_scores * 0.4  # 40% weight to histopathology
        
        # Add organ weight contributions
        weight_cols = [col for col in subject_df.columns if any(col.startswith(w) for w in features_dict['organ_weights'])]
        if weight_cols:
            # Calculate percent change from control
            weight_data = subject_df[weight_cols].copy()
            
            # Get control group data for normalization
            control_mask = subject_df['ARMCD'] == 'C'
            if control_mask.any():
                control_means = weight_data[control_mask].mean()
                
                # Calculate percent change
                for col in weight_data.columns:
                    weight_data[col] = abs((weight_data[col] - control_means[col]) / control_means[col] * 100)
                
                # Add to organ scores with weight
                weight_scores = weight_data.sum(axis=1)
                organ_scores += weight_scores * 0.1  # 10% weight to organ weights
        
        # Add clinical signs contributions
        sign_cols = [col for col in subject_df.columns if any(col.startswith(c) for c in features_dict['clinical_signs'])]
        if sign_cols:
            # Sum severity scores for clinical signs
            sign_data = subject_df[sign_cols].copy().fillna(0)
            sign_scores = sign_data.sum(axis=1)
            
            # Add to organ scores with weight
            organ_scores += sign_scores * 0.1  # 10% weight to clinical signs
        
        # Add to toxicity scores dataframe
        toxicity_scores[f"{organ}_score"] = organ_scores
    
    # Normalize scores to 0-10 scale
    for col in toxicity_scores.columns:
        max_val = toxicity_scores[col].max()
        if max_val > 0:
            toxicity_scores[col] = toxicity_scores[col] / max_val * 10
    
    # Add dose information
    toxicity_scores['EXDOSE'] = subject_df['EXDOSE']
    toxicity_scores['ARMCD'] = subject_df['ARMCD']
    
    results['toxicity_scores'] = toxicity_scores
    
    # Classify toxicity patterns
    # Define thresholds for each organ system
    thresholds = {
        'Liver_score': 3.0,
        'Kidney_score': 3.0,
        'Hematological_score': 3.0,
        'Cardiovascular_score': 3.0
    }
    
    # Classify each subject
    toxicity_patterns = {}
    
    for subj, row in toxicity_scores.iterrows():
        patterns = []
        
        for organ, threshold in thresholds.items():
            if row[organ] >= threshold:
                # Convert score column name to pattern name
                pattern = organ.replace('_score', 'toxicity').lower()
                patterns.append(pattern)
        
        toxicity_patterns[subj] = patterns
    
    results['toxicity_patterns'] = toxicity_patterns
    
    # Calculate pattern prevalence by dose group
    pattern_prevalence = {}
    
    for pattern in ['hepatotoxicity', 'nephrotoxicity', 'hematologicaltoxicity', 'cardiovasculartoxicity']:
        prevalence_by_dose = {}
        
        for dose in features['doses']:
            # Get subjects in this dose group
            subjects = features['dose_groups'][dose]
            
            # Count subjects with this pattern
            count = sum(1 for subj in subjects if pattern in toxicity_patterns.get(subj, []))
            
            # Calculate prevalence
            prevalence = count / len(subjects) * 100 if subjects else 0
            prevalence_by_dose[dose] = prevalence
        
        pattern_prevalence[pattern] = prevalence_by_dose
    
    results['pattern_prevalence'] = pattern_prevalence
    
    # Identify co-occurring patterns
    pattern_co_occurrence = {}
    
    all_patterns = ['hepatotoxicity', 'nephrotoxicity', 'hematologicaltoxicity', 'cardiovasculartoxicity']
    
    for i, pattern1 in enumerate(all_patterns):
        for pattern2 in all_patterns[i+1:]:
            # Count subjects with both patterns
            count = sum(1 for patterns in toxicity_patterns.values() 
                       if pattern1 in patterns and pattern2 in patterns)
            
            # Calculate co-occurrence percentage
            total_subjects = len(toxicity_patterns)
            co_occurrence = count / total_subjects * 100 if total_subjects else 0
            
            pattern_co_occurrence[f"{pattern1}_{pattern2}"] = co_occurrence
    
    results['pattern_co_occurrence'] = pattern_co_occurrence
    
    return results

# Function to visualize toxicity patterns
def visualize_toxicity_patterns(features, pattern_results):
    """
    Create visualizations of toxicity patterns
    """
    print("Creating visualizations...")
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adverse Effect Pattern Recognition and Classification', fontsize=16)
    
    # Plot 1: Toxicity scores by dose group
    ax = axs[0, 0]
    
    toxicity_scores = pattern_results['toxicity_scores']
    
    # Group by dose
    dose_groups = toxicity_scores.groupby('ARMCD')
    
    # Calculate mean scores for each organ system by dose
    mean_scores = dose_groups[['Liver_score', 'Kidney_score', 'Hematological_score', 'Cardiovascular_score']].mean()
    
    # Create grouped bar chart
    mean_scores.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Dose Group')
    ax.set_ylabel('Mean Toxicity Score')
    ax.set_title('Organ System Toxicity Scores by Dose Group')
    ax.legend(title='Organ System')
    
    # Plot 2: Pattern prevalence by dose
    ax = axs[0, 1]
    
    pattern_prevalence = pattern_results['pattern_prevalence']
    
    # Create dataframe for plotting
    prevalence_df = pd.DataFrame(pattern_prevalence)
    
    # Create grouped bar chart
    prevalence_df.plot(kind='bar', ax=ax)
    
    ax.set_xlabel('Dose Group')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Toxicity Pattern Prevalence by Dose Group')
    ax.legend(title='Toxicity Pattern')
    
    # Plot 3: Dimensionality reduction of feature matrix
    ax = axs[1, 0]
    
    # Get feature matrix
    subject_df = features['subject_features']
    
    # Select numeric columns
    numeric_cols = subject_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['EXDOSE', 'SEX']]
    
    if len(numeric_cols) >= 2:
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(subject_df[numeric_cols].fillna(0))
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_data = tsne.fit_transform(scaled_data)
        
        # Create dataframe with t-SNE results
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_data[:, 0],
            'TSNE2': tsne_data[:, 1],
            'ARMCD': subject_df['ARMCD'].values
        })
        
        # Add toxicity pattern information
        toxicity_patterns = pattern_results['toxicity_patterns']
        
        # Create a new column for the primary toxicity pattern
        pattern_labels = []
        
        for subj in subject_df.index:
            patterns = toxicity_patterns.get(subj, [])
            if 'hepatotoxicity' in patterns:
                pattern_labels.append('Hepatotoxicity')
            elif 'nephrotoxicity' in patterns:
                pattern_labels.append('Nephrotoxicity')
            elif 'hematologicaltoxicity' in patterns:
                pattern_labels.append('Hematotoxicity')
            elif 'cardiovasculartoxicity' in patterns:
                pattern_labels.append('Cardiotoxicity')
            else:
                pattern_labels.append('No toxicity')
        
        tsne_df['Pattern'] = pattern_labels
        
        # Plot t-SNE results colored by toxicity pattern
        for pattern, color in zip(['Hepatotoxicity', 'Nephrotoxicity', 'Hematotoxicity', 'Cardiotoxicity', 'No toxicity'],
                                 ['red', 'blue', 'green', 'purple', 'gray']):
            mask = tsne_df['Pattern'] == pattern
            if mask.any():
                ax.scatter(tsne_df.loc[mask, 'TSNE1'], tsne_df.loc[mask, 'TSNE2'], 
                          c=color, label=pattern, alpha=0.7)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of Toxicity Patterns')
        ax.legend(title='Primary Pattern')
    else:
        ax.text(0.5, 0.5, 'Insufficient numeric features for dimensionality reduction', 
               ha='center', va='center')
    
    # Plot 4: Pattern co-occurrence heatmap
    ax = axs[1, 1]
    
    # Create co-occurrence matrix
    patterns = ['hepatotoxicity', 'nephrotoxicity', 'hematologicaltoxicity', 'cardiovasculartoxicity']
    co_occurrence_matrix = np.zeros((len(patterns), len(patterns)))
    
    for i, pattern1 in enumerate(patterns):
        for j, pattern2 in enumerate(patterns):
            if i == j:
                # Diagonal elements show pattern prevalence
                prevalence = np.mean([prev.get(dose, 0) for dose, prev in 
                                     pattern_results['pattern_prevalence'].get(pattern1, {}).items()])
                co_occurrence_matrix[i, j] = prevalence
            else:
                # Off-diagonal elements show co-occurrence
                key = f"{pattern1}_{pattern2}" if i < j else f"{pattern2}_{pattern1}"
                co_occurrence = pattern_results['pattern_co_occurrence'].get(key, 0)
                co_occurrence_matrix[i, j] = co_occurrence
    
    # Create heatmap
    im = ax.imshow(co_occurrence_matrix, cmap='YlOrRd')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Co-occurrence / Prevalence (%)')
    
    # Set labels
    pattern_labels = ['Hepato', 'Nephro', 'Hemato', 'Cardio']
    ax.set_xticks(np.arange(len(pattern_labels)))
    ax.set_yticks(np.arange(len(pattern_labels)))
    ax.set_xticklabels(pattern_labels)
    ax.set_yticklabels(pattern_labels)
    
    # Add text annotations
    for i in range(len(pattern_labels)):
        for j in range(len(pattern_labels)):
            text = ax.text(j, i, f"{co_occurrence_matrix[i, j]:.1f}%",
                          ha="center", va="center", color="black" if co_occurrence_matrix[i, j] < 50 else "white")
    
    ax.set_title('Toxicity Pattern Co-occurrence Matrix')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('/home/ubuntu/noael_project/demo_code/use_case4_results.png')
    print("Visualization saved to /home/ubuntu/noael_project/demo_code/use_case4_results.png")
    
    return '/home/ubuntu/noael_project/demo_code/use_case4_results.png'

# Function to prepare TxGemma input
def prepare_txgemma_input(features, pattern_results):
    """
    Prepare input for TxGemma model
    """
    print("Preparing TxGemma input...")
    
    # Create a summary of the toxicity patterns
    pattern_summary = {
        'study_design': {
            'species': 'Rat',  # Assuming rat for demonstration
            'dose_groups': [str(dose) for dose in features['doses']]
        },
        'toxicity_patterns': {
            'identified_patterns': [],
            'pattern_prevalence': {},
            'pattern_co_occurrence': {},
            'dose_response': {}
        }
    }
    
    # Add identified patterns
    for pattern, prevalence in pattern_results['pattern_prevalence'].items():
        # Clean up pattern name
        clean_pattern = pattern.replace('toxicity', ' toxicity')
        
        # Add to identified patterns
        pattern_summary['toxicity_patterns']['identified_patterns'].append(clean_pattern)
        
        # Add prevalence by dose
        pattern_summary['toxicity_patterns']['pattern_prevalence'][clean_pattern] = {
            str(dose): round(prev, 1) for dose, prev in prevalence.items()
        }
        
        # Add dose-response relationship
        # Calculate slope of prevalence vs. dose
        doses = [float(dose) for dose in prevalence.keys() if dose != 0]  # Exclude control group
        prevalences = [prevalence[dose] for dose in prevalence.keys() if dose != 0]
        
        if doses and prevalences:
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(doses, prevalences)
            
            pattern_summary['toxicity_patterns']['dose_response'][clean_pattern] = {
                'slope': round(slope, 3),
                'r_squared': round(r_value**2, 3),
                'p_value': round(p_value, 3)
            }
    
    # Add pattern co-occurrence
    for key, value in pattern_results['pattern_co_occurrence'].items():
        # Split key into two patterns
        pattern1, pattern2 = key.split('_')
        
        # Clean up pattern names
        clean_pattern1 = pattern1.replace('toxicity', ' toxicity')
        clean_pattern2 = pattern2.replace('toxicity', ' toxicity')
        
        # Add to co-occurrence dictionary
        pattern_summary['toxicity_patterns']['pattern_co_occurrence'][f"{clean_pattern1} + {clean_pattern2}"] = round(value, 1)
    
    # Add key biomarkers and findings for each pattern
    pattern_summary['toxicity_patterns']['key_indicators'] = {
        'hepato toxicity': {
            'biomarkers': ['ALT', 'AST', 'ALP', 'BILI', 'ALB'],
            'histopathology': ['Hepatocellular necrosis', 'Hepatocellular hypertrophy', 'Inflammation'],
            'organ_weights': ['Liver weight increase']
        },
        'nephro toxicity': {
            'biomarkers': ['BUN', 'CREAT', 'K', 'NA', 'CL'],
            'histopathology': ['Tubular necrosis', 'Tubular degeneration', 'Inflammation'],
            'organ_weights': ['Kidney weight increase']
        },
        'hematological toxicity': {
            'biomarkers': ['WBC', 'RBC', 'HGB', 'HCT', 'PLT'],
            'histopathology': ['Bone marrow hypocellularity', 'Decreased erythropoiesis', 'Decreased myelopoiesis'],
            'organ_weights': ['Spleen weight decrease']
        },
        'cardio toxicity': {
            'biomarkers': ['CK', 'LDH', 'TROP'],
            'histopathology': ['Myocardial degeneration', 'Myocardial necrosis', 'Inflammation'],
            'organ_weights': ['Heart weight increase']
        }
    }
    
    # Convert to JSON string
    pattern_summary_json = json.dumps(pattern_summary, indent=2)
    
    # Create prompt for TxGemma
    prompt = f"""
You are a toxicology expert analyzing data from a preclinical safety study to recognize and classify patterns of adverse effects.

Below is a summary of the identified toxicity patterns:

{pattern_summary_json}

Based on this data, please:
1. Classify each identified toxicity pattern according to severity (mild, moderate, severe) and mechanism
2. Analyze the relationships between different toxicity patterns, including potential common mechanisms
3. Identify the most sensitive and specific biomarkers for each toxicity pattern
4. Determine which patterns are most relevant for human risk assessment
5. Suggest additional analyses or endpoints that could further characterize these patterns

Your response should be structured as a JSON object with the following fields:
- pattern_classifications: object with pattern names as keys and classification details as values
- pattern_relationships: analysis of relationships between patterns
- key_biomarkers: most sensitive and specific biomarkers for each pattern
- human_relevance: assessment of relevance for human risk assessment
- additional_analyses: suggestions for additional analyses
"""
    
    return prompt

# Function to simulate TxGemma response
def simulate_txgemma_response(prompt):
    """
    Simulate TxGemma response for demonstration purposes
    """
    print("Simulating TxGemma response...")
    
    # Extract pattern summary from prompt
    try:
        start_idx = prompt.find('{')
        end_idx = prompt.find('\n\nBased on this data')
        pattern_summary_json = prompt[start_idx:end_idx]
        pattern_summary = json.loads(pattern_summary_json)
    except:
        print("Error parsing pattern summary from prompt")
        pattern_summary = {}
    
    # Extract toxicity patterns
    toxicity_patterns = pattern_summary.get('toxicity_patterns', {})
    identified_patterns = toxicity_patterns.get('identified_patterns', [])
    pattern_prevalence = toxicity_patterns.get('pattern_prevalence', {})
    pattern_co_occurrence = toxicity_patterns.get('pattern_co_occurrence', {})
    dose_response = toxicity_patterns.get('dose_response', {})
    key_indicators = toxicity_patterns.get('key_indicators', {})
    
    # Classify patterns
    pattern_classifications = {}
    
    for pattern in identified_patterns:
        # Get prevalence at highest dose
        prevalence_by_dose = pattern_prevalence.get(pattern, {})
        max_prevalence = max(prevalence_by_dose.values()) if prevalence_by_dose else 0
        
        # Get dose-response relationship
        dr_info = dose_response.get(pattern, {})
        slope = dr_info.get('slope', 0)
        
        # Determine severity based on prevalence and slope
        if max_prevalence > 75 and slope > 0.3:
            severity = "severe"
        elif max_prevalence > 50 or slope > 0.2:
            severity = "moderate"
        else:
            severity = "mild"
        
        # Determine mechanism based on pattern
        if "hepato" in pattern.lower():
            mechanism = "Direct hepatocellular injury with enzyme leakage and potential cholestasis"
        elif "nephro" in pattern.lower():
            mechanism = "Tubular epithelial cell damage with impaired filtration and electrolyte imbalance"
        elif "hemato" in pattern.lower():
            mechanism = "Bone marrow suppression affecting multiple hematopoietic lineages"
        elif "cardio" in pattern.lower():
            mechanism = "Myocardial cell damage with enzyme leakage and potential functional impairment"
        else:
            mechanism = "Unknown"
        
        # Add classification
        pattern_classifications[pattern] = {
            "severity": severity,
            "mechanism": mechanism,
            "max_prevalence": max_prevalence,
            "dose_response_slope": slope
        }
    
    # Analyze pattern relationships
    pattern_relationships = {
        "co_occurrence_analysis": {},
        "common_mechanisms": [],
        "sequential_relationships": []
    }
    
    # Analyze co-occurrence
    high_co_occurrence_pairs = []
    
    for pair, value in pattern_co_occurrence.items():
        if value > 50:
            pattern_relationships["co_occurrence_analysis"][pair] = {
                "co_occurrence_percentage": value,
                "interpretation": "Strong co-occurrence suggesting shared mechanism or secondary effects"
            }
            high_co_occurrence_pairs.append(pair)
    
    # Add common mechanisms based on co-occurrence
    if "hepato toxicity + nephro toxicity" in high_co_occurrence_pairs:
        pattern_relationships["common_mechanisms"].append({
            "patterns": ["hepato toxicity", "nephro toxicity"],
            "mechanism": "Oxidative stress affecting both liver and kidney"
        })
    
    if "hepato toxicity + hematological toxicity" in high_co_occurrence_pairs:
        pattern_relationships["common_mechanisms"].append({
            "patterns": ["hepato toxicity", "hematological toxicity"],
            "mechanism": "Impaired production of hematopoietic growth factors by damaged liver"
        })
    
    # Add sequential relationships
    pattern_relationships["sequential_relationships"] = [
        {
            "primary": "hepato toxicity",
            "secondary": "nephro toxicity",
            "relationship": "Liver damage may lead to altered metabolism of compounds, increasing kidney exposure"
        },
        {
            "primary": "cardio toxicity",
            "secondary": "hepato toxicity",
            "relationship": "Cardiac dysfunction may lead to hepatic congestion and secondary liver injury"
        }
    ]
    
    # Identify key biomarkers
    key_biomarkers = {}
    
    for pattern, indicators in key_indicators.items():
        biomarkers = indicators.get('biomarkers', [])
        
        if biomarkers:
            # Determine sensitivity and specificity (simulated)
            biomarker_metrics = {}
            
            for biomarker in biomarkers:
                # Simulate metrics based on biomarker
                if pattern == "hepato toxicity":
                    if biomarker == "ALT":
                        sensitivity = 0.95
                        specificity = 0.85
                    elif biomarker == "AST":
                        sensitivity = 0.90
                        specificity = 0.80
                    elif biomarker == "BILI":
                        sensitivity = 0.70
                        specificity = 0.95
                    else:
                        sensitivity = 0.75
                        specificity = 0.75
                elif pattern == "nephro toxicity":
                    if biomarker == "CREAT":
                        sensitivity = 0.85
                        specificity = 0.90
                    elif biomarker == "BUN":
                        sensitivity = 0.80
                        specificity = 0.85
                    else:
                        sensitivity = 0.70
                        specificity = 0.80
                elif pattern == "hematological toxicity":
                    if biomarker == "PLT":
                        sensitivity = 0.90
                        specificity = 0.85
                    elif biomarker == "WBC":
                        sensitivity = 0.85
                        specificity = 0.80
                    else:
                        sensitivity = 0.80
                        specificity = 0.75
                elif pattern == "cardio toxicity":
                    if biomarker == "TROP":
                        sensitivity = 0.95
                        specificity = 0.95
                    elif biomarker == "CK":
                        sensitivity = 0.85
                        specificity = 0.75
                    else:
                        sensitivity = 0.75
                        specificity = 0.70
                else:
                    sensitivity = 0.75
                    specificity = 0.75
                
                biomarker_metrics[biomarker] = {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "overall_utility": (sensitivity + specificity) / 2
                }
            
            # Sort by overall utility
            sorted_biomarkers = sorted(biomarker_metrics.items(), 
                                      key=lambda x: x[1]['overall_utility'], 
                                      reverse=True)
            
            # Add top 3 biomarkers
            key_biomarkers[pattern] = {
                "top_biomarkers": [b[0] for b in sorted_biomarkers[:3]],
                "metrics": {b[0]: b[1] for b in sorted_biomarkers[:3]}
            }
    
    # Assess human relevance
    human_relevance = {}
    
    for pattern in identified_patterns:
        if "hepato" in pattern.lower():
            relevance = "high"
            rationale = "Liver toxicity mechanisms are generally well-conserved across species, and hepatotoxicity is a common cause of drug attrition in clinical development"
            translational_biomarkers = ["ALT", "AST", "BILI", "ALP"]
        elif "nephro" in pattern.lower():
            relevance = "high"
            rationale = "Kidney toxicity mechanisms are generally well-conserved, though there are some species differences in transporter expression and metabolism"
            translational_biomarkers = ["Creatinine", "BUN", "Cystatin C", "KIM-1"]
        elif "hemato" in pattern.lower():
            relevance = "medium"
            rationale = "Hematological parameters are generally translatable, but humans may have different sensitivity and recovery capacity compared to rodents"
            translational_biomarkers = ["Complete blood count", "Reticulocyte count"]
        elif "cardio" in pattern.lower():
            relevance = "medium"
            rationale = "Cardiac toxicity can translate to humans, but there are significant differences in heart rate, contractility, and ion channel expression"
            translational_biomarkers = ["Troponin", "NT-proBNP", "ECG parameters"]
        else:
            relevance = "unknown"
            rationale = "Insufficient data to assess human relevance"
            translational_biomarkers = []
        
        human_relevance[pattern] = {
            "relevance": relevance,
            "rationale": rationale,
            "translational_biomarkers": translational_biomarkers
        }
    
    # Suggest additional analyses
    additional_analyses = [
        {
            "analysis": "Toxicogenomic profiling",
            "rationale": "Gene expression analysis could provide mechanistic insights and identify early molecular markers of toxicity",
            "implementation": "RNA-seq or microarray analysis of affected tissues"
        },
        {
            "analysis": "Metabolomic analysis",
            "rationale": "Metabolite profiling could identify altered biochemical pathways and potential biomarkers",
            "implementation": "LC-MS/MS analysis of plasma and urine samples"
        },
        {
            "analysis": "Immunohistochemistry",
            "rationale": "IHC could characterize cell-specific effects and identify mechanisms (apoptosis, necrosis, etc.)",
            "implementation": "Staining for cleaved caspase-3, Ki-67, and cell-type specific markers"
        },
        {
            "analysis": "Functional assays",
            "rationale": "Functional tests could assess physiological impact of observed changes",
            "implementation": "Bile flow measurement for hepatotoxicity, GFR for nephrotoxicity, echocardiography for cardiotoxicity"
        },
        {
            "analysis": "In vitro mechanistic studies",
            "rationale": "In vitro studies could confirm mechanisms and assess human relevance",
            "implementation": "Human cell lines or primary cells exposed to test compound"
        }
    ]
    
    # Create simulated response
    response = {
        "pattern_classifications": pattern_classifications,
        "pattern_relationships": pattern_relationships,
        "key_biomarkers": key_biomarkers,
        "human_relevance": human_relevance,
        "additional_analyses": additional_analyses
    }
    
    return json.dumps(response, indent=2)

# Main function
def main():
    """
    Main function to demonstrate TxGemma for adverse effect pattern recognition
    """
    print("Starting adverse effect pattern recognition demonstration using TxGemma...")
    
    # Step 1: Load SEND domains
    domains = load_send_domains(SEND_DATA_PATH)
    
    # Step 2: Extract features from multiple domains
    features = extract_multidomain_features(domains)
    
    # Step 3: Identify toxicity patterns
    pattern_results = identify_toxicity_patterns(features)
    
    # Print identified patterns
    print("\nIdentified toxicity patterns:")
    for subj, patterns in list(pattern_results['toxicity_patterns'].items())[:10]:  # Show first 10 subjects
        print(f"- {subj}: {', '.join(patterns) if patterns else 'No toxicity'}")
    
    # Print pattern prevalence
    print("\nPattern prevalence by dose:")
    for pattern, prevalence in pattern_results['pattern_prevalence'].items():
        print(f"- {pattern}: {prevalence}")
    
    # Step 4: Prepare input for TxGemma
    txgemma_prompt = prepare_txgemma_input(features, pattern_results)
    
    # Step 5: Get TxGemma response (simulated for demonstration)
    txgemma_response = simulate_txgemma_response(txgemma_prompt)
    print("\nTxGemma Response:")
    print(txgemma_response)
    
    # Step 6: Visualize toxicity patterns
    visualization_path = visualize_toxicity_patterns(features, pattern_results)
    
    # Step 7: Validate against known patterns (if available)
    if '_toxicity_patterns' in domains:
        known_patterns = domains['_toxicity_patterns']
        
        print("\nValidation against known patterns:")
        
        # Calculate accuracy metrics
        total_subjects = len(known_patterns)
        correct_predictions = 0
        
        for subj, known in list(known_patterns.items())[:10]:  # Show first 10 subjects
            predicted = pattern_results['toxicity_patterns'].get(subj, [])
            
            # Convert pattern names to match
            known_converted = []
            for pattern in known:
                if pattern == 'hepatotoxicity':
                    known_converted.append('hepatotoxicity')
                elif pattern == 'nephrotoxicity':
                    known_converted.append('nephrotoxicity')
                elif pattern == 'hematotoxicity':
                    known_converted.append('hematologicaltoxicity')
                elif pattern == 'cardiotoxicity':
                    known_converted.append('cardiovasculartoxicity')
            
            # Check if prediction matches known pattern
            is_correct = set(predicted) == set(known_converted)
            if is_correct:
                correct_predictions += 1
            
            print(f"- {subj}: Known={known}, Predicted={predicted}, Correct={is_correct}")
        
        # Calculate overall accuracy
        accuracy = correct_predictions / total_subjects * 100
        print(f"\nOverall accuracy: {accuracy:.1f}%")
    
    print("\nDemonstration completed successfully!")
    print(f"Results visualization saved to: {visualization_path}")
    
    return {
        'features': features,
        'pattern_results': pattern_results,
        'txgemma_response': json.loads(txgemma_response),
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    main()
