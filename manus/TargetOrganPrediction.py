"""
Use Case 2: Prediction of Target Organ Toxicity from Early Study Data

This script demonstrates how to use TxGemma to predict potential target organ toxicity
based on early indicators in toxicology studies from SEND datasets.
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Path to SEND dataset
SEND_DATA_PATH = "/home/ubuntu/noael_project/sample_datasets/phuse-scripts/data/send/CBER-POC-Pilot-Study1-Vaccine"

# Function to load and process SEND datasets
def load_send_domains(base_path, interim=False):
    """
    Load relevant SEND domains from XPT files
    
    Parameters:
    -----------
    base_path : str
        Path to SEND dataset
    interim : bool
        If True, load only interim data (e.g., day 7 or 14)
        If False, load terminal data (e.g., day 28)
    """
    print(f"Loading SEND domains ({'interim' if interim else 'terminal'} data)...")
    
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
        return create_mock_send_data(interim)

# Function to create mock SEND data for demonstration
def create_mock_send_data(interim=False):
    """
    Create mock SEND data for demonstration purposes
    
    Parameters:
    -----------
    interim : bool
        If True, create interim data (e.g., day 7 or 14)
        If False, create terminal data (e.g., day 28)
    """
    domains = {}
    
    # Set study day based on interim flag
    study_day = 7 if interim else 28
    
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
    
    # Define biomarkers that are early indicators of organ toxicity
    liver_biomarkers = ['ALT', 'AST', 'ALP', 'BILI', 'ALB']
    kidney_biomarkers = ['BUN', 'CREAT', 'K', 'NA', 'CL']
    hematology_biomarkers = ['WBC', 'RBC', 'HGB', 'HCT', 'PLT']
    
    # Baseline values for each biomarker
    baseline_values = {
        # Liver biomarkers
        'ALT': 45, 'AST': 80, 'ALP': 120, 'BILI': 0.3, 'ALB': 4.0,
        # Kidney biomarkers
        'BUN': 15, 'CREAT': 0.6, 'K': 4.5, 'NA': 140, 'CL': 105,
        # Hematology biomarkers
        'WBC': 10, 'RBC': 8, 'HGB': 15, 'HCT': 45, 'PLT': 800
    }
    
    # Define effect multipliers for each dose group (control, low, mid, high)
    # For interim data, effects are less pronounced
    if interim:
        effect_multipliers = {
            # Liver biomarkers - early signs of hepatotoxicity at high dose
            'ALT': [1.0, 1.0, 1.1, 1.3],
            'AST': [1.0, 1.0, 1.1, 1.2],
            'ALP': [1.0, 1.0, 1.05, 1.15],
            'BILI': [1.0, 1.0, 1.0, 1.1],
            'ALB': [1.0, 1.0, 0.95, 0.9],
            
            # Kidney biomarkers - subtle early signs at high dose
            'BUN': [1.0, 1.0, 1.0, 1.1],
            'CREAT': [1.0, 1.0, 1.0, 1.05],
            'K': [1.0, 1.0, 1.0, 1.05],
            'NA': [1.0, 1.0, 1.0, 0.98],
            'CL': [1.0, 1.0, 1.0, 0.98],
            
            # Hematology biomarkers - early signs of bone marrow toxicity
            'WBC': [1.0, 1.0, 0.95, 0.9],
            'RBC': [1.0, 1.0, 0.98, 0.95],
            'HGB': [1.0, 1.0, 0.98, 0.95],
            'HCT': [1.0, 1.0, 0.98, 0.95],
            'PLT': [1.0, 0.98, 0.95, 0.9]
        }
    else:
        # Terminal data shows more pronounced effects
        effect_multipliers = {
            # Liver biomarkers - clear hepatotoxicity at high dose
            'ALT': [1.0, 1.1, 1.3, 2.0],
            'AST': [1.0, 1.1, 1.2, 1.8],
            'ALP': [1.0, 1.05, 1.2, 1.5],
            'BILI': [1.0, 1.0, 1.1, 1.3],
            'ALB': [1.0, 0.98, 0.9, 0.8],
            
            # Kidney biomarkers - clear nephrotoxicity at high dose
            'BUN': [1.0, 1.0, 1.1, 1.4],
            'CREAT': [1.0, 1.0, 1.1, 1.3],
            'K': [1.0, 1.0, 1.05, 1.2],
            'NA': [1.0, 1.0, 0.98, 0.95],
            'CL': [1.0, 1.0, 0.98, 0.95],
            
            # Hematology biomarkers - clear bone marrow toxicity
            'WBC': [1.0, 0.98, 0.9, 0.7],
            'RBC': [1.0, 0.98, 0.95, 0.85],
            'HGB': [1.0, 0.98, 0.95, 0.85],
            'HCT': [1.0, 0.98, 0.95, 0.85],
            'PLT': [1.0, 0.95, 0.85, 0.7]
        }
    
    # Generate laboratory data
    all_biomarkers = liver_biomarkers + kidney_biomarkers + hematology_biomarkers
    
    for subj in domains['dm']['USUBJID']:
        arm_idx = {'C': 0, 'LD': 1, 'MD': 2, 'HD': 3}
        arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
        
        for biomarker in all_biomarkers:
            # Add biological variability
            base_value = baseline_values[biomarker]
            multiplier = effect_multipliers[biomarker][arm_idx[arm]]
            
            # Add random variation (10% CV)
            value = base_value * multiplier * np.random.normal(1, 0.1)
            
            # Determine units based on biomarker
            if biomarker in ['ALT', 'AST', 'ALP']:
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
            else:
                units = ''
            
            lb_data.append({
                'USUBJID': subj,
                'LBTEST': biomarker,
                'LBSTRESN': value,
                'LBSTRESU': units,
                'VISITDY': study_day
            })
    
    domains['lb'] = pd.DataFrame(lb_data)
    
    # Create mock histopathology findings (MI) domain for terminal data only
    if not interim:
        mi_data = []
        
        # Define histopathology findings for each organ
        liver_findings = ['Hepatocellular necrosis', 'Hepatocellular hypertrophy', 'Inflammation']
        kidney_findings = ['Tubular necrosis', 'Tubular degeneration', 'Inflammation']
        bone_marrow_findings = ['Hypocellularity', 'Decreased erythropoiesis', 'Decreased myelopoiesis']
        
        # Define severity grades
        severity_grades = {
            'C': {'grade': 0, 'term': 'Normal'},
            'LD': {'grade': 0, 'term': 'Normal'},
            'MD': {'grade': 1, 'term': 'Minimal'},
            'HD': {'grade': 2, 'term': 'Mild'}
        }
        
        # Generate histopathology data
        for subj in domains['dm']['USUBJID']:
            arm = domains['dm'].loc[domains['dm']['USUBJID'] == subj, 'ARMCD'].values[0]
            
            # Add liver findings for mid and high dose
            if arm in ['MD', 'HD']:
                for finding in liver_findings:
                    # Add some variability - not all animals show all findings
                    if np.random.random() < (0.5 if arm == 'MD' else 0.8):
                        mi_data.append({
                            'USUBJID': subj,
                            'MISPEC': 'LIVER',
                            'MISTRESC': finding,
                            'MISEV': severity_grades[arm]['grade'],
                            'MISEVTXT': severity_grades[arm]['term']
                        })
            
            # Add kidney findings for high dose only
            if arm == 'HD':
                for finding in kidney_findings:
                    # Add some variability
                    if np.random.random() < 0.7:
                        mi_data.append({
                            'USUBJID': subj,
                            'MISPEC': 'KIDNEY',
                            'MISTRESC': finding,
                            'MISEV': severity_grades[arm]['grade'],
                            'MISEVTXT': severity_grades[arm]['term']
                        })
            
            # Add bone marrow findings for high dose only
            if arm == 'HD':
                for finding in bone_marrow_findings:
                    # Add some variability
                    if np.random.random() < 0.6:
                        mi_data.append({
                            'USUBJID': subj,
                            'MISPEC': 'BONE MARROW',
                            'MISTRESC': finding,
                            'MISEV': severity_grades[arm]['grade'],
                            'MISEVTXT': severity_grades[arm]['term']
                        })
        
        domains['mi'] = pd.DataFrame(mi_data)
    
    print("Created mock SEND data with the following domains:")
    for domain, df in domains.items():
        print(f"- {domain}: {len(df)} records")
    
    return domains

# Function to extract biomarker features from laboratory data
def extract_biomarker_features(domains):
    """
    Extract biomarker features from laboratory data
    """
    print("Extracting biomarker features...")
    
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
                            'percent_change': ((np.mean(values) / test_stats.get(min(features['doses']), {}).get('mean', np.mean(values))) - 1) * 100 if dose != min(features['doses']) else 0
                        }
                
                lb_features[test] = test_stats
            
            features['laboratory_tests'] = lb_features
            print(f"Extracted data for {len(lb_features)} laboratory tests")
    
    # Extract histopathology findings if available
    if 'mi' in domains:
        mi_data = domains['mi']
        
        if 'MISPEC' in mi_data.columns and 'MISTRESC' in mi_data.columns:
            mi_features = {}
            
            # Group by organ and finding
            for organ in mi_data['MISPEC'].unique():
                organ_data = mi_data[mi_data['MISPEC'] == organ]
                
                # Count findings for each dose group
                organ_stats = {}
                for dose, subjects in features['dose_groups'].items():
                    dose_data = organ_data[organ_data['USUBJID'].isin(subjects)]
                    
                    if not dose_data.empty:
                        # Count unique findings
                        findings = dose_data['MISTRESC'].value_counts().to_dict()
                        
                        # Calculate incidence (% of animals affected)
                        incidence = len(dose_data['USUBJID'].unique()) / len(subjects) * 100
                        
                        organ_stats[dose] = {
                            'findings': findings,
                            'incidence': incidence,
                            'severity': dose_data['MISEV'].mean() if 'MISEV' in dose_data.columns else 0
                        }
                    else:
                        organ_stats[dose] = {
                            'findings': {},
                            'incidence': 0,
                            'severity': 0
                        }
                
                mi_features[organ] = organ_stats
            
            features['histopathology'] = mi_features
            print(f"Extracted histopathology data for {len(mi_features)} organs")
    
    return features

# Function to create biomarker profiles for each subject
def create_biomarker_profiles(domains):
    """
    Create biomarker profiles for each subject
    """
    print("Creating biomarker profiles...")
    
    profiles = []
    
    if 'dm' in domains and 'lb' in domains:
        lb_data = domains['lb']
        
        # Get unique subjects and biomarkers
        subjects = domains['dm']['USUBJID'].unique()
        biomarkers = lb_data['LBTEST'].unique()
        
        # Create profile for each subject
        for subj in subjects:
            profile = {'USUBJID': subj}
            
            # Get subject metadata
            subj_data = domains['dm'][domains['dm']['USUBJID'] == subj]
            profile['ARMCD'] = subj_data['ARMCD'].values[0] if not subj_data.empty else None
            profile['SEX'] = subj_data['SEX'].values[0] if not subj_data.empty else None
            
            # Get dose information
            if 'ex' in domains:
                ex_data = domains['ex'][domains['ex']['USUBJID'] == subj]
                profile['EXDOSE'] = ex_data['EXDOSE'].values[0] if not ex_data.empty and 'EXDOSE' in ex_data.columns else None
            
            # Get biomarker values
            for biomarker in biomarkers:
                biomarker_data = lb_data[(lb_data['USUBJID'] == subj) & (lb_data['LBTEST'] == biomarker)]
                if not biomarker_data.empty and 'LBSTRESN' in biomarker_data.columns:
                    profile[biomarker] = biomarker_data['LBSTRESN'].values[0]
                else:
                    profile[biomarker] = None
            
            profiles.append(profile)
    
    return pd.DataFrame(profiles)

# Function to analyze biomarker patterns
def analyze_biomarker_patterns(biomarker_profiles):
    """
    Analyze biomarker patterns to identify potential target organ toxicity
    """
    print("Analyzing biomarker patterns...")
    
    results = {}
    
    # Group biomarkers by organ system
    organ_systems = {
        'Liver': ['ALT', 'AST', 'ALP', 'BILI', 'ALB'],
        'Kidney': ['BUN', 'CREAT', 'K', 'NA', 'CL'],
        'Hematology': ['WBC', 'RBC', 'HGB', 'HCT', 'PLT']
    }
    
    # Calculate organ-specific scores for each subject
    organ_scores = {}
    
    for organ, biomarkers in organ_systems.items():
        # Create subset of profiles with only relevant biomarkers
        organ_data = biomarker_profiles[['USUBJID', 'ARMCD', 'EXDOSE'] + [b for b in biomarkers if b in biomarker_profiles.columns]]
        
        # Drop rows with missing values
        organ_data = organ_data.dropna()
        
        if len(organ_data) > 0:
            # Standardize biomarker values
            scaler = StandardScaler()
            biomarker_cols = [b for b in biomarkers if b in organ_data.columns]
            
            if len(biomarker_cols) > 0:
                scaled_data = scaler.fit_transform(organ_data[biomarker_cols])
                
                # Calculate organ toxicity score (sum of absolute standardized values)
                # Higher score indicates more abnormal values
                scores = np.sum(np.abs(scaled_data), axis=1)
                
                # Add scores to dataframe
                organ_data[f'{organ}_score'] = scores
                
                # Group by dose and calculate mean score
                if 'EXDOSE' in organ_data.columns:
                    dose_scores = organ_data.groupby('EXDOSE')[f'{organ}_score'].mean().to_dict()
                    
                    # Calculate fold change relative to control
                    control_score = dose_scores.get(min(dose_scores.keys()), 1)
                    fold_changes = {dose: score / control_score for dose, score in dose_scores.items()}
                    
                    organ_scores[organ] = {
                        'scores': dose_scores,
                        'fold_changes': fold_changes
                    }
    
    results['organ_scores'] = organ_scores
    
    # Perform PCA on biomarker profiles to visualize patterns
    pca_results = {}
    
    for organ, biomarkers in organ_systems.items():
        # Create subset of profiles with only relevant biomarkers
        organ_data = biomarker_profiles[['USUBJID', 'ARMCD', 'EXDOSE'] + [b for b in biomarkers if b in biomarker_profiles.columns]]
        
        # Drop rows with missing values
        organ_data = organ_data.dropna()
        
        if len(organ_data) > 0:
            # Standardize biomarker values
            scaler = StandardScaler()
            biomarker_cols = [b for b in biomarkers if b in organ_data.columns]
            
            if len(biomarker_cols) >= 2:  # Need at least 2 features for PCA
                scaled_data = scaler.fit_transform(organ_data[biomarker_cols])
                
                # Perform PCA
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)
                
                # Create dataframe with PCA results
                pca_df = pd.DataFrame({
                    'USUBJID': organ_data['USUBJID'],
                    'ARMCD': organ_data['ARMCD'],
                    'EXDOSE': organ_data['EXDOSE'],
                    'PC1': pca_data[:, 0],
                    'PC2': pca_data[:, 1]
                })
                
                pca_results[organ] = {
                    'pca_df': pca_df,
                    'explained_variance': pca.explained_variance_ratio_,
                    'loadings': pd.DataFrame(
                        pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=biomarker_cols
                    )
                }
    
    results['pca_results'] = pca_results
    
    # Perform clustering to identify patterns
    cluster_results = {}
    
    for organ, biomarkers in organ_systems.items():
        # Create subset of profiles with only relevant biomarkers
        organ_data = biomarker_profiles[['USUBJID', 'ARMCD', 'EXDOSE'] + [b for b in biomarkers if b in biomarker_profiles.columns]]
        
        # Drop rows with missing values
        organ_data = organ_data.dropna()
        
        if len(organ_data) > 0:
            # Standardize biomarker values
            scaler = StandardScaler()
            biomarker_cols = [b for b in biomarkers if b in organ_data.columns]
            
            if len(biomarker_cols) > 0:
                scaled_data = scaler.fit_transform(organ_data[biomarker_cols])
                
                # Perform K-means clustering (assuming 2 clusters: normal and abnormal)
                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Add cluster labels to dataframe
                organ_data['cluster'] = clusters
                
                # Determine which cluster represents abnormal values
                # Assume the cluster with higher mean values for liver and kidney biomarkers is abnormal
                # For hematology, the cluster with lower mean values is abnormal (due to decreases in cell counts)
                cluster_means = {}
                for cluster in [0, 1]:
                    cluster_data = scaled_data[clusters == cluster]
                    cluster_means[cluster] = np.mean(cluster_data, axis=0)
                
                if organ in ['Liver', 'Kidney']:
                    # For liver and kidney, higher values typically indicate toxicity
                    abnormal_cluster = 0 if np.mean(cluster_means[0]) > np.mean(cluster_means[1]) else 1
                else:
                    # For hematology, lower values typically indicate toxicity
                    abnormal_cluster = 0 if np.mean(cluster_means[0]) < np.mean(cluster_means[1]) else 1
                
                # Calculate percentage of subjects in abnormal cluster by dose
                if 'EXDOSE' in organ_data.columns:
                    abnormal_pct = organ_data.groupby('EXDOSE').apply(
                        lambda x: (x['cluster'] == abnormal_cluster).mean() * 100
                    ).to_dict()
                    
                    cluster_results[organ] = {
                        'abnormal_cluster': abnormal_cluster,
                        'abnormal_percentage': abnormal_pct
                    }
    
    results['cluster_results'] = cluster_results
    
    return results

# Function to predict target organ toxicity
def predict_target_organ_toxicity(biomarker_analysis, histopathology=None):
    """
    Predict target organ toxicity based on biomarker analysis
    
    Parameters:
    -----------
    biomarker_analysis : dict
        Results of biomarker pattern analysis
    histopathology : dict, optional
        Histopathology findings (for validation in terminal dataset)
    """
    print("Predicting target organ toxicity...")
    
    predictions = {}
    
    # Define thresholds for toxicity prediction
    score_threshold = 1.5  # Fold change threshold for organ scores
    cluster_threshold = 30  # Percentage threshold for abnormal cluster
    
    # Predict toxicity for each organ system
    organ_systems = ['Liver', 'Kidney', 'Hematology']
    
    for organ in organ_systems:
        # Initialize prediction
        predictions[organ] = {
            'toxicity_predicted': False,
            'confidence': 'Low',
            'affected_doses': [],
            'evidence': []
        }
        
        # Check organ scores
        if 'organ_scores' in biomarker_analysis and organ in biomarker_analysis['organ_scores']:
            organ_score = biomarker_analysis['organ_scores'][organ]
            
            # Check fold changes
            for dose, fold_change in organ_score['fold_changes'].items():
                if fold_change >= score_threshold:
                    predictions[organ]['toxicity_predicted'] = True
                    predictions[organ]['affected_doses'].append(dose)
                    predictions[organ]['evidence'].append(f"Organ score fold change: {fold_change:.2f} at dose {dose}")
        
        # Check cluster results
        if 'cluster_results' in biomarker_analysis and organ in biomarker_analysis['cluster_results']:
            cluster_result = biomarker_analysis['cluster_results'][organ]
            
            # Check percentage of subjects in abnormal cluster
            for dose, pct in cluster_result['abnormal_percentage'].items():
                if pct >= cluster_threshold:
                    predictions[organ]['toxicity_predicted'] = True
                    if dose not in predictions[organ]['affected_doses']:
                        predictions[organ]['affected_doses'].append(dose)
                    predictions[organ]['evidence'].append(f"Abnormal cluster: {pct:.1f}% at dose {dose}")
        
        # Determine confidence level
        if predictions[organ]['toxicity_predicted']:
            # Higher confidence if multiple lines of evidence
            if len(predictions[organ]['evidence']) >= 3:
                predictions[organ]['confidence'] = 'High'
            elif len(predictions[organ]['evidence']) == 2:
                predictions[organ]['confidence'] = 'Medium'
            else:
                predictions[organ]['confidence'] = 'Low'
    
    # Validate predictions against histopathology if available
    if histopathology:
        for organ in organ_systems:
            # Map organ system to histopathology organ names
            if organ == 'Liver':
                histo_organ = 'LIVER'
            elif organ == 'Kidney':
                histo_organ = 'KIDNEY'
            elif organ == 'Hematology':
                histo_organ = 'BONE MARROW'
            else:
                continue
            
            # Check if histopathology findings exist for this organ
            if histo_organ in histopathology:
                organ_histo = histopathology[histo_organ]
                
                # Check incidence at each dose
                for dose, stats in organ_histo.items():
                    if stats['incidence'] > 0:
                        # Add validation to prediction
                        if predictions[organ]['toxicity_predicted'] and dose in predictions[organ]['affected_doses']:
                            predictions[organ]['validation'] = 'True Positive'
                            predictions[organ]['validation_details'] = f"Histopathology confirms {organ.lower()} toxicity at dose {dose}"
                        elif predictions[organ]['toxicity_predicted'] and dose not in predictions[organ]['affected_doses']:
                            predictions[organ]['validation'] = 'False Positive'
                            predictions[organ]['validation_details'] = f"Histopathology shows {organ.lower()} toxicity at dose {dose}, but not predicted"
                        elif not predictions[organ]['toxicity_predicted']:
                            predictions[organ]['validation'] = 'False Negative'
                            predictions[organ]['validation_details'] = f"Histopathology shows {organ.lower()} toxicity at dose {dose}, but not predicted"
                
                # If no findings for any dose, and no toxicity predicted, it's a true negative
                if all(stats['incidence'] == 0 for stats in organ_histo.values()) and not predictions[organ]['toxicity_predicted']:
                    predictions[organ]['validation'] = 'True Negative'
                    predictions[organ]['validation_details'] = f"No histopathology findings for {organ.lower()}, correctly predicted"
    
    return predictions

# Function to prepare TxGemma input
def prepare_txgemma_input(biomarker_profiles, biomarker_analysis, toxicity_predictions):
    """
    Prepare input for TxGemma model
    """
    print("Preparing TxGemma input...")
    
    # Create a summary of the biomarker data and analysis
    study_summary = {
        'study_design': {
            'species': 'Rat',  # Assuming rat for demonstration
            'timepoint': 'Interim (Day 7)',  # Assuming interim data
            'biomarkers_evaluated': list(biomarker_profiles.columns[3:])  # Skip USUBJID, ARMCD, EXDOSE
        },
        'biomarker_analysis': {
            'organ_systems': {}
        },
        'preliminary_predictions': {}
    }
    
    # Add organ system scores
    if 'organ_scores' in biomarker_analysis:
        for organ, scores in biomarker_analysis['organ_scores'].items():
            study_summary['biomarker_analysis']['organ_systems'][organ] = {
                'fold_changes': {str(dose): round(fc, 2) for dose, fc in scores['fold_changes'].items()}
            }
    
    # Add cluster analysis results
    if 'cluster_results' in biomarker_analysis:
        for organ, clusters in biomarker_analysis['cluster_results'].items():
            if organ in study_summary['biomarker_analysis']['organ_systems']:
                study_summary['biomarker_analysis']['organ_systems'][organ]['abnormal_percentage'] = {
                    str(dose): round(pct, 1) for dose, pct in clusters['abnormal_percentage'].items()
                }
    
    # Add preliminary predictions
    for organ, prediction in toxicity_predictions.items():
        study_summary['preliminary_predictions'][organ] = {
            'toxicity_predicted': prediction['toxicity_predicted'],
            'confidence': prediction['confidence'],
            'affected_doses': [str(dose) for dose in prediction['affected_doses']],
            'evidence': prediction['evidence']
        }
    
    # Convert to JSON string
    study_summary_json = json.dumps(study_summary, indent=2)
    
    # Create prompt for TxGemma
    prompt = f"""
You are a toxicology expert analyzing interim biomarker data from a preclinical safety study to predict potential target organ toxicity that might develop by study termination.

Below is a summary of the interim biomarker data and preliminary analysis:

{study_summary_json}

Based on this interim data, please:
1. Predict which organs are likely to show histopathological findings at study termination
2. Estimate the lowest dose at which histopathological findings might be observed for each organ
3. Explain your reasoning, including which biomarker patterns are most indicative of developing toxicity
4. Assess your confidence in each prediction (high, medium, or low)
5. Suggest additional interim measurements that could strengthen the predictions

Your response should be structured as a JSON object with the following fields:
- predicted_target_organs: list of organs predicted to show histopathology at termination
- dose_predictions: object with organ names as keys and lowest affected doses as values
- reasoning: explanation of your predictions for each organ
- confidence: object with organ names as keys and confidence levels as values
- additional_measurements: suggestions for additional interim measurements
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
        end_idx = prompt.find('\n\nBased on this interim data')
        study_summary_json = prompt[start_idx:end_idx]
        study_summary = json.loads(study_summary_json)
    except:
        print("Error parsing study summary from prompt")
        study_summary = {}
    
    # Extract preliminary predictions
    preliminary_predictions = study_summary.get('preliminary_predictions', {})
    
    # Extract biomarker analysis
    biomarker_analysis = study_summary.get('biomarker_analysis', {}).get('organ_systems', {})
    
    # Generate predictions based on preliminary analysis
    predicted_target_organs = []
    dose_predictions = {}
    reasoning = {}
    confidence = {}
    
    for organ, prediction in preliminary_predictions.items():
        if prediction.get('toxicity_predicted', False):
            predicted_target_organs.append(organ)
            
            # Determine lowest affected dose
            affected_doses = [float(dose) for dose in prediction.get('affected_doses', [])]
            if affected_doses:
                lowest_dose = min(affected_doses)
                dose_predictions[organ] = str(lowest_dose)
            
            # Generate reasoning
            organ_analysis = biomarker_analysis.get(organ, {})
            fold_changes = organ_analysis.get('fold_changes', {})
            abnormal_pct = organ_analysis.get('abnormal_percentage', {})
            
            reasoning_text = f"For {organ}, "
            
            # Add fold change information
            if fold_changes:
                max_fold_change = max([float(fc) for fc in fold_changes.values()])
                max_dose = [dose for dose, fc in fold_changes.items() if float(fc) == max_fold_change][0]
                reasoning_text += f"biomarker patterns show a {max_fold_change:.2f}-fold change at dose {max_dose}, "
            
            # Add cluster information
            if abnormal_pct:
                max_pct = max([float(pct) for pct in abnormal_pct.values()])
                max_dose = [dose for dose, pct in abnormal_pct.items() if float(pct) == max_pct][0]
                reasoning_text += f"with {max_pct:.1f}% of subjects showing abnormal patterns at dose {max_dose}. "
            
            # Add specific biomarker information based on organ
            if organ == 'Liver':
                reasoning_text += "Elevated ALT and AST are early indicators of hepatocellular injury, which often precedes histopathological changes. The pattern of enzyme elevations suggests potential hepatocellular necrosis or hypertrophy may develop by study termination."
            elif organ == 'Kidney':
                reasoning_text += "Changes in BUN and creatinine, even subtle ones, are sensitive indicators of developing kidney injury. The observed pattern suggests potential tubular damage may be evident in histopathology at study termination."
            elif organ == 'Hematology':
                reasoning_text += "Decreases in multiple hematology parameters indicate potential bone marrow toxicity. These changes often precede histopathological evidence of bone marrow hypocellularity or decreased hematopoiesis."
            
            reasoning[organ] = reasoning_text
            
            # Set confidence based on preliminary prediction
            confidence[organ] = prediction.get('confidence', 'Medium')
    
    # Add additional measurements suggestions
    additional_measurements = [
        "Perform interim histopathology on a subset of animals to confirm developing lesions",
        "Add specialized biomarkers of cell injury such as HMGB1 or miRNAs for affected organs",
        "Include oxidative stress markers (e.g., GSH, MDA) to assess mechanism of toxicity",
        "Measure inflammatory cytokines to evaluate contribution of inflammation to observed changes",
        "Add functional tests for affected organs (e.g., bile acid for liver, urinalysis for kidney)"
    ]
    
    # Create simulated response
    response = {
        "predicted_target_organs": predicted_target_organs,
        "dose_predictions": dose_predictions,
        "reasoning": reasoning,
        "confidence": confidence,
        "additional_measurements": additional_measurements
    }
    
    return json.dumps(response, indent=2)

# Function to validate predictions with terminal data
def validate_predictions(toxicity_predictions, terminal_features):
    """
    Validate toxicity predictions with terminal histopathology data
    
    Parameters:
    -----------
    toxicity_predictions : dict
        Predictions of target organ toxicity
    terminal_features : dict
        Features extracted from terminal dataset
    """
    print("Validating predictions with terminal data...")
    
    validation_results = {}
    
    # Check if histopathology data is available
    if 'histopathology' in terminal_features:
        histopathology = terminal_features['histopathology']
        
        # Map organ systems to histopathology organ names
        organ_mapping = {
            'Liver': 'LIVER',
            'Kidney': 'KIDNEY',
            'Hematology': 'BONE MARROW'
        }
        
        # Validate each prediction
        for organ, prediction in toxicity_predictions.items():
            histo_organ = organ_mapping.get(organ)
            
            if histo_organ and histo_organ in histopathology:
                organ_histo = histopathology[histo_organ]
                
                # Initialize validation result
                validation_results[organ] = {
                    'prediction': prediction['toxicity_predicted'],
                    'histopathology_findings': False,
                    'affected_doses_predicted': prediction['affected_doses'],
                    'affected_doses_actual': []
                }
                
                # Check for findings at each dose
                for dose, stats in organ_histo.items():
                    if stats['incidence'] > 0:
                        validation_results[organ]['histopathology_findings'] = True
                        validation_results[organ]['affected_doses_actual'].append(dose)
                
                # Determine validation outcome
                if prediction['toxicity_predicted'] and validation_results[organ]['histopathology_findings']:
                    validation_results[organ]['outcome'] = 'True Positive'
                elif prediction['toxicity_predicted'] and not validation_results[organ]['histopathology_findings']:
                    validation_results[organ]['outcome'] = 'False Positive'
                elif not prediction['toxicity_predicted'] and validation_results[organ]['histopathology_findings']:
                    validation_results[organ]['outcome'] = 'False Negative'
                else:
                    validation_results[organ]['outcome'] = 'True Negative'
                
                # Calculate dose prediction accuracy
                if validation_results[organ]['outcome'] == 'True Positive':
                    # Convert to float for comparison
                    predicted_doses = [float(d) for d in prediction['affected_doses']]
                    actual_doses = [float(d) for d in validation_results[organ]['affected_doses_actual']]
                    
                    # Find lowest doses
                    lowest_predicted = min(predicted_doses) if predicted_doses else None
                    lowest_actual = min(actual_doses) if actual_doses else None
                    
                    if lowest_predicted and lowest_actual:
                        if lowest_predicted == lowest_actual:
                            validation_results[organ]['dose_prediction'] = 'Exact'
                        elif lowest_predicted < lowest_actual:
                            validation_results[organ]['dose_prediction'] = 'Conservative'
                        else:
                            validation_results[organ]['dose_prediction'] = 'Non-conservative'
    
    return validation_results

# Function to visualize results
def visualize_results(biomarker_profiles, biomarker_analysis, toxicity_predictions, txgemma_response, validation_results=None):
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
    fig.suptitle('Target Organ Toxicity Prediction from Early Study Data', fontsize=16)
    
    # Plot 1: Organ toxicity scores
    ax = axs[0, 0]
    
    if 'organ_scores' in biomarker_analysis:
        organ_scores = biomarker_analysis['organ_scores']
        
        # Extract fold changes for each organ
        organs = list(organ_scores.keys())
        doses = sorted([float(d) for d in list(organ_scores[organs[0]]['fold_changes'].keys())])
        
        # Create grouped bar chart
        bar_width = 0.25
        index = np.arange(len(doses))
        
        for i, organ in enumerate(organs):
            fold_changes = [organ_scores[organ]['fold_changes'][str(dose)] for dose in doses]
            ax.bar(index + i*bar_width, fold_changes, bar_width, label=organ)
        
        ax.set_xlabel('Dose')
        ax.set_ylabel('Fold Change in Organ Score')
        ax.set_title('Organ Toxicity Scores (Fold Change vs Control)')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels([str(d) for d in doses])
        ax.axhline(y=1.5, color='red', linestyle='--', label='Threshold')
        ax.legend()
    
    # Plot 2: PCA of biomarker profiles
    ax = axs[0, 1]
    
    if 'pca_results' in biomarker_analysis:
        pca_results = biomarker_analysis['pca_results']
        
        # Select one organ system for visualization
        if 'Liver' in pca_results:
            organ = 'Liver'
        elif 'Kidney' in pca_results:
            organ = 'Kidney'
        elif 'Hematology' in pca_results:
            organ = 'Hematology'
        else:
            organ = list(pca_results.keys())[0] if pca_results else None
        
        if organ:
            pca_df = pca_results[organ]['pca_df']
            
            # Create scatter plot colored by dose
            doses = sorted(pca_df['EXDOSE'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(doses)))
            
            for i, dose in enumerate(doses):
                dose_data = pca_df[pca_df['EXDOSE'] == dose]
                ax.scatter(dose_data['PC1'], dose_data['PC2'], color=colors[i], label=f'Dose {dose}')
            
            ax.set_xlabel(f"PC1 ({pca_results[organ]['explained_variance'][0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca_results[organ]['explained_variance'][1]:.1%})")
            ax.set_title(f'PCA of {organ} Biomarkers')
            ax.legend()
    
    # Plot 3: Abnormal cluster percentages
    ax = axs[1, 0]
    
    if 'cluster_results' in biomarker_analysis:
        cluster_results = biomarker_analysis['cluster_results']
        
        # Extract abnormal percentages for each organ
        organs = list(cluster_results.keys())
        doses = sorted([float(d) for d in list(cluster_results[organs[0]]['abnormal_percentage'].keys())])
        
        # Create grouped bar chart
        bar_width = 0.25
        index = np.arange(len(doses))
        
        for i, organ in enumerate(organs):
            abnormal_pct = [cluster_results[organ]['abnormal_percentage'][str(dose)] for dose in doses]
            ax.bar(index + i*bar_width, abnormal_pct, bar_width, label=organ)
        
        ax.set_xlabel('Dose')
        ax.set_ylabel('Percentage of Subjects in Abnormal Cluster')
        ax.set_title('Abnormal Biomarker Pattern Detection')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels([str(d) for d in doses])
        ax.axhline(y=30, color='red', linestyle='--', label='Threshold')
        ax.legend()
    
    # Plot 4: TxGemma predictions and validation
    ax = axs[1, 1]
    ax.axis('off')  # Turn off axis
    
    # Create text summary from TxGemma response and validation
    text = "TxGemma Predictions:\n\n"
    
    if txgemma_data:
        text += f"Predicted Target Organs: {', '.join(txgemma_data.get('predicted_target_organs', []))}\n\n"
        
        text += "Dose Predictions:\n"
        for organ, dose in txgemma_data.get('dose_predictions', {}).items():
            text += f"- {organ}: {dose}\n"
        
        text += "\nConfidence:\n"
        for organ, conf in txgemma_data.get('confidence', {}).items():
            text += f"- {organ}: {conf}\n"
        
        if validation_results:
            text += "\nValidation with Terminal Data:\n"
            for organ, result in validation_results.items():
                text += f"- {organ}: {result.get('outcome', 'Unknown')}"
                if 'dose_prediction' in result:
                    text += f", Dose prediction: {result['dose_prediction']}"
                text += "\n"
    
    ax.text(0, 1, text, va='top', ha='left', wrap=True, fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('/home/ubuntu/noael_project/demo_code/use_case2_results.png')
    print("Visualization saved to /home/ubuntu/noael_project/demo_code/use_case2_results.png")
    
    return '/home/ubuntu/noael_project/demo_code/use_case2_results.png'

# Main function
def main():
    """
    Main function to demonstrate TxGemma for predicting target organ toxicity
    """
    print("Starting target organ toxicity prediction demonstration using TxGemma...")
    
    # Step 1: Load interim SEND data
    interim_domains = load_send_domains(SEND_DATA_PATH, interim=True)
    
    # Step 2: Create biomarker profiles for each subject
    biomarker_profiles = create_biomarker_profiles(interim_domains)
    
    # Step 3: Analyze biomarker patterns
    biomarker_analysis = analyze_biomarker_patterns(biomarker_profiles)
    
    # Step 4: Predict target organ toxicity
    toxicity_predictions = predict_target_organ_toxicity(biomarker_analysis)
    print("\nPreliminary toxicity predictions:")
    for organ, prediction in toxicity_predictions.items():
        print(f"- {organ}: {'Toxicity predicted' if prediction['toxicity_predicted'] else 'No toxicity predicted'} ({prediction['confidence']} confidence)")
    
    # Step 5: Prepare input for TxGemma
    txgemma_prompt = prepare_txgemma_input(biomarker_profiles, biomarker_analysis, toxicity_predictions)
    
    # Step 6: Get TxGemma response (simulated for demonstration)
    txgemma_response = simulate_txgemma_response(txgemma_prompt)
    print("\nTxGemma Response:")
    print(txgemma_response)
    
    # Step 7: Load terminal data for validation (if available)
    terminal_domains = load_send_domains(SEND_DATA_PATH, interim=False)
    terminal_features = extract_biomarker_features(terminal_domains)
    
    # Step 8: Validate predictions with terminal data
    validation_results = validate_predictions(toxicity_predictions, terminal_features)
    print("\nValidation results:")
    for organ, result in validation_results.items():
        print(f"- {organ}: {result.get('outcome', 'Unknown')}")
    
    # Step 9: Visualize results
    visualization_path = visualize_results(biomarker_profiles, biomarker_analysis, toxicity_predictions, txgemma_response, validation_results)
    
    print("\nDemonstration completed successfully!")
    print(f"Results visualization saved to: {visualization_path}")
    
    return {
        'biomarker_profiles': biomarker_profiles,
        'biomarker_analysis': biomarker_analysis,
        'toxicity_predictions': toxicity_predictions,
        'txgemma_response': json.loads(txgemma_response),
        'validation_results': validation_results,
        'visualization_path': visualization_path
    }

if __name__ == "__main__":
    main()
