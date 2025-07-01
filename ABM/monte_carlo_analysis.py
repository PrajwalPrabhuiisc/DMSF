import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import logging
import os
from datetime import datetime
from glob import glob

# Configure logging
logging.basicConfig(filename='analysis_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_simulation_data(sim_files):
    """Aggregate data from individual simulation files."""
    model_dfs = []
    for file in sim_files:
        try:
            # Load Configuration sheet to get parameters
            config_df = pd.read_excel(file, sheet_name='Configuration')
            if config_df.empty:
                logging.warning(f"No configuration data in {file}")
                continue
            config = config_df.iloc[0]
            sim_id = config['simulation_id']
            
            # Load Model_Metrics for Step 150
            model_df = pd.read_excel(file, sheet_name='Model_Metrics')
            model_df = model_df[model_df['Step'] == 150].copy()
            if model_df.empty:
                logging.warning(f"No Step 150 data in {file}")
                continue
            
            # Add configuration details
            model_df['simulation_id'] = sim_id
            model_df['reporting_structure'] = config['reporting_structure']
            model_df['org_structure'] = config['org_structure']
            model_df['hazard_prob'] = config['hazard_prob']
            model_df['delay_prob'] = config['delay_prob']
            model_df['reporter_detection'] = config['reporter_detection']
            model_dfs.append(model_df)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            continue
    
    if not model_dfs:
        logging.error("No valid simulation data loaded")
        print("Error: No valid simulation data loaded")
        return None
    return pd.concat(model_dfs, ignore_index=True)

def check_anova_assumptions(df, metric, group_col):
    """Check normality and homogeneity of variance."""
    groups = df.groupby(group_col)
    normality_pvals = {}
    for name, group in groups:
        stat, pval = stats.shapiro(group[metric].dropna())
        normality_pvals[name] = pval
    levene_stat, levene_pval = stats.levene(*[group[metric].dropna() for _, group in groups])
    return normality_pvals, levene_stat, levene_pval

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

def analyze_results(sim_files, output_dir):
    """Perform statistical analysis for hypotheses H1–H5."""
    df = load_simulation_data(sim_files)
    if df is None:
        return
    
    # Ensure required columns
    required_columns = ['Average_SA', 'Safety_Incidents', 'Schedule_Adherence', 
                        'reporting_structure', 'org_structure', 'hazard_prob', 'delay_prob']
    if not all(col in df.columns for col in required_columns):
        logging.error("Missing required columns in aggregated data")
        print("Error: Missing required columns")
        return

    # Metrics to analyze
    metrics = ['Average_SA', 'Safety_Incidents', 'Schedule_Adherence']
    
    # Initialize results storage
    results = {
        'H1_SA_ANOVA': {}, 'H1_SA_Tukey': {}, 'H1_SA_Effect_Sizes': {},
        'H2_Safety_ANOVA': {}, 'H2_Safety_Tukey': {}, 'H2_Safety_Effect_Sizes': {},
        'H3_Schedule_ANOVA': {}, 'H3_Schedule_Tukey': {}, 'H3_Schedule_Effect_Sizes': {},
        'H4_TwoWay_ANOVA': {}, 'H5_ThreeWay_ANOVA': {}
    }

    # H1, H2, H3: One-way ANOVA for reporting_structure
    for metric, h_prefix in zip(metrics, ['H1_SA', 'H2_Safety', 'H3_Schedule']):
        normality_pvals, levene_stat, levene_pval = check_anova_assumptions(df, metric, 'reporting_structure')
        logging.debug(f"{metric} Normality p-values: {normality_pvals}")
        logging.debug(f"{metric} Levene's test: stat={levene_stat:.4f}, p-value={levene_pval:.4f}")
        print(f"{metric} Normality p-values: {normality_pvals}")
        print(f"{metric} Levene's test: stat={levene_stat:.4f}, p-value={levene_pval:.4f}")

        if all(p > 0.05 for p in normality_pvals.values()) and levene_pval > 0.05:
            groups = [group[metric].dropna() for _, group in df.groupby('reporting_structure')]
            anova_stat, anova_pval = stats.f_oneway(*groups)
            test_type = 'ANOVA'
        else:
            groups = [group[metric].dropna() for _, group in df.groupby('reporting_structure')]
            anova_stat, anova_pval = stats.kruskal(*groups)
            test_type = 'Kruskal-Wallis'
        results[f'{h_prefix}_ANOVA'] = {'stat': anova_stat, 'pval': anova_pval, 'test_type': test_type}
        logging.debug(f"{h_prefix} {test_type}: stat={anova_stat:.4f}, p-value={anova_pval:.4f}")
        print(f"{h_prefix} {test_type}: stat={anova_stat:.4f}, p-value={anova_pval:.4f}")

        tukey = pairwise_tukeyhsd(df[metric].dropna(), df['reporting_structure'].dropna())
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                columns=tukey._results_table.data[0])
        results[f'{h_prefix}_Tukey'] = tukey_df

        effect_sizes = {}
        report_structures = df['reporting_structure'].unique()
        for i, rs1 in enumerate(report_structures):
            for rs2 in report_structures[i+1:]:
                group1 = df[df['reporting_structure'] == rs1][metric].dropna()
                group2 = df[df['reporting_structure'] == rs2][metric].dropna()
                d = cohen_d(group1, group2)
                effect_sizes[f"{rs1} vs {rs2}"] = d
        results[f'{h_prefix}_Effect_Sizes'] = effect_sizes

    # H4: Two-way ANOVA
    model = ols('Average_SA ~ C(reporting_structure) + C(org_structure) + C(reporting_structure):C(org_structure)', 
                data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    results['H4_TwoWay_ANOVA'] = anova_table
    logging.debug(f"H4 Two-way ANOVA results:\n{anova_table}")
    print("H4 Two-way ANOVA results:\n", anova_table)

    # H5: Three-way ANOVA
    model = ols('Average_SA ~ C(reporting_structure) + C(org_structure) + C(hazard_prob) + '
                'C(reporting_structure):C(org_structure) + C(reporting_structure):C(hazard_prob) + '
                'C(org_structure):C(hazard_prob) + C(reporting_structure):C(org_structure):C(hazard_prob)', 
                data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    results['H5_ThreeWay_ANOVA'] = anova_table
    logging.debug(f"H5 Three-way ANOVA results:\n{anova_table}")
    print("H5 Three-way ANOVA results:\n", anova_table)

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='reporting_structure', y='Average_SA', hue='org_structure', data=df)
    plt.title('H1,H4: Average SA by Reporting and Organizational Structure (Step 150)')
    plt.xlabel('Reporting Structure')
    plt.ylabel('Average Situational Awareness')
    plt.savefig(os.path.join(output_dir, 'h1_h4_sa_boxplot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.pointplot(x='org_structure', y='Average_SA', hue='reporting_structure', data=df, dodge=True)
    plt.title('H4: Interaction Plot - SA by Reporting and Organizational Structure')
    plt.xlabel('Organizational Structure')
    plt.ylabel('Average Situational Awareness')
    plt.savefig(os.path.join(output_dir, 'h4_interaction_plot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.pointplot(x='hazard_prob', y='Average_SA', hue='reporting_structure', data=df, dodge=True)
    plt.title('H5: Interaction Plot - SA by Reporting Structure and Hazard Probability')
    plt.xlabel('Hazard Probability')
    plt.ylabel('Average Situational Awareness')
    plt.savefig(os.path.join(output_dir, 'h5_hazard_interaction_plot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='reporting_structure', y='Safety_Incidents', data=df)
    plt.title('H2: Safety Incidents by Reporting Structure (Step 150)')
    plt.xlabel('Reporting Structure')
    plt.ylabel('Safety Incidents')
    plt.savefig(os.path.join(output_dir, 'h2_safety_boxplot.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='reporting_structure', y='Schedule_Adherence', data=df)
    plt.title('H3: Schedule Adherence by Reporting Structure (Step 150)')
    plt.xlabel('Reporting Structure')
    plt.ylabel('Schedule Adherence (%)')
    plt.savefig(os.path.join(output_dir, 'h3_schedule_boxplot.png'))
    plt.close()

    # Save results
    output_file = os.path.join(output_dir, f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for key, value in results.items():
            if 'ANOVA' in key:
                value.to_excel(writer, sheet_name=key, index=True)
            elif 'Tukey' in key:
                value.to_excel(writer, sheet_name=key, index=False)
            elif 'Effect_Sizes' in key:
                pd.DataFrame([value]).to_excel(writer, sheet_name=key, index=False)
    print(f"Analysis results saved to {output_file}")
    logging.debug(f"Analysis results saved to {output_file}")

def main():
    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all simulation files
    sim_files = glob('simulation_outputs/construction_simulation_*.xlsx')
    if len(sim_files) != 24300:
        print(f"Warning: Found {len(sim_files)} files, expected 24300")
        logging.warning(f"Found {len(sim_files)} files, expected 24300")
    
    analyze_results(sim_files, output_dir)

if __name__ == "__main__":
    main()
