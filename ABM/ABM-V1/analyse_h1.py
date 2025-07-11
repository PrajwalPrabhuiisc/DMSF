import pandas as pd
import os
import glob
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
import numpy as np
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clean_step_value(step):
    """Convert Step value to integer, handling strings like '[100]' or other formats."""
    try:
        if isinstance(step, str):
            match = re.search(r'\d+', step)
            if match:
                return int(match.group())
            raise ValueError(f"Invalid step format: {step}")
        return int(step)
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not convert step value '{step}' to integer: {e}")
        return None

def clean_numeric_value(value):
    """Convert value to float, handling strings, lists, or NaN."""
    try:
        if isinstance(value, str):
            match = re.search(r'\d+\.?\d*', value)
            if match:
                return float(match.group())
            raise ValueError(f"Invalid numeric format: {value}")
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def load_simulation_data(directory="simulation_outputs"):
    """Load Model_Metrics and Agent_SA data for org_structure='functional'."""
    all_metrics = []
    agent_decisions = []
    excel_files = glob.glob(os.path.join(directory, "*.xlsx"))
    simulation_counts = {"dedicated": 0, "self": 0, "none": 0}
    sa_columns = ["Worker_SA", "Manager_SA", "Director_SA", "Reporter_SA"]

    for file in excel_files:
        try:
            # Load Configuration sheet
            config_df = pd.read_excel(file, sheet_name="Configuration", engine="openpyxl")
            sim_id = config_df["Simulation_ID"].iloc[0]
            org_structure = config_df ["Org_Structure"].iloc[0]
            reporting_structure = config_df["Reporting_Structure"].iloc[0]

            if org_structure != "functional":
                logging.warning(f"Skipping {file}: Unexpected org_structure={org_structure}")
                continue

            # Load Model_Metrics sheet
            metrics_df = pd.read_excel(file, sheet_name="Model_Metrics", engine="openpyxl")
            # Clean Step column
            metrics_df["Step"] = metrics_df["Step"].apply(clean_step_value)
            metrics_df = metrics_df.dropna(subset=["Step"])  # Drop rows with invalid steps
            if metrics_df.empty:
                logging.warning(f"Skipping {file}: No valid steps after cleaning")
                continue

            max_step = metrics_df["Step"].max()
            if max_step < 1:
                logging.warning(f"Skipping {file}: No valid steps found")
                continue

            # Use Step 100 if available, else use last available step
            if 100 in metrics_df["Step"].values:
                final_step = metrics_df[metrics_df["Step"] == 100]
                step_used = 100
            else:
                final_step = metrics_df[metrics_df["Step"] == max_step]
                step_used = max_step
                logging.warning(f"No data for Step 100 in {file}. Using Step {max_step} instead.")

            # Clean SA columns
            for col in sa_columns:
                if col in final_step.columns:
                    final_step[col] = final_step[col].apply(clean_numeric_value)
                    if final_step[col].isna().all():
                        logging.warning(f"All {col} values are NaN in {file} (Step {step_used})")
                else:
                    logging.warning(f"Column {col} missing in {file} (Step {step_used})")
                    final_step[col] = np.nan

            # Calculate Average_SA
            final_step["Average_SA"] = final_step[sa_columns].mean(axis=1, numeric_only=True)
            if final_step["Average_SA"].isna().all():
                logging.warning(f"Skipping {file}: Average_SA is NaN for Step {step_used}")
                continue

            final_step = final_step.copy()
            final_step["Simulation_ID"] = sim_id
            final_step["Reporting_Structure"] = reporting_structure
            final_step["Step_Used"] = step_used
            all_metrics.append(final_step)
            simulation_counts[reporting_structure] += 1

            # Load Agent_SA sheet for decision tracking
            agent_df = pd.read_excel(file, sheet_name="Agent_SA", engine="openpyxl")
            agent_df["Step"] = agent_df["Step"].apply(clean_step_value)
            agent_df = agent_df.dropna(subset=["Step"])
            agent_step = agent_df[agent_df["Step"] == step_used]
            if not agent_step.empty:
                agent_step = agent_step.copy()
                for col in ["Reports_Sent", "Reports_Received", "Workload", "SA_Score"]:
                    if col in agent_step.columns:
                        agent_step[col] = agent_step[col].apply(clean_numeric_value)
                        if agent_step[col].isna().all():
                            logging.warning(f"All {col} values are NaN in Agent_SA for {file} (Step {step_used})")
                    else:
                        logging.warning(f"Column {col} missing in Agent_SA for {file} (Step {step_used})")
                agent_step["Simulation_ID"] = sim_id
                agent_step["Reporting_Structure"] = reporting_structure
                agent_step["Step_Used"] = step_used
                agent_decisions.append(agent_step)
            else:
                logging.warning(f"No Agent_SA data for Step {step_used} in {file}")

            logging.info(f"Loaded data from {file} (Step {step_used}, Reporting_Structure: {reporting_structure})")
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue

    if not all_metrics:
        raise ValueError("No valid simulation data found for org_structure='functional'")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    agent_df = pd.concat(agent_decisions, ignore_index=True) if agent_decisions else pd.DataFrame()

    # Log simulation counts
    logging.info(f"Simulation counts: {simulation_counts}")
    if sum(simulation_counts.values()) < 2:
        raise ValueError("Insufficient simulations for ANOVA (need at least 2 reporting structures)")

    return metrics_df, agent_df

def calculate_effect_size(data, variable):
    """Calculate Cohen's f effect size for ANOVA."""
    groups = data.groupby("Reporting_Structure")[variable]
    group_means = groups.mean()
    grand_mean = data[variable].mean()
    n_groups = len(groups)
    between_var = sum(len(groups) * (mean - grand_mean)**2 for _, mean in group_means.items()) / (n_groups - 1)
    within_var = groups.var().mean()
    if within_var == 0:
        logging.warning("Within-group variance is zero, leading to large effect size.")
        return np.inf
    effect_size = np.sqrt(between_var / within_var) if not np.isnan(between_var) else 0
    if effect_size > 10:
        logging.warning(f"Large effect size detected ({effect_size:.4f}). Check data for low variance.")
    return effect_size

def estimate_required_simulations(effect_size, alpha=0.05, power=0.8, k_groups=3):
    """Estimate number of simulations needed for sufficient power."""
    if effect_size == 0 or np.isinf(effect_size):
        logging.warning("Effect size is 0 or infinite. Cannot estimate required simulations.")
        return None
    try:
        power_analysis = FTestAnovaPower()
        # Cap effect size to prevent numerical instability
        effect_size = min(effect_size, 10.0)  # Arbitrary cap to avoid fdtri errors
        sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)
        return int(np.ceil(sample_size))
    except Exception as e:
        logging.error(f"Power analysis failed: {e}")
        return None

def summarize_agent_decisions(agent_df):
    """Summarize agent decisions (reports sent/received, actions) by role and reporting structure."""
    if agent_df.empty:
        print("\nNo Agent_SA data available for decision tracking.")
        return

    decision_summary = agent_df.groupby(["Reporting_Structure", "Role"]).agg({
        "Reports_Sent": ["mean", "std"],
        "Reports_Received": ["mean", "std"],
        "Workload": ["mean", "std"],
        "SA_Score": ["mean", "std"],
        "Step_Used": "mean"
    }).round(2)

    print("\nAgent Decision Summary (Last Step):")
    print(decision_summary)

def analyze_h1(metrics_df, agent_df):
    """Test H1 using one-way ANOVA on Average_SA at last step (preferably Step 100)."""
    print("\nH1: Dedicated reporting structure results in higher Average_SA than Self or None")

    # Calculate Average_SA if not present
    if "Average_SA" not in metrics_df.columns:
        sa_columns = ["Worker_SA", "Manager_SA", "Director_SA", "Reporter_SA"]
        metrics_df["Average_SA"] = metrics_df[sa_columns].mean(axis=1, numeric_only=True)

    # Drop rows where Average_SA is NaN
    metrics_df = metrics_df.dropna(subset=["Average_SA"])
    if metrics_df.empty:
        raise ValueError("No valid Average_SA data available for ANOVA")

    # Group Average_SA by Reporting_Structure
    sa_groups = [
        metrics_df[metrics_df["Reporting_Structure"] == rep]["Average_SA"]
        for rep in ["dedicated", "self", "none"]
        if rep in metrics_df["Reporting_Structure"].values
    ]

    if len(sa_groups) < 2:
        raise ValueError("Not enough reporting structures with data for ANOVA (need at least 2)")

    # One-way ANOVA
    anova_result = f_oneway(*sa_groups)
    print(f"\nOne-Way ANOVA Results for Average_SA (Last Step):")
    print(f"F-statistic: {anova_result.statistic:.4f}")
    print(f"P-value: {anova_result.pvalue:.4f}")

    # Interpret results
    if anova_result.pvalue < 0.05 and not np.isnan(anova_result.pvalue):
        print("Result: Significant difference in Average_SA across reporting structures (p < 0.05)")
        # Post-hoc Tukey HSD
        tukey = pairwise_tukeyhsd(endog=metrics_df["Average_SA"], groups=metrics_df["Reporting_Structure"], alpha=0.05)
        print("\nTukey HSD Post-Hoc Test:")
        print(tukey)
    else:
        print("Result: No significant difference in Average_SA across reporting structures (p >= 0.05 or NaN)")

    # Mean SA by reporting structure
    print("\nMean Average_SA by Reporting Structure:")
    for rep in ["dedicated", "self", "none"]:
        if rep in metrics_df["Reporting_Structure"].values:
            mean_sa = metrics_df[metrics_df["Reporting_Structure"] == rep]["Average_SA"].mean()
            mean_step = metrics_df[metrics_df["Reporting_Structure"] == rep]["Step_Used"].mean()
            print(f"{rep.capitalize()}: {mean_sa:.2f} (Mean Step: {mean_step:.2f})")

    # Effect size and power analysis
    effect_size = calculate_effect_size(metrics_df, "Average_SA")
    print(f"\nEffect Size (Cohen's f): {effect_size:.4f}")
    required_simulations = estimate_required_simulations(effect_size)
    if required_simulations:
        print(f"Estimated simulations needed per reporting structure (alpha=0.05, power=0.8): {required_simulations}")
    else:
        print("Cannot estimate required simulations due to zero or large effect size.")

    # Summarize agent decisions
    summarize_agent_decisions(agent_df)

def main():
    try:
        metrics_df, agent_df = load_simulation_data()
        analyze_h1(metrics_df, agent_df)
        print("\nIf results are not significant or too few simulations are available, check simulation outputs for completeness or increase num_simulations in main.py.")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()