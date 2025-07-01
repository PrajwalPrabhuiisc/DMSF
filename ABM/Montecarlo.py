import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime
import multiprocessing as mp
from itertools import product
import time
from multiprocessing import Lock
from construction_model import ConstructionModel
from enums import ReportingStructure, OrgStructure

# Configure logging
logging.basicConfig(filename='monte_carlo_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lock for safe file writing
file_lock = Lock()

def run_simulation(config, run_id, seed):
    try:
        random.seed(seed)
        np.random.seed(seed)
        model = ConstructionModel(
            width=20,
            height=20,
            reporting_structure=config['reporting_structure'],
            org_structure=config['org_structure'],
            hazard_prob=config['hazard_prob'],
            delay_prob=0.10,  # Fixed at 0.10
            resource_prob=0.05,
            comm_failure_dedicated=config['comm_failure_dedicated'],
            comm_failure_self=config['comm_failure_self'],
            comm_failure_none=config['comm_failure_none'],
            worker_detection=0.80,
            manager_detection=0.90,
            reporter_detection=config['reporter_detection'],
            director_detection=0.85,
            worker_reporting=0.80,
            manager_reporting=0.90,
            reporter_reporting=0.95,
            director_reporting=0.85,
            initial_budget=1000000,
            initial_equipment=500,
            run_id=run_id
        )
        model.run_simulation(steps=150)
        model_data = model.datacollector.get_model_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
        agent_data = model.datacollector.get_agent_vars_dataframe().reset_index().rename(columns={'index': 'Step'})
        return {
            'run_id': run_id,
            'config': config,
            'model_data': model_data,
            'agent_data': agent_data
        }
    except Exception as e:
        logging.error(f"Error in simulation run {run_id}: {str(e)}")
        print(f"Error in simulation run {run_id}: {str(e)}")
        return None

def aggregate_results(results, output_file):
    model_dfs = []
    agent_dfs = []
    for res in results:
        if res is None:
            continue
        model_df = res['model_data'].copy()
        model_df['run_id'] = res['run_id']
        model_df['reporting_structure'] = res['config']['reporting_structure']
        model_df['org_structure'] = res['config']['org_structure']
        model_df['hazard_prob'] = res['config']['hazard_prob']
        model_df['reporter_detection'] = res['config']['reporter_detection']
        model_dfs.append(model_df)
        
        agent_df = res['agent_data'].copy()
        agent_df['run_id'] = res['run_id']
        agent_df['reporting_structure'] = res['config']['reporting_structure']
        agent_df['org_structure'] = res['config']['org_structure']
        agent_df['hazard_prob'] = res['config']['hazard_prob']
        agent_df['reporter_detection'] = res['config']['reporter_detection']
        agent_dfs.append(agent_df)
    
    if not model_dfs or not agent_dfs:
        logging.error("No valid simulation results to aggregate")
        print("Error: No valid simulation results to aggregate")
        return
    
    model_summary = pd.concat(model_dfs).groupby(['reporting_structure', 'org_structure', 'hazard_prob', 'reporter_detection', 'Step']).agg(['mean', 'std']).reset_index()
    agent_summary = pd.concat(agent_dfs).groupby(['reporting_structure', 'org_structure', 'hazard_prob', 'reporter_detection', 'Step', 'Role']).agg(['mean', 'std']).reset_index()
    
    with file_lock:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a' if os.path.exists(output_file) else 'w') as writer:
                    model_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
                    agent_summary.to_excel(writer, sheet_name='Agent_Summary', index=False)
                print(f"Monte Carlo results saved to {output_file}")
                logging.debug(f"Monte Carlo results saved to {output_file}")
                break
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"Error saving to {output_file} after {max_attempts} attempts: {e}")
                logging.error(f"Error saving to {output_file}: {str(e)}")
                model_summary.to_csv(output_file.replace('.xlsx', '_model_summary.csv'), index=False)
                agent_summary.to_csv(output_file.replace('.xlsx', '_agent_summary.csv'), index=False)
                print(f"Fallback: Saved to CSV files")
                logging.debug(f"Fallback: Saved to CSV files")

def main():
    # Configurations (fixed delay_prob at 0.10)
    configs = [
        {
            'reporting_structure': rs.value,
            'org_structure': os.value,
            'hazard_prob': hp,
            'comm_failure_dedicated': 0.05,
            'comm_failure_self': 0.10,
            'comm_failure_none': 0.50,
            'reporter_detection': rd
        }
        for rs, os, hp, rd in product(
            ReportingStructure, OrgStructure, [0.05, 0.10, 0.15], [0.90, 0.95, 0.99]
        )
    ]
    n_runs = 100
    output_dir = "monte_carlo_outputs"
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Created output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir}: {str(e)}")
        print(f"Failed to create output directory {output_dir}: {e}")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"monte_carlo_{timestamp}.xlsx")

    # Run simulations in parallel
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(run_simulation, args=(config, i * len(configs) + j, random.randint(0, 1000000)))
               for i in range(n_runs) for j, config in enumerate(configs)]
    results = [r.get() for r in results]
    pool.close()
    pool.join()

    # Aggregate and save results
    aggregate_results(results, output_file)

if __name__ == "__main__":
    main()
