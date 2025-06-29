import pandas as pd
import numpy as np
import random
from construction_model import ConstructionModel
from enums import ReportingStructure, OrgStructure
import os
from datetime import datetime
import multiprocessing as mp
from itertools import product

def run_simulation(config, run_id, seed):
    random.seed(seed)
    np.random.seed(seed)
    model = ConstructionModel(
        reporting_structure=config['reporting_structure'],
        org_structure=config['org_structure'],
        hazard_prob=config['hazard_prob'],
        delay_prob=config['delay_prob'],
        comm_failure_dedicated=config['comm_failure_dedicated'],
        comm_failure_self=config['comm_failure_self'],
        comm_failure_none=config['comm_failure_none'],
        reporter_detection=config['reporter_detection']
    )
    model.run_simulation(steps=150)
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    return {
        'run_id': run_id,
        'config': config,
        'model_data': model_data,
        'agent_data': agent_data
    }

def aggregate_results(results):
    model_dfs = []
    agent_dfs = []
    for res in results:
        model_df = res['model_data'].copy()
        model_df['run_id'] = res['run_id']
        model_df['reporting_structure'] = res['config']['reporting_structure']
        model_df['org_structure'] = res['config']['org_structure']
        model_df['hazard_prob'] = res['config']['hazard_prob']
        model_dfs.append(model_df)
        
        agent_df = res['agent_data'].copy()
        agent_df['run_id'] = res['run_id']
        agent_df['reporting_structure'] = res['config']['reporting_structure']
        agent_df['org_structure'] = res['config']['org_structure']
        agent_dfs.append(agent_df)
    
    model_summary = pd.concat(model_dfs).groupby(['reporting_structure', 'org_structure', 'Step']).agg(['mean', 'std']).reset_index()
    agent_summary = pd.concat(agent_dfs).groupby(['reporting_structure', 'org_structure', 'Step', 'Role']).agg(['mean', 'std']).reset_index()
    return model_summary, agent_summary

def main():
    # Configurations
    configs = [
        {
            'reporting_structure': rs.value,
            'org_structure': os.value,
            'hazard_prob': hp,
            'delay_prob': 0.10,
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
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"monte_carlo_{timestamp}.xlsx")

    # Run simulations in parallel
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(run_simulation, args=(config, i, i)) for i in range(n_runs) for config in configs]
    results = [r.get() for r in results]
    pool.close()
    pool.join()

    # Aggregate and save results
    model_summary, agent_summary = aggregate_results(results)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        model_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
        agent_summary.to_excel(writer, sheet_name='Agent_Summary', index=False)
    print(f"Monte Carlo results saved to {output_file}")

if __name__ == "__main__":
    main()
