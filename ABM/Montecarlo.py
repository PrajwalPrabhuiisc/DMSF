import itertools
import logging
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from filelock import FileLock
from construction_model import ConstructionModel

# Configure logging
logging.basicConfig(filename='monte_carlo_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_simulation(params):
    run_id, hazard_prob, delay_prob, resource_prob, reporting_structure, org_structure = params
    try:
        model = ConstructionModel(
            width=20,
            height=20,
            reporting_structure=reporting_structure,
            org_structure=org_structure,
            hazard_prob=hazard_prob,
            delay_prob=delay_prob,
            resource_prob=resource_prob,
            comm_failure_dedicated=0.05,
            comm_failure_self=0.05,
            comm_failure_none=0.10,
            worker_detection=0.80,
            manager_detection=0.90,
            reporter_detection=0.99,
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
        output_file = model.excel_filepath
        lock_file = output_file + '.lock'
        with FileLock(lock_file):
            if os.path.exists(output_file):
                logging.debug(f"Simulation {run_id} completed, output saved to {output_file}")
                return output_file
            else:
                logging.error(f"Simulation {run_id} failed to save output file {output_file}")
                return None
    except Exception as e:
        logging.error(f"Error in simulation {run_id}: {str(e)}")
        return None

def main():
    output_dir = os.path.join(os.getcwd(), "monte_carlo_outputs_2700")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.debug(f"Created output directory: {output_dir}")

    hazard_probs = [0.05, 0.10, 0.15]
    delay_probs = [0.05, 0.10, 0.15]
    resource_probs = [0.05, 0.10, 0.15]
    reporting_structures = ['dedicated', 'self', 'none']
    org_structures = ['functional', 'flat', 'hierarchical']
    n_runs = 33

    param_combinations = list(itertools.product(
        range(n_runs),
        hazard_probs,
        delay_probs,
        resource_probs,
        reporting_structures,
        org_structures
    ))

    total_simulations = len(param_combinations)
    logging.debug(f"Starting {total_simulations} Monte Carlo simulations")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, param_combinations))

    successful_sims = [r for r in results if r is not None]
    logging.debug(f"Completed {len(successful_sims)} out of {total_simulations} simulations")
    print(f"Completed {len(successful_sims)} out of {total_simulations} simulations")

if __name__ == "__main__":
    main()
