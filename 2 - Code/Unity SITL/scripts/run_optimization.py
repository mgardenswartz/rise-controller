import argparse
import optuna
import numpy as np
import yaml
from typing import Any
from src.run_sim import SimRun

def evaluate_minibatch(param_dict: dict[str, Any]) -> float:
    """Runs a mini-batch of robust domain randomizations and returns the worst-case cost."""
    with open("conf/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']

    num_seeds = base_config['num_eval_seeds']
    base_seed = base_config['base_seed']
    xy_range = base_config['xy_rand_range']
    z_range = base_config['z_rand_range']

    costs = []
    for i in range(num_seeds):
        # Deterministic perturbation based on seed index
        np.random.seed(base_seed + i)
        
        batch_params = param_dict.copy()
        batch_params['init_x'] = base_config['init_x'] + np.random.uniform(-xy_range, xy_range)
        batch_params['init_y'] = base_config['init_y'] + np.random.uniform(-xy_range, xy_range)
        batch_params['hover_start_z'] = base_config['hover_start_z'] + np.random.uniform(-z_range, z_range)
        
        sim = SimRun(batch_params, yaml_config_path="conf/config.yaml")
        cost = sim.run()
        costs.append(cost)
        
    return np.max(costs) # Return the worst-case robust cost

def run_stage_1a(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'noresnet',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_2': trial.suggest_float("k_2", 0.01, 8.0, log=True),
        'k_3': trial.suggest_float("k_3", 0.01, 8.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 5.0, log=True)
    }
    return evaluate_minibatch(param_dict)

def run_stage_1b(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'noresnet',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_2': trial.suggest_float("k_2", 0.01, 8.0, log=True),
        'k_3': trial.suggest_float("k_3", 0.01, 8.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 5.0, log=True)
    }
    return evaluate_minibatch(param_dict)

def run_stage_2(trial: optuna.Trial) -> float:
    param_dict = {
        'initial_weight_scale_factor': 0.1, 
        'num_blocks': trial.suggest_categorical("num_blocks", [4, 6, 8]),
        'k_0': trial.suggest_categorical("k_0", [2, 4, 8]),
        'k_i': trial.suggest_categorical("k_i", [2, 4, 8]),
        'hidden_width': trial.suggest_categorical("hidden_width", [4, 8, 12]),
        'gamma': trial.suggest_float("gamma", 0.1, 10.0, log=True),
        'sigma_mod': trial.suggest_float("sigma_mod", 0.5, 5.0, log=True)
    }
    return evaluate_minibatch(param_dict)

def run_stage_3(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'supertwisting',
        'k_1': trial.suggest_float("k_st_1", 0.001, 5.0, log=True),
        'k_2': trial.suggest_float("k_st_2", 0.001, 5.0, log=True),
        'k_3': trial.suggest_float("k_st_3", 0.001, 5.0, log=True)
    }
    return evaluate_minibatch(param_dict)

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Optuna Orchestrator for Quadcopter Adaptive Control")
    parser.add_argument("--stage", type=str, required=True, choices=['1A', '1B', '2', '3', 'LHS'], help="Optimization stage to run.")
    parser.add_argument("--num_trials", type=int, required=True, help="Number of trials.")
    parser.add_argument("--db", type=str, required=True, help="Optuna database string.")
    args = parser.parse_args()

    # Extract the file path from the SQLite URL to check if it exists
    db_file_path = args.db.replace("sqlite:///", "")
    
    if os.path.exists(db_file_path):
        print("\n=========================================================================")
        print(f" ⚠️ WARNING: Database '{db_file_path}' already exists.")
        print(" Optuna will RESUME the existing study where it left off.")
        print(" NOTE: This assumes your search space bounds have NOT changed.")
        print(" If you altered the parameter bounds, cancel this run and delete the .db file.")
        print("=========================================================================\n")

    study_name = f"stage_{args.stage}_study"
    study = optuna.create_study(study_name=study_name, storage=args.db, load_if_exists=True, direction="minimize")

    if args.stage == '1A':
        study.optimize(run_stage_1a, n_trials=args.num_trials)
    elif args.stage == '1B':
        study.optimize(run_stage_1b, n_trials=args.num_trials)
    elif args.stage == '2':
        study.optimize(run_stage_2, n_trials=args.num_trials)
    elif args.stage == '3':
        study.optimize(run_stage_3, n_trials=args.num_trials)