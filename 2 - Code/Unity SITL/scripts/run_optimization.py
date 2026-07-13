import argparse
import optuna
import numpy as np
import yaml
from typing import Any
import time
from datetime import timedelta
from src.run_sim import SimRun

class ETACallback:
    def __init__(self, target_trials: int):
        self.target_trials = target_trials
        self.start_time = time.time()
        self.trials_completed = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.trials_completed += 1
        elapsed = time.time() - self.start_time
        avg_time_per_trial = elapsed / self.trials_completed
        remaining_trials = self.target_trials - self.trials_completed
        
        if remaining_trials > 0:
            eta_seconds = remaining_trials * avg_time_per_trial
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            print(f"\n[ETA] Trial {self.trials_completed}/{self.target_trials} finished. "
                  f"Avg time/trial: {avg_time_per_trial:.1f}s. Estimated time remaining: {eta_str}\n")


class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.patience <= 0:
            return
            
        try:
            best_trial = study.best_trial
            # If the current trial number is X, and the best trial was found at X - patience, stop!
            if trial.number - best_trial.number >= self.patience:
                print("\n=========================================================================")
                print(f" [Early Stopping] No improvement in the last {self.patience} trials!")
                print(f" Best trial was #{best_trial.number} with cost {best_trial.value:.4f}.")
                print(" Exiting study prematurely.")
                print("=========================================================================\n")
                study.stop()
        except ValueError:
            # Raised if no trial has completed yet
            pass


def evaluate_minibatch(param_dict: dict[str, Any]) -> float:
    """Runs a mini-batch of robust domain randomizations and returns the worst-case cost."""
    with open("conf/config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']

    num_seeds = base_config['num_eval_seeds']
    base_seed = base_config['base_seed']
    xy_range = base_config['xy_rand_range_m']
    z_range = base_config['z_rand_range_m']

    costs = []
    print(f"\n[Mini-Batch] Evaluating {num_seeds} randomized initial conditions:")
    for i in range(num_seeds):
        # Deterministic perturbation based on seed index
        np.random.seed(base_seed + i)
        
        batch_params = param_dict.copy()
        batch_params['init_x_m_ned_aviary'] = base_config['init_x_m_ned_aviary'] + np.random.uniform(-xy_range, xy_range)
        batch_params['init_y_m_ned_aviary'] = base_config['init_y_m_ned_aviary'] + np.random.uniform(-xy_range, xy_range)
        batch_params['hover_start_z_m_ned_aviary'] = base_config['hover_start_z_m_ned_aviary'] + np.random.uniform(-z_range, z_range)
        
        sim = SimRun(batch_params, yaml_config_path="conf/config.yaml")
        cost = sim.run()
        costs.append(cost)
        print(f"  -> Seed #{i+1} | Pos: ({batch_params['init_x_m_ned_aviary']:.2f}, {batch_params['init_y_m_ned_aviary']:.2f}, {batch_params['hover_start_z_m_ned_aviary']:.2f}) | Cost: {cost:.4f}")
        
    worst_cost = float(np.max(costs))
    print(f"[Mini-Batch] Completed. Worst-Case Cost: {worst_cost:.4f}")
    return worst_cost

def run_stage_1a(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'noresnet',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_2': trial.suggest_float("k_2", 0.01, 8.0, log=True),
        'k_3': trial.suggest_float("k_3", 0.01, 8.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 8.0, log=True)
    }
    return evaluate_minibatch(param_dict)

def run_stage_1b(trial: optuna.Trial) -> float:
    param_dict = {
        'controller_type': 'noresnet',
        'k_1': trial.suggest_float("k_1", 0.01, 2.0, log=True),
        'k_2': trial.suggest_float("k_2", 0.01, 8.0, log=True),
        'k_3': trial.suggest_float("k_3", 0.01, 8.0, log=True),
        'k_rise': trial.suggest_float("k_rise", 0.01, 8.0, log=True)
    }
    return evaluate_minibatch(param_dict)

def run_stage_2(trial: optuna.Trial, db_dir: str) -> float:
    # Fetch best gains from Stage 1B to evaluate the NN on top of the best robust baseline
    stage_1b_db = f"sqlite:///{os.path.join(db_dir, 'stage_1B.db')}"
    try:
        study_1b = optuna.load_study(study_name="stage_1B_study", storage=stage_1b_db)
        best_1b_params = study_1b.best_params
    except Exception as e:
        print(f"Warning: Could not load stage_1B.db to seed stage 2! Falling back to config.yaml. Error: {e}")
        best_1b_params = {}

    param_dict = {
        'controller_type': 'baseline', # Stage 2 tunes the Neural Network
        'initial_weight_scale_factor': 0.1, 
        'num_blocks': trial.suggest_categorical("num_blocks", [4, 6, 8]),
        'k_0': trial.suggest_categorical("k_0", [2, 4, 8]),
        'k_i': trial.suggest_categorical("k_i", [2, 4, 8]),
        'hidden_width': trial.suggest_categorical("hidden_width", [4, 8, 12]),
        'gamma': trial.suggest_float("gamma", 0.1, 10.0, log=True),
        'sigma_mod': trial.suggest_float("sigma_mod", 0.5, 5.0, log=True),
        **best_1b_params
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
    parser.add_argument("--db_dir", type=str, required=True, help="Directory for Optuna databases (e.g. output/traj1).")
    parser.add_argument("--patience", type=int, required=True, help="Number of trials to wait for improvement before stopping early. 0 to disable.")
    args = parser.parse_args()

    # Construct the file path and SQLite URL
    db_file_path = os.path.join(args.db_dir, f"stage_{args.stage}.db")
    db_url = f"sqlite:///{db_file_path}"
    
    if os.path.exists(db_file_path):
        print("\n=========================================================================")
        print(f" ⚠️ WARNING: Database '{db_file_path}' already exists.")
        print(" Optuna will RESUME the existing study where it left off.")
        print(" NOTE: This assumes your search space bounds have NOT changed.")
        print(" If you altered the parameter bounds, cancel this run and delete the .db file.")
        print("=========================================================================\n")

    study_name = f"stage_{args.stage}_study"
    study = optuna.create_study(study_name=study_name, storage=db_url, load_if_exists=True, direction="minimize")

    eta_callback = ETACallback(args.num_trials)
    early_stop_callback = EarlyStoppingCallback(args.patience)
    from typing import Callable
    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = [eta_callback, early_stop_callback]

    if args.stage == '1A':
        study.optimize(run_stage_1a, n_trials=args.num_trials, callbacks=callbacks)
    elif args.stage == '1B':
        study.optimize(run_stage_1b, n_trials=args.num_trials, callbacks=callbacks)
    elif args.stage == '2':
        stage_1b_db = f"sqlite:///{os.path.join(args.db_dir, 'stage_1B.db')}"
        try:
            study_1b = optuna.load_study(study_name="stage_1B_study", storage=stage_1b_db)
            print("\n=========================================================================")
            print(" [Sanity Check] Stage 1B Best Gains injected into Stage 2:")
            for k, v in study_1b.best_params.items():
                print(f"    {k}: {v}")
            print("=========================================================================\n")
        except Exception:
            print("\n[!] Could not load Stage 1B gains for sanity check print.\n")
            
        study.optimize(lambda t: run_stage_2(t, args.db_dir), n_trials=args.num_trials, callbacks=callbacks)
    elif args.stage == '3':
        study.optimize(run_stage_3, n_trials=args.num_trials, callbacks=callbacks)