import argparse
import optuna
import yaml
import os
from typing import Any, Dict

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract best gains from Optuna database")
    parser.add_argument("--db_dir", type=str, required=True, help="Directory containing the Optuna SQLite databases.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
        aviary_config = full_config['aviary_rise_node']['ros__parameters']

    best_gains_path = os.path.join(args.db_dir, "best_gains.yaml")
    best_gains: Dict[str, Dict[str, Any]] = {}
    
    def get_best_params(study_name: str, db_file: str) -> Dict[str, Any]:
        full_path = os.path.join(args.db_dir, db_file)
        if not os.path.exists(full_path):
            print(f"Warning: Database file {db_file} does not exist. Skipping.")
            return {}
        
        db_path = f"sqlite:///{full_path}"
        try:
            study = optuna.load_study(study_name=study_name, storage=db_path)
            e_rms = study.best_trial.user_attrs.get('e_RMS', 'N/A')
            u_rms = study.best_trial.user_attrs.get('u_RMS', 'N/A')
            e_str = f"{e_rms:.4f}" if isinstance(e_rms, float) else e_rms
            u_str = f"{u_rms:.4f}" if isinstance(u_rms, float) else u_rms
            print(f"Loaded best params for {study_name} from {db_file}: Cost = {study.best_value:.4f} | e_RMS = {e_str} | u_RMS = {u_str}")
            return dict(study.best_params)
        except Exception as e:
            print(f"Warning: Could not load study {study_name} from {db_file}: {e}")
            return {}
    
    # Load 1A for RISE
    rise_params_no_wind = get_best_params("stage_1A_study", "stage_1A.db")
    if rise_params_no_wind:
        best_gains['BEST_RISE_NO_WIND'] = {
            'controller_type': 'baseline_no_wind',
            **rise_params_no_wind
        }

    # Load 1B for RISE
    rise_params = get_best_params("stage_1B_study", "stage_1B.db")
    if rise_params:
        best_gains['BEST_RISE'] = {
            'controller_type': 'baseline',
            **rise_params
        }

    # Load Stage 3 for SuperTwisting
    st_params = get_best_params("stage_3_study", "stage_3.db")
    if st_params:
        best_gains['BEST_ST'] = {
            'controller_type': 'supertwisting',
            # Map k_st_1 to k_1, etc. as defined in optimization.py
            'k_1': st_params.get('k_st_1', 1.0),
            'k_2': st_params.get('k_st_2', 1.0),
            'k_3': st_params.get('k_st_3', 1.0)
        }

    # Load Stage 4 for PID
    pid_params = get_best_params("stage_4_study", "stage_4.db")
    if pid_params:
        best_gains['BEST_PID'] = {
            'controller_type': 'pid',
            **pid_params
        }
        
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
    base_stage = base_config['stage2_base_gains']
    stage2_base_params = rise_params_no_wind if base_stage == '1A' else rise_params

    fixed_arch = {
        'num_blocks': 6,
        'k_0': 2,
        'k_i': 2,
        'hidden_width': 16,
    }

    # Load Stage 2A for Neural Network Baseline
    nn_params = get_best_params("stage_2A_study", "stage_2A.db")
    if nn_params and stage2_base_params:
        best_gains['BEST_NN'] = {
            'controller_type': 'resnet',
            **stage2_base_params,
            **fixed_arch,
            **nn_params
        }
        
    # Load Stage 2B for Integrated Neural Network
    inn_params = get_best_params("stage_2B_study", "stage_2B.db")
    if inn_params and stage2_base_params:
        best_gains['BEST_INN'] = {
            'controller_type': 'integrated_resnet',
            **stage2_base_params,
            **fixed_arch,
            **inn_params
        }

    os.makedirs(os.path.dirname(best_gains_path), exist_ok=True)
    with open(best_gains_path, 'w') as f:
        yaml.dump(best_gains, f)
        
    print(f"\n[*] Extracted best gains and saved to '{best_gains_path}'")

if __name__ == "__main__":
    main()
