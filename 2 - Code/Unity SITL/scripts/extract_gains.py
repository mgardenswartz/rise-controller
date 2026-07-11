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
        full_config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
        
    best_gains_path = full_config.get('best_gains_path', 'conf/best_gains.yaml')
    best_gains: Dict[str, Dict[str, Any]] = {}
    
    def get_best_params(study_name: str, db_file: str) -> Dict[str, Any]:
        db_path = f"sqlite:///{os.path.join(args.db_dir, db_file)}"
        try:
            study = optuna.load_study(study_name=study_name, storage=db_path)
            print(f"Loaded best params for {study_name} from {db_file}: Cost = {study.best_value:.4f}")
            return dict(study.best_params)
        except Exception as e:
            print(f"Warning: Could not load study {study_name} from {db_file}: {e}")
            return {}

    # Load 1B for RISE (assumed robust baseline)
    rise_params = get_best_params("stage_1B_study", "stage_1B.db")
    if rise_params:
        best_gains['BEST_RISE'] = {
            'controller_type': 'noresnet',
            **rise_params
        }

    # Load Stage 3 for SuperTwisting
    st_params = get_best_params("stage_3_study", "stage_3.db")
    if st_params:
        best_gains['BEST_ST'] = {
            'controller_type': 'supertwisting',
            # Map k_st_1 to k_1, etc. as defined in run_optimization.py
            'k_1': st_params.get('k_st_1', 1.0),
            'k_2': st_params.get('k_st_2', 1.0),
            'k_3': st_params.get('k_st_3', 1.0)
        }
        
    # Load Stage 2 for Neural Network
    nn_params = get_best_params("stage_2_study", "stage_2.db")
    if nn_params and rise_params:
        # NN Feedforward uses same NN params but controller_type 'baseline'
        best_gains['BEST_NN'] = {
            'controller_type': 'baseline',
            **rise_params,
            **nn_params
        }
        
        # INN Integrated uses same NN params but controller_type 'developed'
        best_gains['BEST_INN'] = {
            'controller_type': 'developed',
            **rise_params,
            **nn_params
        }

    os.makedirs(os.path.dirname(best_gains_path), exist_ok=True)
    with open(best_gains_path, 'w') as f:
        yaml.dump(best_gains, f)
        
    print(f"\n[*] Extracted best gains and saved to '{best_gains_path}'")

if __name__ == "__main__":
    main()
