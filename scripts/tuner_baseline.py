import optuna
import jax
import jax.numpy as jnp
from pathlib import Path

# Adjust imports based on your repository structure
from src.core.config_schema import ExperimentConfig, SimulationConfig, MathConstantsConfig
from src.simulation.runner import run_simulation
from src.io.statistics import calculate_and_save_statistics

# Mock config loader for the standalone script
def get_base_config() -> ExperimentConfig:
    # In practice, load this via Hydra or compose it directly
    pass 

def generate_monte_carlo_x0(num_samples: int, key: jax.Array, bounds: float = 3.0) -> jax.Array:
    """Generates a batch of random initial conditions uniformly distributed within [-bounds, bounds]."""
    return jax.random.uniform(key, shape=(num_samples, 2), minval=-bounds, maxval=bounds)

def objective(trial: optuna.Trial, base_config: ExperimentConfig, f_sys_choice: int) -> float:
    # 1. Optuna suggests robust gains
    k_1 = trial.suggest_float("k_1", 0.1, 15.0)
    k_2 = trial.suggest_float("k_2", 0.1, 15.0)
    beta = trial.suggest_float("beta", 0.0, 10.0)
    
    # 2. Force NN OFF (Lock theta to zero and disable learning)
    base_config.math_constants.k_1 = k_1
    base_config.math_constants.k_2 = k_2
    base_config.math_constants.beta = beta
    base_config.math_constants.k_theta_hat = 0.0
    base_config.agent_network.init_mean = 0.0
    base_config.agent_network.init_std = 0.0
    
    # Freeze the learning rate to prevent unnecessary computation
    base_config.math_constants.learning_rate_upper_bound_mult = 1.0
    base_config.math_constants.learning_rate_lower_bound_mult = 1.0
    
    # 3. Monte Carlo Setup
    num_mc_samples = 5
    key = jax.random.PRNGKey(trial.number)
    x0_batch = generate_monte_carlo_x0(num_mc_samples, key, bounds=2.5)
    
    mc_tracking_errors = []
    mc_control_efforts = []
    
    # For a true massive sweep, you would vmap the entire run_simulation function.
    # For baseline tuning, a simple loop is usually fast enough if JIT compiled.
    for i in range(num_mc_samples):
        base_config.simulation.x0 = x0_batch[i].tolist()
        
        try:
            # Note: You will need to pass f_sys_choice to run_simulation or swap it dynamically
            sim_data = run_simulation(base_config)
            stats = calculate_and_save_statistics(sim_data, Path("/tmp"), base_config)
            
            mc_tracking_errors.append(stats["rms_tracking_error_norm"])
            mc_control_efforts.append(stats["rms_control_input_norm"])
            
        except RuntimeError:
            # Heavily penalize finite-time escapes
            return 1e9 

    # 4. Aggregate MC Results
    avg_rms_e = float(jnp.mean(jnp.array(mc_tracking_errors)))
    avg_rms_u = float(jnp.mean(jnp.array(mc_control_efforts)))

    # 5. The "Mediocre Baseline" Cost Function
    # We want a tracking error around 1.25 so the NN has room to prove its worth.
    # Simultaneously, we strictly penalize high control effort.
    target_rms_e = 1.25
    
    error_penalty = abs(avg_rms_e - target_rms_e)
    # Scale u_penalty so it doesn't overpower the error targeting, 
    # but still forces the solver to pick the most efficient gains.
    u_penalty = 0.01 * avg_rms_u 
    
    return error_penalty + u_penalty

if __name__ == "__main__":
    # Example usage for tuning System 1
    config = get_base_config()
    
    study = optuna.create_study(direction="minimize", study_name="baseline_sys1")
    
    # Optimize using a lambda to pass the config and system choice
    study.optimize(lambda trial: objective(trial, config, f_sys_choice=1), n_trials=100)
    
    print("\nBest Baseline Gains for System 1:")
    print(f"k_1:  {study.best_params['k_1']:.3f}")
    print(f"k_2:  {study.best_params['k_2']:.3f}")
    print(f"beta: {study.best_params['beta']:.3f}")