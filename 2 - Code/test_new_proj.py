import jax
import jax.numpy as jnp

@jax.jit
def discrete_adaptive_projection(
    theta_hat: jax.Array,
    theta_dot_unprojected: jax.Array,
    dt: float,
    theta_bar: float,
    gamma_diag: jax.Array
) -> jax.Array:
    """
    Discrete-time projection that guarantees Lyapunov stability for a diagonal Gamma matrix.
    """
    # 1. Take the unconstrained discrete Euler step
    theta_temp = theta_hat + dt * theta_dot_unprojected
    
    # 2. Check if we are already safely inside the hypersphere boundary
    is_inside = jnp.sum(theta_temp**2) <= theta_bar**2
    
    def apply_projection(_: None) -> jax.Array:
        # We need to find the Lagrange multiplier 'eta' via bisection.
        
        # Calculate a strict mathematical upper bound for eta.
        # This guarantees our root is between 0 and eta_upper_init.
        gamma_min = jnp.min(gamma_diag)
        norm_temp = jnp.linalg.norm(theta_temp)
        eta_upper_init = (norm_temp / theta_bar - 1.0) / gamma_min
        
        # Initial bisection state: (low, high)
        init_state = (0.0, eta_upper_init)
        
        def bisection_step(i, state):
            eta_low, eta_high = state
            eta_mid = 0.5 * (eta_low + eta_high)
            
            # Evaluate the constraint boundary at eta_mid
            theta_test = theta_temp / (1.0 + eta_mid * gamma_diag)
            val = jnp.sum(theta_test**2) - theta_bar**2
            
            # If val > 0, we haven't projected enough (we need a larger eta).
            new_low = jnp.where(val > 0, eta_mid, eta_low)
            new_high = jnp.where(val > 0, eta_high, eta_mid)
            return (new_low, new_high)
        
        # 30 iterations of bisection is highly precise for float32/float64
        # lax.fori_loop guarantees fixed execution time for real-time controllers
        final_low, final_high = jax.lax.fori_loop(0, 30, bisection_step, init_state)
        eta_opt = 0.5 * (final_low + final_high)
        
        # Apply the optimal Lagrange multiplier to get the final projected state
        return theta_temp / (1.0 + eta_opt * gamma_diag)

    def bypass_projection(_: None) -> jax.Array:
        return theta_temp

    # 3. Conditionally run the solver only if we escaped the boundary
    return jax.lax.cond(
        is_inside,
        bypass_projection,
        apply_projection,
        None
    )


def test_discrete_projection():
    print("--- Testing Discrete JAX Bisection Projection ---\n")
    
    dt = 1.0 / 60.0  # 60 Hz
    theta_bar = 2.0  
    
    # A non-uniform diagonal Gamma matrix (as a 1D array)
    gamma_diag = jnp.array([1.0, 5.0])
    
    # Start on the boundary
    theta_hat = jnp.array([2.0, 0.0])
    
    # Aggressive update pushing strictly outward and upward
    theta_dot_unprojected = jnp.array([1.0, 50.0])
    
    print(f"Initial theta_hat: {theta_hat}")
    print(f"Unprojected velocity: {theta_dot_unprojected}\n")
    
    # Run the projection
    theta_next = discrete_adaptive_projection(
        theta_hat=theta_hat,
        theta_dot_unprojected=theta_dot_unprojected,
        dt=dt,
        theta_bar=theta_bar,
        gamma_diag=gamma_diag
    )
    
    final_norm = jnp.linalg.norm(theta_next)
    
    print(f"Next theta_hat (Projected): {theta_next}")
    print(f"Next theta_hat Norm:        {final_norm:.6f}")
    print(f"Target Boundary (theta_bar): {theta_bar:.6f}\n")
    
    if final_norm <= theta_bar + 1e-5:
        print("SUCCESS: Escape prevented. The value is strictly inside or on the boundary.")
    else:
        print("FAILURE: The value escaped the projection set.")

if __name__ == "__main__":
    test_discrete_projection()