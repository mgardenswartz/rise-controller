import jax
import jax.numpy as jnp

def cai_projection(
    theta_dot_unprojected: jax.Array,
    theta_hat: jax.Array,
    theta_bar: float,
    gamma: jax.Array
) -> jax.Array:
    """Original continuous-time projection law."""
    cond1 = jnp.dot(theta_hat, theta_hat) >= theta_bar**2
    cond2 = jnp.dot(theta_hat, theta_dot_unprojected) > 0.0

    def apply_projection(_: None) -> jax.Array:
        numerator = jnp.dot(theta_hat, theta_dot_unprojected)
        denominator = jnp.dot(theta_hat, jnp.dot(gamma, theta_hat))
        return theta_dot_unprojected - (numerator / denominator) * jnp.dot(gamma, theta_hat)

    def bypass_projection(_: None) -> jax.Array:
        return theta_dot_unprojected

    return jax.lax.cond(
        jnp.logical_and(cond1, cond2),
        apply_projection,
        bypass_projection,
        None
    )

def simulate_euler_escape():
    print("--- Testing Continuous Projection under Discrete Euler Integration ---\n")
    
    # 1. Setup the environment parameters
    dt = 1.0 / 60.0  # 60 Hz controller
    theta_bar = 2.0  # Radius of our hypersphere boundary
    
    # Gamma is typically a symmetric positive definite adaptation gain matrix. 
    # We will use the Identity matrix for simplicity.
    gamma = jnp.eye(2)
    
    # 2. Place theta_hat exactly on the boundary of the hypersphere.
    # The 2-norm of [2.0, 0.0] is exactly 2.0 (which equals theta_bar).
    theta_hat = jnp.array([2.0, 0.0])
    
    # 3. Create an aggressive update law term.
    # The x-component (1.0) pushes outward (triggering the projection).
    # The y-component (50.0) is a stiff lateral movement along the tangent.
    theta_dot_unprojected = jnp.array([1.0, 50.0])
    
    # 4. Apply the projection algorithm
    theta_dot_proj = cai_projection(
        theta_dot_unprojected=theta_dot_unprojected,
        theta_hat=theta_hat,
        theta_bar=theta_bar,
        gamma=gamma
    )
    
    # 5. Perform the Euler Integration step
    theta_hat_next = theta_hat + dt * theta_dot_proj
    
    # 6. Evaluate the results
    initial_norm = jnp.linalg.norm(theta_hat)
    final_norm = jnp.linalg.norm(theta_hat_next)
    
    print(f"Time step (dt):                  {dt:.4f} seconds (60 Hz)")
    print(f"Boundary Radius (theta_bar):     {theta_bar:.4f}")
    print(f"Initial theta_hat:               {theta_hat}")
    print(f"Initial theta_hat Norm:          {initial_norm:.4f}\n")
    
    print(f"Unprojected Derivative:          {theta_dot_unprojected}")
    print(f"Projected Derivative:            {theta_dot_proj}")
    print("  * Notice the projected derivative is perfectly orthogonal to theta_hat.\n")
    
    print(f"Next theta_hat (Euler Step):     {theta_hat_next}")
    print(f"Next theta_hat Norm:             {final_norm:.4f}")
    
    # 7. Proof check
    escape_magnitude = final_norm - theta_bar
    print("\n--- Conclusion ---")
    if final_norm > theta_bar:
        print(f"PROOF OF ESCAPE: The projection set was escaped by {escape_magnitude:.4f} units.")
        print("Because the projected vector is tangent to the boundary, moving along it in "
              "a straight line via discrete integration inherently steps outside the curved boundary.")
    else:
        print("The value remained inside the projection set.")

if __name__ == "__main__":
    simulate_euler_escape()