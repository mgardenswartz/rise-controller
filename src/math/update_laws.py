import jax
import jax.numpy as jnp

def cai_projection(
    theta_dot_unprojected: jax.Array,
    theta_hat: jax.Array,
    theta_bar: float,
    gamma: jax.Array
) -> jax.Array:
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

def compute_theta_hat_dot(
    error: jax.Array,
    theta_hat: jax.Array,
    jacobian: jax.Array,
    gamma: jax.Array,
    k_theta_hat: float,
    theta_bar: float
) -> jax.Array:
    unprojected = jnp.dot(gamma, jnp.dot(jacobian.T, error) - k_theta_hat * theta_hat)
    return cai_projection(unprojected, theta_hat, theta_bar, gamma)
