from typing import Callable
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

def compute_gamma_dot(
    gamma: jax.Array,
    jacobian: jax.Array,
    gamma_bar_upper: float,
    gamma_bar_lower: float,
    nu: float,
    p: int
) -> jax.Array:
    alpha = (gamma_bar_upper * (gamma_bar_lower ** 3)) / (gamma_bar_upper ** 2 - gamma_bar_lower ** 2)
    beta = gamma_bar_lower
    gamma_scalar = (gamma_bar_lower * gamma_bar_upper) / (gamma_bar_upper ** 2 - gamma_bar_lower ** 2)

    norm_j_sq = jnp.linalg.norm(jacobian) ** 2
    j_t_j = jnp.dot(jacobian.T, jacobian)

    term1 = alpha * jnp.eye(p)
    term2 = beta * gamma
    term3 = gamma_scalar * jnp.dot(gamma, gamma)
    matrix_fraction = jnp.dot(gamma, jnp.dot(j_t_j, gamma)) / (1.0 + nu * norm_j_sq)
    
    return term1 + term2 - term3 - matrix_fraction

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

def compute_control_input(
    x: jax.Array,
    x_d_dot: jax.Array,
    error: jax.Array,
    phi_eval: jax.Array,
    u_1: jax.Array,
    k_e: float,
    g_func: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    g_val = g_func(x)
    g_pseudo = jnp.linalg.pinv(g_val)
    return jnp.dot(g_pseudo, x_d_dot - k_e * error - phi_eval) + u_1