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

def compute_controller_in_integral(
    e: jax.Array,
    x_d_dot: jax.Array,
    z: jax.Array,
    u_1: jax.Array,
    phi_eval: jax.Array,
    k_1: float,
    k_2: float,
    beta: float
) -> tuple[jax.Array, jax.Array]:
    u = (k_1 + k_2) * e + x_d_dot + z + u_1
    z_dot = beta * jnp.sign(e) + (k_1 * k_2 + 1.0) * e - phi_eval
    return u, z_dot

def compute_controller_outside_integral(
    e: jax.Array,
    x_d_dot: jax.Array,
    z: jax.Array,
    u_1: jax.Array,
    phi_eval: jax.Array,
    k_1: float,
    k_2: float,
    beta: float
) -> tuple[jax.Array, jax.Array]:
    u = x_d_dot + phi_eval + (k_1 + k_2) * e + z + u_1
    z_dot = (k_1 * k_2 + 1.0) * e + beta * jnp.sign(e)
    return u, z_dot