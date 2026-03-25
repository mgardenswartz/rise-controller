import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

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

def setup_rho_filter(rho_k1: float, rho_k2: float, rho_k3: float, dt: float):
    """Computes the exact ZOH discrete matrices for the Rho Filter."""
    m12 = -(3.0 - rho_k1**2 + (2.0 * rho_k1 + rho_k2 + rho_k3) * (rho_k1 + rho_k2))
    m13 = -(rho_k1 + rho_k2)
    m14 = -(1.0 + (rho_k1 + rho_k2) * (rho_k3 - rho_k1))
    m32 = -((2.0 * rho_k1 + rho_k2 + rho_k3) * (rho_k1 + rho_k2) + 1.0)
    m33 = -(2.0 * rho_k1 + rho_k2)
    m42 = -(2.0 * rho_k1 + rho_k2 + rho_k3)
    m43 = -(rho_k1 + rho_k2) * (rho_k3 - rho_k1)

    M = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [m12, 0.0, m32, m42],
        [m13, 0.0, m33, -1.0],
        [m14, 0.0, m43, -rho_k3]
    ])

    N = jnp.array([0.0, -m12, -m13, -m14])
    E = jnp.array([0.0, 1.0, 0.0, 0.0])
    I_4 = jnp.eye(4)

    A_d = jsp_linalg.expm(M * dt)
    M_inv = jnp.linalg.pinv(M) 
    
    B_q = M_inv @ (A_d - I_4) @ N
    B_s = M_inv @ (A_d - I_4) @ E
    
    return A_d, B_q, B_s

def update_rho_filter(zeta: jax.Array, e_meas: jax.Array, A_d: jax.Array, B_q: jax.Array, B_s: jax.Array, rho_k4: float):
    """
    Propagates the filter state. 
    zeta is a (4, n) matrix where row 0 is z1, row 1 is z2, etc.
    """
    z1 = zeta[0, :]
    z3 = zeta[2, :]
    
    sgn_vec = jnp.sign(e_meas - z1 - z3)
    
    # Elegant matrix multiplication replaces the C++ for-loops
    next_zeta = A_d @ zeta + jnp.outer(B_q, e_meas) + rho_k4 * jnp.outer(B_s, sgn_vec)
    
    # e_hat_dot_rho is z2, which is the 1st index (row 1) of the state matrix
    e_hat_dot_rho = next_zeta[1, :]
    
    return next_zeta, e_hat_dot_rho