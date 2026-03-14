import jax
import jax.numpy as jnp

def f_sys_1(x: jax.Array) -> jax.Array:
    # Smooth, globally stable-ish nonlinearities
    return jnp.array([
        -0.5 * x[0] + jnp.sin(x[1]), 
        -x[1] - jnp.cos(x[0])
    ])

def f_sys_2(x: jax.Array) -> jax.Array:
    # Polynomial growth (Unstable, prone to finite-time escape)
    return jnp.array([
        0.5 * x[1]**2 * jnp.sin(x[0]), 
        x[0] * x[1] - x[0]**3
    ])

def f_sys_3(x: jax.Array) -> jax.Array:
    # Highly oscillatory, coupled state dependencies
    return jnp.array([
        jnp.sin(x[0] * x[1]),
        -x[1] + jnp.exp(-0.1 * x[0]**2) * jnp.cos(x[0])
    ])

def desired_trajectory(t: float) -> jax.Array:
    # PE condition: Sum of incommensurate frequencies
    w1, w2, w3, w4 = 1.1, 2.73, 0.85, 4.12
    return jnp.array([
        2.0 * jnp.sin(w1 * t) + 1.5 * jnp.cos(w2 * t),
        1.5 * jnp.sin(w3 * t) - 1.0 * jnp.cos(w4 * t)
    ])

def desired_velocity(t: float) -> jax.Array:
    w1, w2, w3, w4 = 1.1, 2.73, 0.85, 4.12
    return jnp.array([
        2.0 * w1 * jnp.cos(w1 * t) - 1.5 * w2 * jnp.sin(w2 * t),
        1.5 * w3 * jnp.cos(w3 * t) + 1.0 * w4 * jnp.sin(w4 * t)
    ])
def excitation_signal(t: float, excitation_duration: float) -> jax.Array:
    p1_term1 = 20.0 * jnp.sin(jnp.sqrt(232.0) * jnp.pi * t) * jnp.cos(jnp.sqrt(20.0) * jnp.pi * t)
    safe_exp_1 = jnp.clip(18.0 * jnp.exp(2.0 * t), a_min=-1e2, a_max=1e2)
    p1_term2 = 6.0 * jnp.sin(safe_exp_1)
    p1_term3 = 20.0 * jnp.cos(40.0 * t) * jnp.cos(21.0 * t)
    p1 = 2.55 * jnp.tanh(2.0 * t) * (p1_term1 + p1_term2 + p1_term3)

    p2_term1 = 20.0 * jnp.sin(jnp.sqrt(132.0) * jnp.pi * t) * jnp.cos(jnp.sqrt(10.0) * jnp.pi * t)
    safe_exp_2 = jnp.clip(8.0 * jnp.exp(t), a_min=-1e2, a_max=1e2)
    p2_term2 = 6.0 * jnp.sin(safe_exp_2)
    p2_term3 = 20.0 * jnp.cos(10.0 * t) * jnp.cos(11.0 * t)
    p2 = 2.25 * jnp.tanh(2.0 * t) * (p2_term1 + p2_term2 + p2_term3)

    u1 = jnp.array([p1, p2])
    return jnp.where(t <= excitation_duration, u1, jnp.zeros(2))