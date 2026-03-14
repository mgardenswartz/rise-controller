import jax
import jax.numpy as jnp

def f_sys(x: jax.Array) -> jax.Array:
    # A plant with polynomial growth. 
    # Weak k_e will cause finite-time escape. Strong k_e will stabilize it.
    return jnp.array([
        0.5 * x[1]**2 * jnp.sin(x[0]), 
        x[0] * x[1] - jnp.cos(x[0])
    ])

def g_sys(x: jax.Array) -> jax.Array:
    return jnp.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

def desired_trajectory(t: float) -> jax.Array:
    w = 2.0 * jnp.pi / 15.0
    return jnp.array([
        5.0 * jnp.cos(w * t),
        5.0 * jnp.sin(w * t) * jnp.cos(w * t)
    ])

def desired_velocity(t: float) -> jax.Array:
    w = 2.0 * jnp.pi / 15.0
    return jnp.array([
        -5.0 * w * jnp.sin(w * t),
        5.0 * w * (jnp.cos(w * t)**2 - jnp.sin(w * t)**2)
    ])

def excitation_signal(t: float, excitation_duration: float) -> jax.Array:
    # p1 dynamics
    p1_term1 = 20.0 * jnp.sin(jnp.sqrt(232.0) * jnp.pi * t) * jnp.cos(jnp.sqrt(20.0) * jnp.pi * t)
    # WARNING: exp(2*t) grows massively. JAX float32 will lose precision around t > 10.
    # We clip the argument to prevent NaNs in the sine wave evaluation.
    safe_exp_1 = jnp.clip(18.0 * jnp.exp(2.0 * t), a_min=-1e2, a_max=1e2)
    p1_term2 = 6.0 * jnp.sin(safe_exp_1)
    p1_term3 = 20.0 * jnp.cos(40.0 * t) * jnp.cos(21.0 * t)
    p1 = 2.55 * jnp.tanh(2.0 * t) * (p1_term1 + p1_term2 + p1_term3)

    # p2 dynamics
    p2_term1 = 20.0 * jnp.sin(jnp.sqrt(132.0) * jnp.pi * t) * jnp.cos(jnp.sqrt(10.0) * jnp.pi * t)
    safe_exp_2 = jnp.clip(8.0 * jnp.exp(t), a_min=-1e2, a_max=1e2)
    p2_term2 = 6.0 * jnp.sin(safe_exp_2)
    p2_term3 = 20.0 * jnp.cos(10.0 * t) * jnp.cos(11.0 * t)
    p2 = 2.25 * jnp.tanh(2.0 * t) * (p2_term1 + p2_term2 + p2_term3)

    u1 = jnp.array([p1, p2])
    
    # Strictly zero out the signal after the excitation duration
    return jnp.where(t <= excitation_duration, u1, jnp.zeros(2))