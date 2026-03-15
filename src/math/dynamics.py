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

def f_sys_4(x):
    # 2-State Polynomial
    return jnp.array([
        x[0] * x[1]**2 - x[0]**3,
        x[0]**2 * jnp.sin(x[1]) - x[1]**3
    ])

def f_sys_5(x):
    # 3-State Polynomial
    return jnp.array([
        x[1]**2 * jnp.sin(x[0]) - x[0]**3,
        x[0] * x[2] - x[1]**3,
        x[0]**2 * x[1] - x[2]**3
    ])

def f_sys_6(x):
    # 4-State Polynomial
    return jnp.array([
        x[1] * x[2] * jnp.sin(x[0]) - x[0]**3,
        x[0] * x[3] - x[1]**3,
        x[0] * x[1] - x[2]**3,
        x[1]**2 * x[2] - x[3]**3
    ])

# --- DIMENSION-AWARE TRAJECTORIES ---

def get_desired_trajectory(t, sys_id):
    if sys_id in [1, 2, 3, 4]:
        return jnp.array([
            2.0 * jnp.sin(1.1 * t) + 1.5 * jnp.cos(2.73 * t),
            1.5 * jnp.sin(0.85 * t) - jnp.cos(4.12 * t)
        ])
    elif sys_id == 5:
        return jnp.array([
            2.0 * jnp.sin(1.1 * t),
            1.5 * jnp.cos(0.85 * t),
            1.0 * jnp.sin(2.1 * t) - 0.5 * jnp.cos(1.5 * t)
        ])
    elif sys_id == 6:
        return jnp.array([
            2.0 * jnp.sin(1.1 * t),
            1.5 * jnp.cos(0.85 * t),
            1.0 * jnp.sin(2.1 * t),
            1.2 * jnp.cos(1.3 * t) - jnp.sin(0.9 * t)
        ])

def get_desired_velocity(t, sys_id):
    if sys_id in [1, 2, 3, 4]:
        return jnp.array([
            2.2 * jnp.cos(1.1 * t) - 4.095 * jnp.sin(2.73 * t),
            1.275 * jnp.cos(0.85 * t) + 4.12 * jnp.sin(4.12 * t)
        ])
    elif sys_id == 5:
        return jnp.array([
            2.2 * jnp.cos(1.1 * t),
            -1.275 * jnp.sin(0.85 * t),
            2.1 * jnp.cos(2.1 * t) + 0.75 * jnp.sin(1.5 * t)
        ])
    elif sys_id == 6:
        return jnp.array([
            2.2 * jnp.cos(1.1 * t),
            -1.275 * jnp.sin(0.85 * t),
            2.1 * jnp.cos(2.1 * t),
            -1.56 * jnp.sin(1.3 * t) - 0.9 * jnp.cos(0.9 * t)
        ])

def get_excitation_signal(t, duration, d_out):
    base = jnp.where(t < duration, jnp.sin(10.0 * t) + jnp.sin(2.5 * t) + jnp.cos(5.3 * t), 0.0)
    return jnp.ones(d_out) * base