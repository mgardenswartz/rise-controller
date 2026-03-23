import jax
import jax.numpy as jnp

# =========================================================================
# 2D SYSTEMS (Systems 1 - 4)
# =========================================================================

def f_sys_1(x: jax.Array) -> jax.Array:
    # 2D Van der Pol Oscillator (Bounded, classic limit cycle, easy)
    mu = 1.5
    return jnp.array([
        x[1], 
        mu * (1.0 - x[0]**2) * x[1] - x[0]
    ])

def f_sys_2(x: jax.Array) -> jax.Array:
    # 2D Duffing Oscillator (Highly oscillatory, chaotic potential)
    alpha, beta, delta = 1.0, 1.0, 0.3
    return jnp.array([
        x[1],
        -delta * x[1] - alpha * x[0] - beta * x[0]**3
    ])

def f_sys_3(x: jax.Array) -> jax.Array:
    # 2D Radially Unstable (Finite-time escape, highly punishing)
    return jnp.array([
        x[0]**3 + x[0] * x[1]**2,
        x[1]**3 + x[1] * x[0]**2
    ])

def f_sys_4(x: jax.Array) -> jax.Array:
    # 2D Asymmetric Drift + Constant Bias (The Integral Controller's Playground)
    # The +2.5 acts as a persistent unmodeled disturbance.
    return jnp.array([
        x[1] + 2.5,
        -0.5 * x[0]**2 * jnp.sign(x[0]) - x[1]
    ])

# =========================================================================
# 3D SYSTEMS (Systems 5 - 6)
# =========================================================================

def f_sys_5(x: jax.Array) -> jax.Array:
    # 3D Lorenz System (Chaotic, bounded but highly complex state exploration)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    return jnp.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2]
    ])

def f_sys_6(x: jax.Array) -> jax.Array:
    # 3D Mixed Stability (1 bounded axis, 2 finite-time escape axes)
    return jnp.array([
        -x[0]**3 + x[1],               
        x[1]**3 + x[1] * x[2]**2,      
        x[2]**3 + x[2] * x[0]**2       
    ])

# =========================================================================
# 4D & 6D SYSTEMS (Systems 7 - 8)
# =========================================================================

def f_sys_7(x: jax.Array) -> jax.Array:
    # 4D Radially Unstable (Finite-time escape in higher dimensions)
    return jnp.array([
        x[0]**3 + x[0] * x[3]**2,
        x[1]**3 + x[1] * x[0]**2,
        x[2]**3 + x[2] * x[1]**2,
        x[3]**3 + x[3] * x[2]**2
    ])

def f_sys_8(x: jax.Array) -> jax.Array:
    # 6D Dissipative Polynomial (Massive scale, globally bounded, local complexity)
    return jnp.array([
        -x[0]**3 + jnp.sin(x[1] * x[2]),
        -x[1]**3 + jnp.cos(x[2] * x[3]),
        -x[2]**3 + jnp.sin(x[3] * x[4]),
        -x[3]**3 + jnp.cos(x[4] * x[5]),
        -x[4]**3 + jnp.sin(x[5] * x[0]),
        -x[5]**3 + jnp.cos(x[0] * x[1])
    ])

# =========================================================================
# TRAJECTORY GENERATORS
# =========================================================================

def get_desired_trajectory(t, sys_id):
    if sys_id in [1, 2, 3, 4]: # 2D Systems
        return jnp.array([
            2.0 * jnp.sin(1.1 * t) + 1.5 * jnp.cos(2.73 * t),
            1.5 * jnp.sin(0.85 * t) - jnp.cos(4.12 * t)
        ])
    elif sys_id in [5, 6]: # 3D Systems
        return jnp.array([
            2.0 * jnp.sin(1.1 * t),
            1.5 * jnp.cos(0.85 * t),
            1.0 * jnp.sin(2.1 * t) - 0.5 * jnp.cos(1.5 * t)
        ])
    elif sys_id == 7: # 4D System
        return jnp.array([
            2.0 * jnp.sin(1.1 * t),
            1.5 * jnp.cos(0.85 * t),
            1.0 * jnp.sin(2.1 * t),
            1.2 * jnp.cos(1.3 * t) - jnp.sin(0.9 * t)
        ])
    elif sys_id == 8: # 6D System
        return jnp.array([
            2.0 * jnp.sin(1.1 * t),
            1.5 * jnp.cos(0.85 * t),
            1.0 * jnp.sin(2.1 * t),
            1.2 * jnp.cos(1.3 * t),
            0.8 * jnp.sin(3.1 * t),
            0.5 * jnp.cos(2.5 * t)
        ])

def get_desired_velocity(t, sys_id):
    if sys_id in [1, 2, 3, 4]: # 2D
        return jnp.array([
            2.2 * jnp.cos(1.1 * t) - 4.095 * jnp.sin(2.73 * t),
            1.275 * jnp.cos(0.85 * t) + 4.12 * jnp.sin(4.12 * t)
        ])
    elif sys_id in [5, 6]: # 3D
        return jnp.array([
            2.2 * jnp.cos(1.1 * t),
            -1.275 * jnp.sin(0.85 * t),
            2.1 * jnp.cos(2.1 * t) + 0.75 * jnp.sin(1.5 * t)
        ])
    elif sys_id == 7: # 4D
        return jnp.array([
            2.2 * jnp.cos(1.1 * t),
            -1.275 * jnp.sin(0.85 * t),
            2.1 * jnp.cos(2.1 * t),
            -1.56 * jnp.sin(1.3 * t) - 0.9 * jnp.cos(0.9 * t)
        ])
    elif sys_id == 8: # 6D
        return jnp.array([
            2.2 * jnp.cos(1.1 * t),
            -1.275 * jnp.sin(0.85 * t),
            2.1 * jnp.cos(2.1 * t),
            -1.56 * jnp.sin(1.3 * t),
            2.48 * jnp.cos(3.1 * t),
            -1.25 * jnp.sin(2.5 * t)
        ])

def get_excitation_signal(t, duration, d_out):
    base = jnp.where(t < duration, jnp.sin(10.0 * t) + jnp.sin(2.5 * t) + jnp.cos(5.3 * t), 0.0)
    return jnp.ones(d_out) * base