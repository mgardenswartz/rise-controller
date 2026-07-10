import jax
import jax.numpy as jnp


@jax.jit
def traj1_spatial_derivs(
    tau: float,
    target_z: float,
    traj1_period: float,
    traj1_x_amp: float,
    traj1_y_amp: float,
    traj1_z_amp: float
) -> jax.Array:
    def pos_fn(t):
        w = (2.0 * jnp.pi) / traj1_period
        wx, wy, wz = 2.0 * w, 1.0 * w, 4.0 * w
        return jnp.array([
            traj1_x_amp * jnp.sin(wx * t),
            traj1_y_amp * jnp.sin(wy * t),
            traj1_z_amp * jnp.sin(wz * t) + target_z
        ])
    return pos_fn(tau), jax.jacfwd(pos_fn)(tau), jax.jacfwd(jax.jacfwd(pos_fn))(tau)


@jax.jit
def traj2_spatial_derivs(
    theta: float,
    target_z: float,
    traj2_A: float
) -> jax.Array:
    def pos_fn(th):
        r = traj2_A * jnp.cos(2.0 * th)
        return jnp.array([
            r * jnp.cos(th),
            r * jnp.sin(th),
            target_z
        ])
    return pos_fn(theta), jax.jacfwd(pos_fn)(theta), jax.jacfwd(jax.jacfwd(pos_fn))(theta)
