import jax
import jax.numpy as jnp

def activate(s: jax.Array, act_idx: jax.Array) -> jax.Array:
    return jax.lax.switch(
        act_idx,
        [
            lambda x: x,
            lambda x: x * jax.nn.sigmoid(x),
            lambda x: jnp.tanh(x)
        ],
        s
    )

def get_block_parameters(d_in: int, hidden_width: int, d_out: int, k: int) -> int:
    if k == 0:
        return (d_in + 1) * d_out
    p_in = (d_in + 1) * hidden_width
    p_hidden = (k - 1) * (hidden_width + 1) * hidden_width if k > 0 else 0
    p_out = (hidden_width + 1) * d_out
    return p_in + p_hidden + p_out

def get_total_parameters(d_in: int, hidden_width: int, d_out: int, b: int, k_0: int, k_i: int) -> int:
    total = get_block_parameters(d_in, hidden_width, d_out, k_0)
    total += b * get_block_parameters(d_out, hidden_width, d_out, k_i)
    return total

def phi_block(
    v: jax.Array, 
    theta: jax.Array, 
    in_dim: int, 
    hidden_width: int, 
    out_dim: int, 
    k: int, 
    h_act_idx: jax.Array,
    o_act_idx: jax.Array
) -> jax.Array:
    idx = 0
    
    if k == 0:
        v0_size = (in_dim + 1) * out_dim
        v0 = jnp.reshape(theta[idx:idx + v0_size], (in_dim + 1, out_dim), order='F')
        return jnp.dot(v, v0)

    v0_size = (in_dim + 1) * hidden_width
    v0 = jnp.reshape(theta[idx:idx + v0_size], (in_dim + 1, hidden_width), order='F')
    idx += v0_size
    phi_j = jnp.dot(v, v0)
    
    for _ in range(k - 1):
        v_size = (hidden_width + 1) * hidden_width
        v_j = jnp.reshape(theta[idx:idx + v_size], (hidden_width + 1, hidden_width), order='F')
        idx += v_size
        sigma_a = jnp.append(activate(phi_j, h_act_idx), 1.0) 
        phi_j = jnp.dot(sigma_a, v_j)
        
    vk_size = (hidden_width + 1) * out_dim
    vk = jnp.reshape(theta[idx:idx + vk_size], (hidden_width + 1, out_dim), order='F')
    sigma_a_out = jnp.append(activate(phi_j, o_act_idx), 1.0) 
    
    return jnp.dot(sigma_a_out, vk)

def resnet_network(
    theta: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,          
    k_0: int,        
    k_i: int,        
    h_act_idx: jax.Array,
    o_act_idx: jax.Array,
    shortcut_act_idx: jax.Array
) -> jax.Array:
    idx = 0
    
    b0_params = get_block_parameters(d_in, hidden_width, d_out, k_0)
    theta_0 = theta[idx:idx + b0_params]
    idx += b0_params
    
    x_a = jnp.append(x, 1.0)
    kappa = phi_block(x_a, theta_0, d_in, hidden_width, d_out, k_0, h_act_idx, o_act_idx)
    
    bi_params = get_block_parameters(d_out, hidden_width, d_out, k_i)
    
    for _ in range(b):
        theta_i = theta[idx:idx + bi_params]
        idx += bi_params
        
        psi_out = jnp.append(activate(kappa, shortcut_act_idx), 1.0)
        kappa = kappa + phi_block(psi_out, theta_i, d_out, hidden_width, d_out, k_i, h_act_idx, o_act_idx)
        
    return kappa

def compute_jacobian(
    theta: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    b: int,
    k_0: int,
    k_i: int,
    h_act_idx: jax.Array,
    o_act_idx: jax.Array,
    shortcut_act_idx: jax.Array
) -> jax.Array:
    return jax.jacrev(resnet_network, argnums=0)(
        theta, x, d_in, hidden_width, d_out, b, k_0, k_i, h_act_idx, o_act_idx, shortcut_act_idx
    )