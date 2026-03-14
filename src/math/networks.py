import jax
import jax.numpy as jnp

def activate(s: jax.Array, act_idx: jax.Array) -> jax.Array:
    return jax.lax.switch(
        act_idx,
        [
            lambda x: x,                             # Linear
            lambda x: x * jax.nn.sigmoid(x),         # Swish
            lambda x: jnp.tanh(x)                    # Tanh
        ],
        s
    )

def get_total_parameters(d_in: int, hidden_width: int, d_out: int, num_layers: int) -> int:
    """Calculates exactly p = sum( d_{j-1}^a * d_j )"""
    p_in = (d_in + 1) * hidden_width
    p_hidden = (num_layers - 1) * (hidden_width + 1) * hidden_width if num_layers > 0 else 0
    p_out = (hidden_width + 1) * d_out
    return p_in + p_hidden + p_out

def phi_network(
    theta: jax.Array,
    x: jax.Array,
    d_in: int,
    hidden_width: int,
    d_out: int,
    num_layers: int,      # Maps to your 'k'
    h_act_idx: jax.Array, # Activation for \sigma_1 to \sigma_{k-1}
    o_act_idx: jax.Array  # Activation for \sigma_k
) -> jax.Array:
    idx = 0
    
    # --- j = 0 ---
    v0_size = (d_in + 1) * hidden_width
    v0 = jnp.reshape(theta[idx:idx + v0_size], (d_in + 1, hidden_width))
    idx += v0_size
    
    x_a = jnp.append(x, 1.0)
    phi_j = jnp.dot(x_a, v0)  # This is \Phi_0
    
    # --- j = 1 to k-1 ---
    for _ in range(num_layers - 1):
        v_size = (hidden_width + 1) * hidden_width
        v_j = jnp.reshape(theta[idx:idx + v_size], (hidden_width + 1, hidden_width))
        idx += v_size
        
        # \sigma_{j,a}(\Phi_{j-1}) using hidden activation
        sigma_a = jnp.append(activate(phi_j, h_act_idx), 1.0)
        phi_j = jnp.dot(sigma_a, v_j)
        
    # --- j = k (Final Layer) ---
    vk_size = (hidden_width + 1) * d_out
    vk = jnp.reshape(theta[idx:idx + vk_size], (hidden_width + 1, d_out))
    
    # \sigma_{k,a}(\Phi_{k-1}) using outer activation (\sigma_k)
    sigma_a_out = jnp.append(activate(phi_j, o_act_idx), 1.0)
    
    # Final output is strictly a linear projection, preserving your math
    return jnp.dot(sigma_a_out, vk)