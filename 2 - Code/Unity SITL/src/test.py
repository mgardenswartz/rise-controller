import jax
import jax.numpy as jnp

my_vector = jnp.array([0, 1, 2])
print(jnp.shape(my_vector.reshape(3,1)))

my_vector = jnp.ones(3)
print(jnp.shape(my_vector))