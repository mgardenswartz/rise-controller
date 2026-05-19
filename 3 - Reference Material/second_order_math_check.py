import sympy as sp

# 1. Define the constant gains and error variables
k1, k2, k3 = sp.symbols('k1 k2 k3')
e, e_dot, e_ddot = sp.symbols('e e_dot e_ddot')

# 2. Define the auxiliary signals based on your definitions
r1 = e_dot + k1 * e
r1_dot = e_ddot + k1 * e_dot
r2 = r1_dot + k2 * r1 + e

# 3. Define the linear portion of your controller derivative (\dot{u})
# Note: We omit \beta*sgn(r1) and \Phi, as they are trivially carried over.
dot_u_linear = (k1 + k2 + k3)*r2 + \
               (2 - k1**2 - k2**2 - k1*k2)*r1 + \
               (k1**3 - 2*k1 - k2)*e

# 4. Expand and collect the terms with respect to the error derivatives
dot_u_expanded = sp.expand(dot_u_linear)
dot_u_simplified = sp.collect(dot_u_expanded, (e_ddot, e_dot, e))

# 5. Output the result
print("Verified Implementable Controller terms:")
print(dot_u_simplified)