import sympy as sp

# Define symbolic variables
e, edot, eddot = sp.symbols('e edot eddot')
k1, k2, k3 = sp.symbols('k1 k2 k3')
Kr_sgn, Phi = sp.symbols('Kr_sgn Phi')

# Define intermediate error signals based on your Section 7 & 8 definitions
r1 = edot + k1 * e
r1dot = eddot + k1 * edot
r2 = r1dot + k2 * r1

# Define the continuous control derivative 
# Note: Using 1 to represent the Identity matrix for algebraic simplification
u_dot = (k1 + k2 + k3)*r2 + (1 - k1**2 - k2**2 - k1*k2)*r1 + k1**3 * e + Kr_sgn + Phi

# Expand and collect by the error derivative terms
u_dot_expanded = sp.expand(u_dot)
u_dot_collected = sp.collect(u_dot_expanded, (eddot, edot, e))

print("Simplified u_dot equation:")
sp.pprint(u_dot_collected)