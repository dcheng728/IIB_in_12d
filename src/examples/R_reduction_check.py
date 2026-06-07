"""
Does the 12d repackaged action of Sec 6.2 reduce to the 10d IIB action?

Gravitational sector check (the eq labelled `axio-dilaton 12d uplift`):
    \hat R  ==  R_base  -  (1/2) |d tau|^2 / tau_2^2
for the block-diagonal ansatz  \hat g = g_base ⊕ \hat g_ab(tau),  with
    \hat g_ab = (1/tau_2) [[1, tau_1],[tau_1, tau_1^2+tau_2^2]]   (det = 1).

We isolate the torus contribution with a *flat* base (so R_base = 0), and let
tau_1, tau_2 be arbitrary functions of one base coordinate x0.  Then we must get
    \hat R = -(1/2)((tau_1')^2 + (tau_2')^2)/tau_2^2.
(The |F4|^2, |F7|^2 and Chern-Simons pieces were checked algebraically by hand.)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sympy as sp
from sugra import Metric

x0, z1, z2 = sp.symbols('x0 z1 z2', real=True)
# Explicit, algebraically-independent test profiles for tau_1(x0), tau_2(x0).
# (Using undefined functions trips the code's H(sqrt(.)) abstraction layer.)
t1 = sp.sin(x0)
t2 = sp.exp(x0)

# Base: 1d flat line (R_base = 0).  Torus: unit-det SL(2)/SO(2) coset block.
g = sp.Matrix([
    [1,         0,                       0                                  ],
    [0,         1/t2,                    t1/t2                              ],
    [0,         t1/t2,                  (t1**2 + t2**2)/t2                  ],
])

metric = Metric(g, [x0, z1, z2])

print("det(torus block) =", sp.simplify(g[1:, 1:].det()))   # should be 1

Rhat = metric.ricci_scalar(simplify_func=sp.cancel)
Rhat = sp.simplify(Rhat)

expected = -sp.Rational(1, 2) * (sp.diff(t1, x0)**2 + sp.diff(t2, x0)**2) / t2**2

print("\n  hat R        =", Rhat)
print("  expected     =", sp.simplify(expected))
print("  difference   =", sp.simplify(Rhat - expected))
print()
print("  MATCH" if sp.simplify(Rhat - expected) == 0 else "  MISMATCH")
