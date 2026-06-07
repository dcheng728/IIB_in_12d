"""
Decompose the failing torus-direction Einstein equations of the F1 uplift.

12d action of Sec 6.2 (for F1: pure gravity + F4).  Genuine 12d Einstein eq
   E_MN = R_MN - T_MN[F4]   (n=4, D=12).
The torus block depends on tau only through  \hat g_ab(tau),  tau_1=0, tau_2=e^{-Phi}=H^{1/2}:
   g_{z1z1}=H^{-1/2}=1/tau_2 ,  g_{z2z2}=H^{1/2}=tau_2 ,  g_{z1z2}=0.

Field-space directions at tau_1=0:
   * tau_2 (dilaton):  delta g_{z1z1}/g_{z1z1} = -delta tau_2/tau_2 ,  delta g_{z2z2}/g_{z2z2} = +delta tau_2/tau_2
     -> conjugate combination is the DIFFERENCE  g^{z1z1}E_{z1z1} - g^{z2z2}E_{z2z2}.
   * volume (radion): delta g_{z1z1}/g_{z1z1} = delta g_{z2z2}/g_{z2z2}
     -> conjugate combination is the SUM       g^{z1z1}E_{z1z1} + g^{z2z2}E_{z2z2}.

Hypothesis: DIFFERENCE = 0 (dilaton EOM satisfied), SUM != 0 (radion EOM fails).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sympy as sp
from sugra import HarmonicFunction, warped_product, FormField, exterior_derivative, Verifier
R = sp.Rational

t, x1, z1, z2 = sp.symbols('t x1 z1 z2', real=True)
y = list(sp.symbols('y0:8', real=True))
coords = [t, x1] + y + [z1, z2]                       # z1=index 10, z2=index 11
hf = HarmonicFunction(transverse_coords=y)
H = sp.Function('H')(hf.r_expr)

metric = warped_product(
    warp_factors     = [H**R(-3,4), H**R(1,4), H**R(-1,2), H**R(1,2)],
    block_dims       = [2, 8, 1, 1],
    block_signatures = ['lorentzian', 'euclidean', 'euclidean', 'euclidean'],
    coordinates      = coords,
)
C3 = FormField(rank=3, dim=12); C3[(0,1,10)] = 1/H
F4 = exterior_derivative(C3, coords)

gz1_inv = (H**R(1,2)).subs(H, hf.H)     # g^{z1z1} = 1/g_{z1z1} = H^{1/2}
gz2_inv = (H**R(-1,2)).subs(H, hf.H)    # g^{z2z2} = 1/g_{z2z2} = H^{-1/2}

def decompose(forms, D_trace, label):
    V = Verifier(dict(metric=metric, forms=forms, Phi=sp.S(0),
                      coords=coords, hf=hf, D_trace=D_trace))
    V.compute()
    E_z1 = hf.substitute(sp.cancel(V._R[10,10] - V._T[10,10]))
    E_z2 = hf.substitute(sp.cancel(V._R[11,11] - V._T[11,11]))
    diff = sp.simplify(gz1_inv*E_z1 - gz2_inv*E_z2)   # dilaton (tau_2) combination
    summ = sp.simplify(gz1_inv*E_z1 + gz2_inv*E_z2)   # volume (radion) combination
    print(f"\n=== {label} ===")
    print("  E_{z1z1}             =", E_z1)
    print("  E_{z2z2}             =", E_z2)
    print("  DIFFERENCE (dilaton) =", diff, "  ->", "ZERO" if diff == 0 else "NONZERO")
    print("  SUM        (radion)  =", summ, "  ->", "ZERO" if summ == 0 else "NONZERO")

# (A) genuine 12d weights:  n=4, D_trace=12
decompose([[F4, 0]],        12, "genuine 12d weights (n=4, D_trace=12)")
# (C) 10d weights:          n_eff=3, D_trace=10
decompose([[F4, 0, 3]],     10, "10d weights (n_eff=3, D_trace=10)")
