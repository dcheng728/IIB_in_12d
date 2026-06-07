"""
Does the F1 12d uplift solve the 12d Einstein equation of the Section 6.2 action?

For F1 the composite 7-form and the CS term vanish, so the Sec 6.2 action is just
   S = int d^12x sqrt(-g) ( R - 1/2 |F4|^2 ),
pure 12d gravity + a 4-form.  We test R_{MN} = T_{MN}[F4] (n=4, D_trace=12, no dilaton).

Controls:
  (B) the GENUINE 12d electric 3-brane sourced by F4 (warps -2/3, +2/7, H harmonic
      in 9 transverse dims) -- should PASS, validating the code and the warp claim.
  (C) the F1 uplift checked against the 10d weights (n_eff=3, D_trace=10) -- to test
      the old "n=3, D=10" phrasing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sympy as sp
from sugra import HarmonicFunction, warped_product
from sugra import FormField, exterior_derivative, Verifier

R = sp.Rational


# ── (A) F1 12d uplift ────────────────────────────────────────────────────────
def f1_uplift():
    t, x1, z1, z2 = sp.symbols('t x1 z1 z2', real=True)
    y = list(sp.symbols('y0:8', real=True))          # 8 F1-transverse dims
    coords = [t, x1] + y + [z1, z2]                  # t=0, x1=1, y..=2..9, z1=10, z2=11

    hf = HarmonicFunction(transverse_coords=y)        # H harmonic in 8 dims
    H = sp.Function('H')(hf.r_expr)

    metric = warped_product(
        warp_factors     = [H**R(-3,4), H**R(1,4), H**R(-1,2), H**R(1,2)],
        block_dims       = [2, 8, 1, 1],
        block_signatures = ['lorentzian', 'euclidean', 'euclidean', 'euclidean'],
        coordinates      = coords,
    )

    C3 = FormField(rank=3, dim=12)
    C3[(0, 1, 10)] = 1 / H                            # B_{t x1} z1 leg:  C3_{t,x1,z1}=H^{-1}
    F4 = exterior_derivative(C3, coords)

    return dict(metric=metric, forms=[[F4, 0]], Phi=sp.S(0), coords=coords, hf=hf)


# ── (B) genuine 12d electric 3-brane (control, should PASS) ──────────────────
def genuine_3brane():
    t, x1, z1 = sp.symbols('t x1 z1', real=True)
    y = list(sp.symbols('y0:9', real=True))           # 9 transverse dims
    coords = [t, x1, z1] + y                           # wv = (t,x1,z1)

    hf = HarmonicFunction(transverse_coords=y)         # H harmonic in 9 dims
    H = sp.Function('H')(hf.r_expr)

    metric = warped_product(
        warp_factors     = [H**R(-2,3), H**R(2,7)],
        block_dims       = [3, 9],
        block_signatures = ['lorentzian', 'euclidean'],
        coordinates      = coords,
    )

    Delta = R(21, 5)                                   # alpha=0, 2*d*dt/(D-2)=42/10
    C3 = FormField(rank=3, dim=12)
    C3[(0, 1, 2)] = (2 / sp.sqrt(Delta)) / H           # electric, Stelle normalization
    F4 = exterior_derivative(C3, coords)

    return dict(metric=metric, forms=[[F4, 0]], Phi=sp.S(0), coords=coords, hf=hf)


if __name__ == '__main__':
    print("\n############  (B) CONTROL: genuine 12d electric 3-brane  ############")
    Verifier(genuine_3brane()).check()

    print("\n############  (A) F1 12d uplift  vs  12d Einstein eq (n=4, D=12)  ############")
    Verifier(f1_uplift()).check()

    print("\n############  (C) F1 uplift  vs  10d weights (n_eff=3, D_trace=10)  ############")
    s = f1_uplift()
    s['forms'] = [[s['forms'][0][0], 0, 3]]            # n_eff = 3
    s['D_trace'] = 10
    Verifier(s).check()
