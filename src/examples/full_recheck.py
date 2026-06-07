"""
Full re-verification of the Sec 6.2-6.3 claims, addressing the central caveat:
when the 12d action is varied *being aware of the embedding* (i.e. w.r.t. the
embedded 10d fields g_mn, tau, forms), the brane uplifts solve the equations of
motion, and the axio-dilaton EOM come out correctly.

Logic:
  (1) Sec 6.2 action reduces to the 10d IIB action on the ansatz   [gravity sector]
  (2) the 10d brane solutions genuinely solve the 10d IIB EOM       [dilaton a scalar]
  ==> (3) the uplifts solve the embedded EOM of the Sec 6.2 action  [logical consequence]

Diagnostics for contrast:
  (4) the genuine 12d membrane solves the *unconstrained* 12d Einstein eq  [control: PASS]
  (5) the F1 uplift FAILS the unconstrained 12d eq -- the wrong variation   [ignores embedding]
  (6) torus-block residual decomposes: tau-combination = axio-dilaton EOM (=0),
      radion-combination != 0 (a direction outside the embedded field space).
"""
import sys, os, contextlib, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sympy as sp
from sugra import Metric, HarmonicFunction, warped_product, FormField, exterior_derivative, Verifier
from examples.Dp_RMN_TMN import branes
from examples.F1_12d_uplift_check import f1_uplift, genuine_3brane
R = sp.Rational

def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

summary = []

# (1) Sec 6.2 gravity sector reduces to IIB:  hat R = R_base - (1/2)|d tau|^2/tau_2^2
x0, z1, z2 = sp.symbols('x0 z1 z2', real=True)
t1, t2 = sp.sin(x0), sp.exp(x0)
g = sp.Matrix([[1,0,0],[0,1/t2,t1/t2],[0,t1/t2,(t1**2+t2**2)/t2]])
Rhat = sp.simplify(Metric(g, [x0,z1,z2]).ricci_scalar(simplify_func=sp.cancel))
expected = -R(1,2)*(sp.diff(t1,x0)**2 + sp.diff(t2,x0)**2)/t2**2
summary.append(("(1) Sec 6.2 action reduces to IIB (gravity sector)",
                sp.simplify(Rhat - expected) == 0))

# (2) 10d brane solutions solve the 10d IIB EOM (dilaton = scalar field)
for name in ['F1', 'D1', 'NS5', 'D5']:
    V = Verifier(branes[name]); quiet(V.compute)
    summary.append((f"(2) 10d {name} solves 10d IIB EOM", quiet(V.print_results)))

# (4) control: genuine 12d membrane solves the unconstrained 12d Einstein eq
Vg = Verifier(genuine_3brane()); quiet(Vg.compute)
summary.append(("(4) genuine 12d membrane solves unconstrained 12d eq (control)",
                quiet(Vg.print_results)))

# (5) F1 uplift vs unconstrained 12d eq (n=4, D=12) -- the WRONG variation
Vf = Verifier(f1_uplift()); quiet(Vf.compute)
summary.append(("(5) F1 uplift solves UNCONSTRAINED 12d eq (expect FAIL)",
                quiet(Vf.print_results)))

# (6) torus-block decomposition for the F1 uplift (n=4, D=12)
hf = None
s = f1_uplift(); V = Verifier(s); quiet(V.compute); hf = s['hf']
E_z1 = hf.substitute(sp.cancel(V._R[10,10] - V._T[10,10]))
E_z2 = hf.substitute(sp.cancel(V._R[11,11] - V._T[11,11]))
gz1i, gz2i = (sp.sqrt(hf.H)), (1/sp.sqrt(hf.H))
tau_comb    = sp.simplify(gz1i*E_z1 - gz2i*E_z2)   # axio-dilaton (volume preserving)
radion_comb = sp.simplify(gz1i*E_z1 + gz2i*E_z2)   # radion (volume changing)
summary.append(("(6a) tau-combination of torus eqs = axio-dilaton EOM (expect 0)",
                tau_comb == 0))
summary.append(("(6b) radion-combination = 0 ?  (expect NONZERO -> outside embedding)",
                radion_comb == 0))

print("\n" + "="*70)
print("  FULL RE-VERIFICATION SUMMARY")
print("="*70)
for label, ok in summary:
    print(f"  [{'PASS' if ok else 'FAIL/NONZERO'}]  {label}")
print("="*70)
print("  radion residual =", radion_comb, " (sourced by the brane, ~ H'^2)")
print("="*70)
