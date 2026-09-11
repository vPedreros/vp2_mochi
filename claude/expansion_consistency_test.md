# Can `expansion_model = lcdm` be used with `stable_params` files?

**Short answer: No — not for the cubic Galileon, and not even for Hu–Sawicki
f(R), when using `stable_params` files generated for the model's own
(non-LCDM) background.** The `{ΔM², D_kin, cs²}` functions are tied to the
background they were computed on; pairing them with a different expansion
makes the system internally inconsistent, and the tiny early-time kinetic
term `D_kin` amplifies that inconsistency until the solver traps — exactly
the "step size too small" behaviour you remembered.

## Experiment

Controlled: each pair of runs is **identical except `expansion_model`**.
Same cosmology, same `{ΔM², D_kin, cs²}` file, same `parameters_smg`,
same `z_gr_smg`. Files in this folder: `cgal_rho_de.ini`, `cgal_lcdm.ini`
(thesis cosmology, generated `cgal_stable_thesis.dat` /
`cgal_rho_de_thesis.dat`).

### Cubic Galileon (thesis cosmology, evolver = rk)

| `z_gr_smg` | `rho_de` (consistent tracker) | `lcdm` (inconsistent) |
|-----------:|-------------------------------|-----------------------|
| 10         | sane, maxP = 2.9e4            | **garbage**, maxP = 3.5e31 |
| 20         | sane, maxP = 2.9e4            | **garbage**, maxP = 1.7e241 |
| 49         | sane, maxP = 2.9e4            | **FAIL: Isnan x''** |
| 99         | sane, maxP = 2.9e4            | **FAIL: Isnan x''** |

The consistent background runs and is stable at every `z_gr_smg`; the LCDM
background is always broken — finite-but-nonsense P(k) when the instability
has little time to grow, hard NaN-trapping otherwise. The universe-closure
shooting even lands on a different H(a) (the output k-grid shifts).

### Hu–Sawicki f(R) (shipped Mathematica file, file cosmology)

| integrator | `rho_de` (exact f(R) bg) | `lcdm`            |
|------------|--------------------------|-------------------|
| rk (=0)    | runs (z_gr=99)           | **FAIL: too small** |
| ndf15 (=1) | runs (z_gr=99)           | **FAIL: too small** (tau≈1339, the z_gr transition) |

Note the HS `rho_de` file differs from LCDM by only ~7e-4 — yet that is
already enough to trap the solver at the gr→smg transition. (HS prefers
*high* `z_gr_smg`; the Galileon prefers *low* — opposite, but both confirm
the same conclusion.)

## Why

In `stable_params`, mochi reconstructs the braiding `α_B` by integrating its
ODE on the *provided* background, while `α_M = dln M*²/dln a` is fixed by the
`ΔM²` file. If the background is not the one the files were built on, the
reconstructed `α_B` and the input functions no longer satisfy the model's
Friedmann/field closure. The scalar EOM is `x'' = (source)/(c_D·(2−c_B))`
with `c_D ∝ D_kin`, and `D_kin → 0` at early times (the GR limit). A small
background inconsistency divided by a tiny `D_kin` is amplified enormously →
exponential growth → trap.

- **Cubic Galileon:** consistent background is the self-accelerating tracker,
  `w_smg` running from ≈ −2 (early) to ≈ −1.18 (today) — grossly non-LCDM.
  LCDM is an O(1) inconsistency → catastrophic.
- **Hu–Sawicki f(R):** consistent background ≈ LCDM to O(f_R0) ~ 7e-4. LCDM is
  *almost* right, but the residual still trips the transition.

## Practical guidance

- **Always use the `rho_de` (or model-matched) expansion file** that goes with
  your `stable_params` file. Your `thesis plots.ipynb` already does this for
  both `mochi_cgal` and `mochi_frhs` — keep it.
- For an LCDM-like f(R) background, supply the (near-constant) `rho_de` file —
  do **not** switch to `expansion_model = lcdm`.
- The stage-1 HS generator writes a *self-consistent* pair (LCDM background +
  matching functions), so its files belong together; do not mix them with the
  shipped (exact-background) files.

## Reproduce

```bash
cd claude
../class cgal_rho_de.ini   # consistent: sane P(k)
../class cgal_lcdm.ini      # inconsistent: garbage / trap
```
