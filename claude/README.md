# mochi_class `stable_params` input generators

Scripts that generate the `stable_params` input files mochi_class needs,
consistently with a user-chosen cosmology — replacing the Mathematica
notebook that produced the files shipped in `stable_params_input/`:

- `make_cubic_galileon_inputs.py` — self-accelerating cubic covariant
  Galileon (exact, via the analytic tracker solution); `expansion_model = rho_de`;
- `make_designer_fr_inputs.py` — **designer f(R)**: f(R) on an *exact* LCDM
  background, parametrised by B0 (solves the designer ODE); `expansion_model = lcdm`;
- `make_hs_fr_inputs.py` — Hu–Sawicki f(R), **stage 1**: leading order in
  `f_R0` (LCDM background + high-curvature limit; array-level approximation).

**Which f(R) script?** Hu–Sawicki and designer are *different* models. HS fixes
the function f(R) and the background comes out ≈LCDM (use the model-matched
`rho_de` file; `lcdm` does **not** work — see `expansion_consistency_test.md`).
Designer fixes the background to *exactly* LCDM and solves for f(R) (use
`expansion_model = lcdm`, no `rho_de` file). For a runnable LCDM-background
f(R), use the **designer** generator; the HS stage-1 files reproduce HS *array
values* but are not a self-consistent runnable LCDM-background model.

(The other `.py` files in this folder are stock CLASS example scripts.)

---

# `make_cubic_galileon_inputs.py`

## Why

The `stable_params` inputs — `{Delta M*^2, D_kin, cs^2}(lna)`, the
`rho_de(lna)/rho_de(0)` expansion file, and the boundary condition
`parameters_smg` — are **not** free of the cosmology: they depend on the
matter/radiation/neutrino fractions through the background solution of the
theory. The files in `stable_params_input/` were generated for the cosmology
of `notebooks/stable_params.ipynb`
(`h=0.6781, omega_b=0.0223828, omega_cdm=0.1201075, N_ur=3.044`, massless
neutrinos). Using them with a different cosmology is internally inconsistent —
for e.g. `H0=67, Omega_b=0.049, Omega_cdm=0.27, 3×0.0193 eV` neutrinos the
mismatch reaches ~3% in the `rho_de` shape, ~8% in `D_kin`, ~4% in `cs^2`,
and 1.5% in `parameters_smg`.

## Physics

On the tracker of the self-accelerating cubic Galileon
(Cataneo et al., arXiv:2407.11968, Eq. 9):

    alpha_K = 3 alpha_B = 6 Omega_phi0 / E^4(a),   M*^2 = 1,   alpha_M = alpha_T = 0

and `rho_phi ∝ 1/H^2`, so the Friedmann equation closes **algebraically** —
no ODE is solved:

    E^2(a) = [ S(a) + sqrt(S(a)^2 + 4 Omega_phi0) ] / 2
    S(a)   = sum of all non-smg Omega_i(a)   (photons, baryons, cdm, ur, ncdm)

From this:

    rho_de(a)/rho_de(0) = 1/E^2(a)
    D_kin               = alpha_K + (3/2) alpha_B^2
    cs^2 D_kin          = (2 - alpha_B)(-Hdot/H^2 + alpha_B/2)
                          + d(alpha_B)/dlna - 3 (rho+p)_m / (M_Pl^2 H^2)
    Delta M*^2          = 0   (identically)

The boundary condition mochi_class reads from `parameters_smg` is
`alpha_B(z=0) = 2 Omega_phi0`.

Massive neutrinos are included via Fermi–Dirac integrals with the CLASS
default temperature ratio `T_ncdm/T_gamma = 0.71611`.

## Usage

Run with a Python that has numpy + scipy (the repo `.venv` works):

```bash
# 1) Reproduce / validate against the shipped Mathematica files
#    (must be run from this directory; uses ../stable_params_input/)
../.venv/bin/python make_cubic_galileon_inputs.py \
    --h 0.6781 --omega-b 0.0223828 --omega-cdm 0.1201075 --N-ur 3.044 --validate

# 2) Generate files for a new cosmology (fractions + massive neutrinos)
../.venv/bin/python make_cubic_galileon_inputs.py \
    --H0 67 --Omega-b 0.049 --Omega-cdm 0.27 \
    --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
    --out-stable cgal_stable.dat --out-rho-de cgal_rho_de.dat
```

Densities can be given physical (`--omega-b`, `--omega-cdm`) or fractional
(`--Omega-b`, `--Omega-cdm`); Hubble as `--h` or `--H0`. Grid options:
`--lna-min` (default −5), `--lna-max-stable` (default 0.25, the stable file
extends past a=1 as mochi requires), `--n-points`. The `rho_de` file is
written on the same grid truncated at lna = 0 and normalised to 1 there.

The script prints the matching ini-file settings, e.g.

```
Omega_phi0   = 0.6795571699295931
alpha_B(z=0) = 1.3591143398591863
  parameters_smg = 1.359114339859186
```

to be used with

```ini
gravity_model       = stable_params
smg_file_name       = <out-stable file>
parameters_smg      = <alpha_B(z=0) printed above>
expansion_model     = rho_de
expansion_smg       = <Omega_phi0 guess; tuned by shooting when Omega_smg = -1>
expansion_file_name = <out-rho-de file>
```

## Validation

`--validate` with the `stable_params.ipynb` cosmology reproduces the shipped
Mathematica files to:

| quantity  | max rel. deviation | median  |
|-----------|--------------------|---------|
| `rho_de`  | 2.9e-6             | 5.8e-8  |
| `D_kin`   | 5.8e-6             | 1.3e-7  |
| `cs^2`    | 9.5e-5 (z>~100 only; 1e-7 at low z) | 2.9e-8 |

and gives `alpha_B(z=0) = 1.3800520` vs the shipped `1.3800774`
(0.002% — traceable to sub-percent differences in the assumed radiation
content of the original notebook).

## Caveats

- **Cubic-Galileon-specific.** The analytic shortcut is the tracker solution;
  other models need their own background calculation (nKGB would be a small
  extension — its alphas are also analytic, arXiv:2407.11968 Eqs. 10–11; for
  Hu–Sawicki f(R) see `make_hs_fr_inputs.py` below).
- The tracker assumes the self-accelerating branch (c2 = −1) and that the
  field is on the attractor at lna = −5; this matches the shipped files.
- Conventions follow CLASS/hi_class: `Omega_gamma h^2 = 4.48131635e-7 T_cmb^4`,
  `N_ur` with the standard 7/8 (4/11)^(4/3) factor.

---

# `make_designer_fr_inputs.py`

Designer f(R): the expansion is fixed to *exactly* LCDM and f(R) is solved for.
Parametrised by **B0** (the Compton wavelength parameter today). The background
is LCDM by construction, so the mochi pairing is `expansion_model = lcdm` with
**no** `rho_de` file — this is the f(R) case where `lcdm` is consistent.

## Physics

From the f(R) trace equation with `f = R - 2Lambda + delta`, `f_R = 1 + delta_R`
(`delta_R = Delta M*^2`), the Lambda/Omega_m pieces cancel and one gets a
*linear, homogeneous* 3rd-order system in `N = ln a` (primes = d/dN, hats =/H0^2):

    delta'   = delta_R r'
    delta_R' = g
    g'       = -(H'/H + 3) g + (1/(3 E^2)) (delta_R r - 2 delta)

with `r = R/H0^2 = 3 Omega_m a^-3 + 12 Omega_Lambda` (+ ncdm trace; radiation is
traceless). Being linear, the amplitude is the single free parameter, fixed by
`B0 = alpha_M(a=1)/(dlnH/dlna)|_1`. The matter-domination growing mode is
`delta_R ~ a^p`, `p = (5+sqrt(73))/4 ≈ 3.386`; it is seeded deep in matter
domination and integrated forward (decaying modes wash out), then renormalised
to B0. Outputs: `Delta M*^2 = f_R`, `alpha_M = f_R'/(1+f_R)`, `alpha_B = -alpha_M`,
`D_kin = (3/2) alpha_B^2`, `cs^2 = 1`; `parameters_smg = alpha_B(z=0)`.

## Usage

```bash
# validate against the shipped designer file
../.venv/bin/python make_designer_fr_inputs.py --h 0.6781 --omega-b 0.0223828 \
    --omega-cdm 0.1201075 --N-ur 3.044 --B0 0.01 --validate

# generate for a new cosmology
../.venv/bin/python make_designer_fr_inputs.py --H0 67 --Omega-b 0.049 \
    --Omega-cdm 0.27 --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
    --B0 0.01 --out-stable designer_stable.dat
```

Use with `expansion_model = lcdm`, `parameters_smg = <alpha_B(z=0) printed>`,
no `expansion_file_name`.

## Validation (vs `designer_fr_stable_params_B0_0p01.dat`, file cosmology)

| quantity   | max rel. deviation |
|------------|--------------------|
| `Delta_M2` | 1.9e-3             |
| `D_kin`    | 3.9e-3             |
| `cs^2`     | exact (=1)         |

The residual is a uniform ~0.2% B0-calibration offset (the `1/(1+f_R)` factor in
the alpha_M definition; `alpha_B0 = 4.659e-3` vs shipped `4.650e-3`). End-to-end,
the generated file run through mochi_class with `expansion_model = lcdm`
reproduces the shipped designer file's P(k,z=0) to **0.02%**.

## References

- **Song, Hu & Sawicki (2007)**, "The Large Scale Structure of f(R) Gravity",
  Phys. Rev. D 75, 044004 — arXiv:astro-ph/0610532.
  *The designer construction: fix the expansion history and solve the
  second-order ODE for f_R; the matter-domination growing/decaying modes
  (the growing exponent p = (5+sqrt(73))/4 used to seed the integration).*
- **Pogosian & Silvestri (2008)**, "The pattern of growth in viable f(R)
  cosmologies", Phys. Rev. D 77, 023503 — arXiv:0709.0296.
  *The Compton-wavelength parameter B0 that parametrises the designer family
  (the single amplitude of the linear solution), and its z=0 boundary
  condition B0 = alpha_M / (dlnH/dlna).*
- **Bellini & Sawicki (2014)**, "Maximal freedom at minimum cost: linear
  large-scale structure in general modifications of gravity", JCAP 07 (2014)
  050 — arXiv:1404.3713.
  *The alpha_i (alpha_K, alpha_B, alpha_M, alpha_T) EFT basis; here
  M*^2 = 1+f_R, alpha_M = dln(1+f_R)/dlna, alpha_B = -alpha_M, alpha_K = 0.*
- **Hu & Sawicki (2007)**, "Models of f(R) cosmic acceleration that evade
  solar-system tests", Phys. Rev. D 76, 064004 — arXiv:0705.1158.
  *The reference f(R) model class (the designer solution is model-independent
  but this fixes conventions and normalisation of f_R0/B0).*
- **mochi_class**, Cataneo et al. (2024) — arXiv:2407.11968.
  *The `stable_params` basis {Delta M*^2, D_kin, cs^2} consumed by the code,
  and `parameters_smg = alpha_B(z=0)` as the boundary condition; the shipped
  `designer_fr_stable_params_B0_0p01.dat` is the validation target.*

---

# `make_hs_fr_inputs.py` (stage 1)

Hu–Sawicki f(R) version of the above, at **leading order in `f_R0`**:

- background expansion = exact LCDM (+radiation +ncdm), so the effective
  dark-energy density is constant and the `rho_de` file is identically 1
  (equivalently one can use `expansion_model = lcdm`);
- high-curvature limit of the Hu–Sawicki function,
  `f_R(a) = f_R0 [R_0/R(a)]^(n+1)` with the Ricci scalar on the LCDM
  background `R/(3H0^2) = (Omega_b+Omega_cdm) a^-3 + (rho_nu-3p_nu)/rho_crit0
  + 4 Omega_Lambda` (radiation is traceless);
- alpha basis: `Delta M*^2 = f_R`, `alpha_B = -alpha_M = -dln(1+f_R)/dlna`,
  `alpha_K = 0`, `D_kin = (3/2) alpha_B^2`, `cs^2 = 1` exactly;
  `parameters_smg = alpha_B(z=0)`.

## Usage

```bash
# validate against the shipped Mathematica files
# (note --fR0=-1e-4 with '=' so argparse accepts the negative value)
../.venv/bin/python make_hs_fr_inputs.py --h 0.6781 --omega-b 0.0223828 \
    --omega-cdm 0.1201075 --N-ur 3.044 --fR0=-1e-4 --n 1 --validate

# generate for a new cosmology
../.venv/bin/python make_hs_fr_inputs.py --H0 67 --Omega-b 0.049 --Omega-cdm 0.27 \
    --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
    --fR0=-1e-4 --n 1 --out-stable hs_stable.dat --out-rho-de hs_rho_de.dat
```

## Validation (vs the shipped Mathematica files, their cosmology)

| file pair              | `Delta_M2` max | `D_kin` max | `cs^2` |
|------------------------|----------------|-------------|--------|
| `f_R0=-1e-4, n=1`      | 5.5e-4         | 1.4e-3      | exact  |
| `f_R0=-1e-4, n=4`      | 2.9e-3         | 9.6e-3      | exact  |
| `f_R0=-1e-8, n=1`      | 9.3e-5         | 1.9e-4      | exact  |

These residuals are the **expected stage-1 systematics**: the shipped files
solve the exact f(R) background, which differs from LCDM at O(f_R0) (and more
strongly for n=4). The flat ~9e-5 offset surviving in the `1e-8` case is a
cosmology-convention difference (the originals' exact radiation content),
the same ~0.002–0.01% floor seen in the cubic-Galileon validation — it is
irrelevant when generating files for your own cosmology. The constant
`rho_de` misses the O(f_R0) structure of the exact background (max 6.7e-4
for `f_R0=-1e-4, n=1`).

**Stage 2** (planned): solve the exact Hu–Sawicki background ODE to capture
the O(f_R0) corrections and the non-trivial `rho_de` shape.
