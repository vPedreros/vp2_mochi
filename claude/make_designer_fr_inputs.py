#!/usr/bin/env python
"""Generate mochi_class stable_params input files for DESIGNER f(R) gravity
on an exact LCDM background, consistently with a user-chosen cosmology.

Unlike Hu-Sawicki (where you specify f(R) and the background comes out close
to -- but not exactly -- LCDM), the *designer* approach FIXES the expansion
to LCDM and solves for the f(R) that produces it. The background is therefore
exactly LCDM by construction, so the matching mochi setup is

    expansion_model = lcdm          (NO rho_de file)
    parameters_smg  = alpha_B(z=0)

This is the f(R) class for which `expansion_model = lcdm` is the *consistent*
pairing (see ../stable_params_input/designer_fr_stable_params_B0_*.dat and
inifiles/designer_fr.ini).

------------------------------------------------------------------------------
Physics (derived from the f(R) trace equation, LCDM background, dust matter):

  Field eq trace:   f_R R - 2 f + 3 [] f_R = 8 pi G T = -8 pi G rho_m
  Write f = R - 2 Lambda + delta,  f_R = 1 + delta_R  (delta_R = Delta M*^2).
  With  R = 3 H0^2 (Omega_m a^-3 + 4 Omega_Lambda)   [radiation is traceless:
  it contributes 0 to R, but it DOES enter H], the Lambda / Omega_m pieces
  cancel exactly and the trace equation reduces to a *linear, homogeneous*
  system in e-folds N = ln a (primes = d/dN, hats = /H0^2):

      delta'  = delta_R r'
      delta_R' = g
      g'      = -(H'/H + 3) g + (1/(3 E^2)) (delta_R r - 2 delta)

  with r = R/H0^2,  E^2 = H^2/H0^2.  Being linear and homogeneous, the overall
  amplitude is free -- it is the single physical parameter, fixed by

      B0 = alpha_M(a=1) / (dln H/dln a)|_{a=1}        (Compton wavelength today)

  In matter domination the power-law solutions delta_R ~ a^p obey
      2 p^3 - 3 p^2 - 11 p - 6 = 0  =>  p = -1, (5 - sqrt(73))/4, (5 + sqrt(73))/4
  The growing mode is p+ = (5 + sqrt(73))/4 ~ 3.386; we seed it deep in matter
  domination and integrate forward (decaying modes wash out), then renormalise
  to the requested B0.

  Stable-basis outputs (alpha_K = alpha_T = 0 for f(R)):
      Delta M*^2 = f_R = delta_R
      alpha_M    = f_R' / (1 + f_R) = g / (1 + delta_R),   alpha_B = -alpha_M
      D_kin      = (3/2) alpha_B^2,      cs^2 = 1   (exactly)
      parameters_smg = alpha_B(z=0)

References (see README.md for annotations):
  Song, Hu & Sawicki 2007, PRD 75, 044004,  arXiv:astro-ph/0610532   (designer ODE, growing mode)
  Pogosian & Silvestri 2008, PRD 77, 023503, arXiv:0709.0296          (B0 parametrisation)
  Bellini & Sawicki 2014, JCAP 07 (2014) 050, arXiv:1404.3713         (alpha_i EFT basis)
  Hu & Sawicki 2007, PRD 76, 064004,         arXiv:0705.1158          (f(R) reference model)
  mochi_class, Cataneo et al. 2024,          arXiv:2407.11968         (stable_params basis)
------------------------------------------------------------------------------

Usage:
  # validate against the shipped Mathematica file (run from this directory):
  ../.venv/bin/python make_designer_fr_inputs.py --h 0.6781 --omega-b 0.0223828 \
      --omega-cdm 0.1201075 --N-ur 3.044 --B0 0.01 --validate

  # generate for a new cosmology:
  ../.venv/bin/python make_designer_fr_inputs.py --H0 67 --Omega-b 0.049 \
      --Omega-cdm 0.27 --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
      --B0 0.01 --out-stable designer_stable.dat
"""
import argparse
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from make_cubic_galileon_inputs import Cosmology
from make_hs_fr_inputs import ncdm_trace_and_deriv

P_GROW = (5.0 + np.sqrt(73.0)) / 4.0   # ~3.3860, matter-domination growing mode


def background_splines(cosmo, lna_lo=-13.0, lna_hi=0.30, n=4000):
    """Spline E^2(N), r(N)=R/H0^2 and their N-derivatives on the LCDM bg."""
    lna = np.linspace(lna_lo, lna_hi, n)
    a = np.exp(lna)
    Om = cosmo.Omega_b + cosmo.Omega_cdm
    S0, _ = cosmo.S_and_friends(1.0)
    Omega_L = 1.0 - S0

    E2 = np.empty_like(a)
    r = np.empty_like(a)
    for i, ai in enumerate(a):
        S, _ = cosmo.S_and_friends(ai)            # all non-Lambda densities
        E2[i] = S + Omega_L
        tr, _ = ncdm_trace_and_deriv(cosmo, ai)   # (rho-3p)_ncdm / rho_crit0
        r[i] = 3.0 * Om * ai**-3 + 12.0 * Omega_L + 3.0 * tr

    E2_s = CubicSpline(lna, E2)
    r_s = CubicSpline(lna, r)
    return E2_s, r_s, Omega_L


def designer_solution(cosmo, lna_out, B0, lna_start=-10.0):
    """Integrate the linear designer ODE, normalise to B0, sample on lna_out."""
    E2_s, r_s, Omega_L = background_splines(cosmo)
    dlnH = lambda N: 0.5 * E2_s(N, 1) / E2_s(N)     # dln H / dln a

    def rhs(N, y):
        delta, delta_R, g = y
        r = r_s(N)
        rp = r_s(N, 1)
        gp = -(dlnH(N) + 3.0) * g + (delta_R * r - 2.0 * delta) / (3.0 * E2_s(N))
        return [delta_R * rp, g, gp]

    # growing-mode initial conditions deep in matter domination
    eps = 1.0e-8
    r_i = r_s(lna_start)
    y0 = [-3.0 * r_i * eps / (P_GROW - 3.0), eps, P_GROW * eps]

    sol = solve_ivp(rhs, (lna_start, lna_out[-1]), y0, method="LSODA",
                    rtol=1e-9, atol=1e-30, dense_output=True, max_step=0.05)
    if not sol.success:
        sys.exit(f"designer ODE integration failed: {sol.message}")

    delta, delta_R, g = sol.sol(lna_out)

    # Renormalise the linear solution to the requested B0 FIRST, while f_R is
    # still the (large) raw amplitude. The physical f_R is tiny, so
    # alpha_M = f_R'/(1+f_R) ~ f_R' = g at a=1, hence
    #   B0 = alpha_M(0)/dlnH(0) ~ scale*g_raw(0)/dlnH(0)  =>  scale below.
    g0_raw = np.interp(0.0, lna_out, g)
    scale = B0 * dlnH(0.0) / g0_raw
    delta *= scale
    delta_R *= scale
    g *= scale

    # now f_R is physical (<<1); compute the alphas exactly
    f_R = delta_R
    alpha_M = g / (1.0 + f_R)
    alpha_B = -alpha_M

    Dkin = 1.5 * alpha_B**2
    cs2 = np.ones_like(lna_out)
    aB0 = np.interp(0.0, lna_out, alpha_B)
    return dict(Delta_M2=f_R, alpha_B=alpha_B, alpha_M=alpha_M,
                Dkin=Dkin, cs2=cs2, alpha_B0=aB0, Omega_L=Omega_L)


def validate(cosmo, B0, stable_file):
    lna, dM2_ref, Dkin_ref, cs2_ref = np.loadtxt(stable_file, unpack=True)
    sol = designer_solution(cosmo, lna, B0)
    print(f"validation against\n  {stable_file}")
    print(f"  Omega_Lambda   = {sol['Omega_L']:.10f}")
    print(f"  alpha_B(z=0)   = {sol['alpha_B0']:.10e}   (parameters_smg)")
    print(f"  shipped file parameters_smg (designer_fr.ini) = 4.65007078221003e-3")

    def report(name, mine, ref):
        ok = np.abs(ref) > 1e-30
        rel = np.abs(mine[ok] / ref[ok] - 1.0)
        print(f"  {name:9s}: max|rel.dev| = {rel.max():.3e}   median = {np.median(rel):.3e}")

    report("Delta_M2", sol['Delta_M2'], dM2_ref)
    report("D_kin", sol['Dkin'], Dkin_ref)
    print(f"  cs2      : ref all ones: {np.all(cs2_ref == 1.0)} (ours: 1 identically)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h', type=float)
    p.add_argument('--H0', type=float)
    p.add_argument('--omega-b', type=float)
    p.add_argument('--omega-cdm', type=float)
    p.add_argument('--Omega-b', type=float)
    p.add_argument('--Omega-cdm', type=float)
    p.add_argument('--T-cmb', type=float, default=2.7255)
    p.add_argument('--N-ur', type=float, default=3.044)
    p.add_argument('--deg-ncdm', type=int, default=0)
    p.add_argument('--m-ncdm', type=float, default=0.0)
    p.add_argument('--B0', type=float, default=0.01,
                   help='Compton wavelength parameter today (designer parameter)')
    p.add_argument('--lna-min', type=float, default=-5.0)
    p.add_argument('--lna-max', type=float, default=0.25)
    p.add_argument('--n-points', type=int, default=2101)
    p.add_argument('--out-stable')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--stable-file',
                   default='../stable_params_input/designer_fr_stable_params_B0_0p01.dat')
    args = p.parse_args()

    h = args.h if args.h else (args.H0 / 100.0 if args.H0 else None)
    if h is None:
        sys.exit('need --h or --H0')
    omega_b = args.omega_b if args.omega_b is not None else args.Omega_b * h**2
    omega_cdm = args.omega_cdm if args.omega_cdm is not None else args.Omega_cdm * h**2

    cosmo = Cosmology(h, omega_b, omega_cdm, T_cmb=args.T_cmb, N_ur=args.N_ur,
                      deg_ncdm=args.deg_ncdm, m_ncdm=args.m_ncdm)

    if args.validate:
        validate(cosmo, args.B0, args.stable_file)
        return

    lna = np.linspace(args.lna_min, args.lna_max, args.n_points)
    sol = designer_solution(cosmo, lna, args.B0)
    print(f"Omega_Lambda = {sol['Omega_L']:.10f}")
    print(f"alpha_B(z=0) = {sol['alpha_B0']:.10e}")
    print("ini-file settings:")
    print(f"  parameters_smg  = {sol['alpha_B0']:.12e}")
    print(f"  expansion_model = lcdm    (no expansion_file_name)")

    if args.out_stable:
        np.savetxt(args.out_stable,
                   np.column_stack([lna, sol['Delta_M2'], sol['Dkin'], sol['cs2']]),
                   fmt='%.16e', delimiter='\t')
        print(f"wrote {args.out_stable}")


if __name__ == '__main__':
    main()
