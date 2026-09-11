#!/usr/bin/env python
"""Generate mochi_class stable_params input files for Hu-Sawicki f(R)
gravity -- STAGE 1: leading order in f_R0.

Approximations (good to O(f_R0) ~ 1e-4 relative):
  * background expansion = exact LCDM (+radiation +ncdm); the effective
    dark-energy density is constant, so rho_de(a)/rho_de(0) = 1.
  * high-curvature limit of the Hu-Sawicki function, giving the standard
    power law
        f_R(a) = f_R0 * [ R_0 / R(a) ]^(n+1),   f_R0 < 0
    with the Ricci scalar evaluated on the LCDM background
        R(a)/(3 H0^2) = (Omega_b+Omega_cdm) a^-3 + (rho_nu - 3 p_nu)/rho_crit0
                        + 4 Omega_Lambda        (radiation is traceless).

f(R) structure in the alpha basis (Bellini & Sawicki conventions):
    M*^2 = 1 + f_R          =>  Delta M*^2 = f_R(a)
    alpha_M = dln(1+f_R)/dlna,   alpha_B = -alpha_M,   alpha_K = 0
    D_kin  = (3/2) alpha_B^2
    cs^2   = 1   (exactly)
    parameters_smg (mochi boundary condition) = alpha_B(z=0)

The known stage-1 systematics vs the shipped Mathematica files (which solve
the exact f(R) background): O(f_R0) relative errors, i.e. ~0.1% for
|f_R0| = 1e-4 and negligible for 1e-8, plus a constant-rho_de file that
misses the O(f_R0) structure (max ~7e-4) of the exact background.

Usage:
  # validate against the shipped files (run from scripts/):
  ../.venv/bin/python make_hs_fr_inputs.py --h 0.6781 --omega-b 0.0223828 \
      --omega-cdm 0.1201075 --N-ur 3.044 --fR0 -1e-4 --n 1 --validate

  # generate for a new cosmology:
  ../.venv/bin/python make_hs_fr_inputs.py --H0 67 --Omega-b 0.049 \
      --Omega-cdm 0.27 --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
      --fR0 -1e-4 --n 1 --out-stable hs_stable.dat --out-rho-de hs_rho_de.dat
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import quad

from make_cubic_galileon_inputs import Cosmology


def ncdm_trace_and_deriv(cosmo, a):
    """(rho_nu - 3 p_nu)/rho_crit0 and its dlna-derivative at scale factor a.

    With y = m a / T_ncdm0 and I_K(y) = int x^2/sqrt(x^2+y^2)/(e^x+1) dx:
        Delta(a) = pref * a^-4 * y^2 I_K(y)
        dDelta/dlna = -2 Delta - pref * a^-4 * y^4 I_L(y),
        I_L(y) = int x^2/(x^2+y^2)^(3/2)/(e^x+1) dx.
    """
    if cosmo.deg_ncdm == 0 or cosmo.m_ncdm == 0.0:
        return 0.0, 0.0
    y = cosmo.m_ncdm * a / cosmo.T_ncdm0_eV
    I_K = quad(lambda x: x**2 / np.sqrt(x**2 + y**2) / (np.exp(x) + 1.0),
               0.0, 50.0 + 2.0 * y, limit=200)[0]
    I_L = quad(lambda x: x**2 / (x**2 + y**2)**1.5 / (np.exp(x) + 1.0),
               0.0, 50.0 + 2.0 * y, limit=200)[0]
    delta = cosmo.ncdm_pref * a**-4 * y**2 * I_K
    ddelta = -2.0 * delta - cosmo.ncdm_pref * a**-4 * y**4 * I_L
    return delta, ddelta


def hs_solution(cosmo, lna, fR0, n):
    """Stage-1 Hu-Sawicki stable functions on the lna grid."""
    a = np.exp(lna)
    Om = cosmo.Omega_b + cosmo.Omega_cdm
    S0, _ = cosmo.S_and_friends(1.0)
    Omega_L = 1.0 - S0  # effective Lambda from flatness closure

    delta_nu = np.empty_like(a)
    ddelta_nu = np.empty_like(a)
    for i, ai in enumerate(a):
        delta_nu[i], ddelta_nu[i] = ncdm_trace_and_deriv(cosmo, ai)
    delta_nu0, _ = ncdm_trace_and_deriv(cosmo, 1.0)

    # R(a)/(3 H0^2) and its logarithmic derivative on the LCDM background
    Rhat = Om * a**-3 + delta_nu + 4.0 * Omega_L
    Rhat0 = Om + delta_nu0 + 4.0 * Omega_L
    dlnR = (-3.0 * Om * a**-3 + ddelta_nu) / Rhat

    f_R = fR0 * (Rhat0 / Rhat)**(n + 1)
    alpha_M = -(n + 1) * f_R * dlnR / (1.0 + f_R)
    alpha_B = -alpha_M
    Dkin = 1.5 * alpha_B**2
    cs2 = np.ones_like(a)

    return dict(a=a, Omega_L=Omega_L, Delta_M2=f_R, alpha_B=alpha_B,
                Dkin=Dkin, cs2=cs2)


def validate(cosmo, fR0, n, indir):
    tag_f = {1e-4: '1em4', 1e-8: '1em8'}.get(abs(fR0))
    if tag_f is None:
        sys.exit(f'no shipped Mathematica file for |fR0| = {abs(fR0)}')
    stable_file = os.path.join(indir, f'hs_fr_stable_params_fr0_{tag_f}_n_{n}_mathematica.dat')
    rho_de_file = os.path.join(indir, f'rho_de_stable_hs_fr_n_{n}_fr0_{tag_f}_mathematica.dat')

    lna_s, dM2_ref, Dkin_ref, cs2_ref = np.loadtxt(stable_file, unpack=True)
    lna_r, rho_ref = np.loadtxt(rho_de_file, unpack=True)
    sol = hs_solution(cosmo, lna_s, fR0, n)

    print(f"validation against\n  {stable_file}\n  {rho_de_file}")
    print(f"  Omega_Lambda_eff = {sol['Omega_L']:.10f}")
    aB0 = hs_solution(cosmo, np.array([0.0]), fR0, n)['alpha_B'][0]
    print(f"  alpha_B(z=0)     = {aB0:.10e}   (parameters_smg)")

    def report(name, mine, ref):
        rel = np.abs(mine / ref - 1.0)
        print(f"  {name:9s}: max|rel.dev| = {rel.max():.3e}   median = {np.median(rel):.3e}")

    report("Delta_M2", sol['Delta_M2'], dM2_ref)
    report("D_kin", sol['Dkin'], Dkin_ref)
    print(f"  cs2      : ref is 1 everywhere: {np.all(cs2_ref == 1.0)} (ours: 1 identically)")
    devr = np.abs(rho_ref - 1.0)
    print(f"  rho_de   : stage-1 is constant 1; exact-background file deviates "
          f"from 1 by max {devr.max():.3e} (known stage-1 limitation)")


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
    p.add_argument('--fR0', type=float, required=True,
                   help='f_R today, negative (e.g. -1e-4)')
    p.add_argument('--n', type=int, default=1, help='Hu-Sawicki exponent')
    p.add_argument('--lna-min', type=float, default=-5.0)
    p.add_argument('--lna-max-stable', type=float, default=0.25)
    p.add_argument('--n-points', type=int, default=2101)
    p.add_argument('--out-stable')
    p.add_argument('--out-rho-de')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--input-dir', default='../stable_params_input')
    args = p.parse_args()

    if args.fR0 >= 0:
        sys.exit('fR0 must be negative (e.g. --fR0 -1e-4)')
    h = args.h if args.h else (args.H0 / 100.0 if args.H0 else None)
    if h is None:
        sys.exit('need --h or --H0')
    omega_b = args.omega_b if args.omega_b is not None else args.Omega_b * h**2
    omega_cdm = args.omega_cdm if args.omega_cdm is not None else args.Omega_cdm * h**2

    cosmo = Cosmology(h, omega_b, omega_cdm, T_cmb=args.T_cmb, N_ur=args.N_ur,
                      deg_ncdm=args.deg_ncdm, m_ncdm=args.m_ncdm)

    if args.validate:
        validate(cosmo, args.fR0, args.n, args.input_dir)
        return

    lna = np.linspace(args.lna_min, args.lna_max_stable, args.n_points)
    sol = hs_solution(cosmo, lna, args.fR0, args.n)
    aB0 = hs_solution(cosmo, np.array([0.0]), args.fR0, args.n)['alpha_B'][0]
    print(f"Omega_Lambda_eff = {sol['Omega_L']:.10f}")
    print(f"alpha_B(z=0)     = {aB0:.10e}")
    print("ini-file settings:")
    print(f"  parameters_smg = {aB0:.12e}")
    print("  expansion_model = rho_de with the constant file below "
          "(or simply expansion_model = lcdm)")

    if args.out_stable:
        np.savetxt(args.out_stable,
                   np.column_stack([lna, sol['Delta_M2'], sol['Dkin'], sol['cs2']]),
                   fmt='%.16e', delimiter='\t')
        print(f"wrote {args.out_stable}")
    if args.out_rho_de:
        lna_rho = lna[lna <= 0.0]
        if lna_rho[-1] != 0.0:
            lna_rho = np.append(lna_rho, 0.0)
        np.savetxt(args.out_rho_de,
                   np.column_stack([lna_rho, np.ones_like(lna_rho)]),
                   fmt='%.16e', delimiter='\t')
        print(f"wrote {args.out_rho_de} (stage 1: constant; exact O(f_R0) "
              "background structure requires stage 2)")


if __name__ == '__main__':
    main()
