    #!/usr/bin/env python
"""Generate mochi_class stable_params + rho_de input files for the
self-accelerating cubic covariant Galileon on its tracker solution,
consistently with a user-chosen cosmology.

Physics (see Cataneo et al., arXiv:2407.11968, Sec. 2):
  On the tracker (H phi_dot = const, self-accelerating branch c2 = -1)
    alpha_K = 3 alpha_B = 6 Omega_phi0 / E^4(a),   M*^2 = 1,  alpha_M = alpha_T = 0
  and the scalar field energy density obeys rho_phi propto 1/H^2, so the
  Friedmann equation closes algebraically:
    E^4 - S(a) E^2 - Omega_phi0 = 0,   S(a) = sum_i Omega_i(a)  (all non-smg)
    => E^2 = [S + sqrt(S^2 + 4 Omega_phi0)]/2 ,  rho_de(a)/rho_de(0) = 1/E^2(a)
  The remaining stable-basis functions follow from the standard Horndeski
  expressions (alpha_M = alpha_T = 0, M*^2 = 1):
    D_kin = alpha_K + 3/2 alpha_B^2
    cs^2 D_kin = (2 - alpha_B)(-Hdot/H^2 + alpha_B/2) + dalpha_B/dlna
                 - 3 (rho_m + p_m)/(M_Pl^2 H^2 ... )   [in Omega units: -3 (rhat+phat)/E^2]
  The boundary condition mochi_class needs (parameters_smg) is
    alpha_B(z=0) = 2 Omega_phi0 .

Everything is determined by {h or H0, omega_b, omega_cdm (or fractions),
T_cmb, N_ur, ncdm masses}; no Mathematica required.

Usage examples:
  # cosmology of the shipped Mathematica files (notebooks/stable_params.ipynb):
  python make_cubic_galileon_inputs.py --h 0.6781 --omega-b 0.0223828 \
      --omega-cdm 0.1201075 --N-ur 3.044 --validate

  # thesis cosmology, write new input files:
  python make_cubic_galileon_inputs.py --H0 67 --Omega-b 0.049 --Omega-cdm 0.27 \
      --N-ur 0 --deg-ncdm 3 --m-ncdm 0.019333333333333334 \
      --out-stable cgal_stable.dat --out-rho-de cgal_rho_de.dat
"""
import argparse
import sys

import numpy as np
from scipy.integrate import quad

KB_EV_PER_K = 8.617333262e-5   # Boltzmann constant [eV/K]
XI_NCDM_DEFAULT = 0.71611      # CLASS default T_ncdm/T_gamma


def omega_gamma_h2(T_cmb):
    """Photon density parameter * h^2 (CLASS input.c convention)."""
    return 4.48131635e-7 * T_cmb**4


def fd_integrals(y):
    """Fermi-Dirac integrals for one ncdm species.

    rho ~ I_rho(y) = int x^2 sqrt(x^2+y^2)/(e^x+1) dx
    p   ~ I_p(y)   = (1/3) int x^4 / sqrt(x^2+y^2) /(e^x+1) dx
    with y = m a / T_ncdm0. Massless limit: I_rho = 3 I_p = 7 pi^4/120.
    """
    I_rho = quad(lambda x: x**2 * np.sqrt(x**2 + y**2) / (np.exp(x) + 1.0),
                 0.0, 50.0 + 2.0 * y, limit=200)[0]
    I_p = quad(lambda x: x**4 / np.sqrt(x**2 + y**2) / (np.exp(x) + 1.0) / 3.0,
               0.0, 50.0 + 2.0 * y, limit=200)[0]
    return I_rho, I_p


class Cosmology:
    """Non-smg background densities in units of rho_crit,0 (Omega units)."""

    def __init__(self, h, omega_b, omega_cdm, T_cmb=2.7255, N_ur=3.044,
                 deg_ncdm=0, m_ncdm=0.0, xi_ncdm=XI_NCDM_DEFAULT):
        self.h = h
        self.Omega_b = omega_b / h**2
        self.Omega_cdm = omega_cdm / h**2
        self.Omega_g = omega_gamma_h2(T_cmb) / h**2
        self.Omega_ur = N_ur * 7.0 / 8.0 * (4.0 / 11.0)**(4.0 / 3.0) * self.Omega_g
        self.deg_ncdm = deg_ncdm
        self.m_ncdm = m_ncdm
        self.xi = xi_ncdm
        self.T_ncdm0_eV = xi_ncdm * T_cmb * KB_EV_PER_K
        # prefactor such that rho_ncdm_hat(a) = pref * a^-4 * I_rho(y(a))
        self.ncdm_pref = deg_ncdm * self.Omega_g * 15.0 / np.pi**4 * xi_ncdm**4

    def ncdm_rho_p(self, a):
        """(rho_hat, p_hat) of all ncdm species at scale factor a."""
        if self.deg_ncdm == 0 or self.m_ncdm == 0.0:
            rho = self.ncdm_pref * 7.0 * np.pi**4 / 120.0 * a**-4
            return rho, rho / 3.0
        y = self.m_ncdm * a / self.T_ncdm0_eV
        I_rho, I_p = fd_integrals(y)
        return (self.ncdm_pref * a**-4 * I_rho,
                self.ncdm_pref * a**-4 * I_p)

    def S_and_friends(self, a):
        """Return S(a) = sum rho_hat_i, and sum (rho_hat+p_hat)_i."""
        rho_nu, p_nu = self.ncdm_rho_p(a)
        S = ((self.Omega_b + self.Omega_cdm) * a**-3
             + (self.Omega_g + self.Omega_ur) * a**-4 + rho_nu)
        rho_plus_p = ((self.Omega_b + self.Omega_cdm) * a**-3
                      + 4.0 / 3.0 * (self.Omega_g + self.Omega_ur) * a**-4
                      + rho_nu + p_nu)
        return S, rho_plus_p


def tracker_solution(cosmo, lna):
    """All cubic-Galileon tracker quantities on the lna grid."""
    a = np.exp(lna)
    S = np.empty_like(a)
    rho_plus_p = np.empty_like(a)
    for i, ai in enumerate(a):
        S[i], rho_plus_p[i] = cosmo.S_and_friends(ai)

    S0, _ = cosmo.S_and_friends(1.0)
    Omega_phi0 = 1.0 - S0

    E2 = 0.5 * (S + np.sqrt(S**2 + 4.0 * Omega_phi0))
    # d ln E^2 / d lna from the closure (continuity: dS/dlna = -3 sum(rho+p))
    Sprime = -3.0 * rho_plus_p
    dlnE2 = Sprime / (2.0 * E2 - S)

    alpha_B = 2.0 * Omega_phi0 / E2**2
    alpha_K = 3.0 * alpha_B
    Dkin = alpha_K + 1.5 * alpha_B**2
    dalpha_B = -2.0 * alpha_B * dlnE2

    minus_Hdot_over_H2 = -0.5 * dlnE2
    cs2 = ((2.0 - alpha_B) * (minus_Hdot_over_H2 + 0.5 * alpha_B)
           + dalpha_B - 3.0 * rho_plus_p / E2) / Dkin

    return dict(a=a, E2=E2, Omega_phi0=Omega_phi0, alpha_B=alpha_B,
                alpha_K=alpha_K, Dkin=Dkin, cs2=cs2, rho_de_shape=1.0 / E2)


def validate(cosmo, stable_file, rho_de_file):
    lna_s, dM2_ref, Dkin_ref, cs2_ref = np.loadtxt(stable_file, unpack=True)
    lna_r, rho_ref = np.loadtxt(rho_de_file, unpack=True)

    sol_s = tracker_solution(cosmo, lna_s)
    sol_r = tracker_solution(cosmo, lna_r)
    rho = sol_r['rho_de_shape'] / tracker_solution(cosmo, np.array([0.0]))['rho_de_shape'][0]

    def report(name, mine, ref):
        rel = np.abs(mine / ref - 1.0)
        print(f"  {name:8s}: max|rel.dev| = {rel.max():.3e}   median = {np.median(rel):.3e}")

    print(f"validation against\n  {stable_file}\n  {rho_de_file}")
    print(f"  Omega_phi0      = {sol_s['Omega_phi0']:.16f}")
    print(f"  alpha_B(z=0)    = {2*sol_s['Omega_phi0']:.16f}   (parameters_smg)")
    print(f"  Delta_M2 ref    : max|ref| = {np.abs(dM2_ref).max():.1e} (should be 0; ours is 0 identically)")
    report("rho_de", rho, rho_ref)
    report("D_kin", sol_s['Dkin'], Dkin_ref)
    report("cs2", sol_s['cs2'], cs2_ref)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h', type=float, help='reduced Hubble parameter')
    p.add_argument('--H0', type=float, help='Hubble constant [km/s/Mpc] (alternative to --h)')
    p.add_argument('--omega-b', type=float, help='physical baryon density omega_b = Omega_b h^2')
    p.add_argument('--omega-cdm', type=float, help='physical cdm density')
    p.add_argument('--Omega-b', type=float, help='baryon fraction (alternative to --omega-b)')
    p.add_argument('--Omega-cdm', type=float, help='cdm fraction')
    p.add_argument('--T-cmb', type=float, default=2.7255)
    p.add_argument('--N-ur', type=float, default=3.044)
    p.add_argument('--deg-ncdm', type=int, default=0, help='number of degenerate massive neutrinos')
    p.add_argument('--m-ncdm', type=float, default=0.0, help='mass per ncdm species [eV]')
    p.add_argument('--lna-min', type=float, default=-5.0)
    p.add_argument('--lna-max-stable', type=float, default=0.25)
    p.add_argument('--n-points', type=int, default=2101)
    p.add_argument('--out-stable', help='output stable_params file (lna, Delta_M2, D_kin, cs2)')
    p.add_argument('--out-rho-de', help='output rho_de file (lna, rho_de/rho_de0)')
    p.add_argument('--validate', action='store_true',
                   help='compare against the shipped Mathematica files instead of writing output')
    args = p.parse_args()

    h = args.h if args.h else (args.H0 / 100.0 if args.H0 else None)
    if h is None:
        sys.exit('need --h or --H0')
    omega_b = args.omega_b if args.omega_b is not None else args.Omega_b * h**2
    omega_cdm = args.omega_cdm if args.omega_cdm is not None else args.Omega_cdm * h**2

    cosmo = Cosmology(h, omega_b, omega_cdm, T_cmb=args.T_cmb, N_ur=args.N_ur,
                      deg_ncdm=args.deg_ncdm, m_ncdm=args.m_ncdm)

    if args.validate:
        validate(cosmo,
                 '../stable_params_input/cubic_galileon_stable_params_mathematica.dat',
                 '../stable_params_input/rho_de_stable_cubic_galileon_mathematica.dat')
        return

    lna_stable = np.linspace(args.lna_min, args.lna_max_stable, args.n_points)
    sol = tracker_solution(cosmo, lna_stable)
    print(f"Omega_phi0   = {sol['Omega_phi0']:.16f}")
    print(f"alpha_B(z=0) = {2*sol['Omega_phi0']:.16f}")
    print("ini-file settings:")
    print(f"  parameters_smg = {2*sol['Omega_phi0']:.15f}")
    print(f"  expansion_model = rho_de   (expansion_smg = Omega_phi0 guess, tuned by shooting)")

    if args.out_stable:
        np.savetxt(args.out_stable,
                   np.column_stack([lna_stable, np.zeros_like(lna_stable),
                                    sol['Dkin'], sol['cs2']]),
                   fmt='%.16e', delimiter='\t')
        print(f"wrote {args.out_stable}")
    if args.out_rho_de:
        lna_rho = lna_stable[lna_stable <= 0.0]
        if lna_rho[-1] != 0.0:
            lna_rho = np.append(lna_rho, 0.0)
        sol_r = tracker_solution(cosmo, lna_rho)
        shape = sol_r['rho_de_shape'] / sol_r['rho_de_shape'][-1]
        np.savetxt(args.out_rho_de, np.column_stack([lna_rho, shape]),
                   fmt='%.16e', delimiter='\t')
        print(f"wrote {args.out_rho_de}")


if __name__ == '__main__':
    main()
