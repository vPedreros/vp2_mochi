#!/usr/bin/env python
"""Thesis-style plots for a standard-CLASS LCDM cosmology:
  (1) lensed CMB C_ell^TT
  (2) matter power spectrum P(k, z=0): linear vs non-linear (HALOFIT)

Reuses the thesis plotting style (scienceplots 'science'+'vpedre', set_size).
Run from this directory with the repo venv:
  ../.venv/bin/python thesis_style_plots.py
"""
import os
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' style)
from classy import Class

# ----------------------------------------------------------------------------
# thesis plotting style
# ----------------------------------------------------------------------------
plt.style.use(['science', 'no-latex', 'vpedre'])
LINESTYLES = plt.rcParams['axes.prop_cycle'].by_key().get('linestyle', ['-', '--', ':', '-.'])
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({
    "axes.labelsize": 12, "font.size": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

thesis_width = 472.03123  # pt


def set_size(width, fraction=1, subplots=(1, 1)):
    """Figure dimensions (inches) to avoid scaling in LaTeX."""
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


fig_dims = set_size(thesis_width)

# ----------------------------------------------------------------------------
# standard CLASS LCDM cosmology (thesis base cosmology, no smg)
# ----------------------------------------------------------------------------
T_cmb = 2.7255
lcdm = {
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'l_max_scalars': 3000,
    'non linear': 'halofit',
    'P_k_max_h/Mpc': 50.,
    'z_max_pk': 5.,
    'H0': 67,
    'Omega_b': 0.049,
    'Omega_cdm': 0.27,
    'A_s': 2.1e-9,
    'n_s': 0.96,
    'N_ur': 0,
    'N_ncdm': 1,
    'deg_ncdm': 3,
    'm_ncdm': 0.058 / 3,
    'T_cmb': T_cmb,
}

print('Computing standard CLASS LCDM ...', end=' ', flush=True)
cosmo = Class()
cosmo.set(lcdm)
cosmo.compute()
print('done')

h = cosmo.h()
OUT = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# (1) lensed CMB C_ell^TT
# ----------------------------------------------------------------------------
cl = cosmo.lensed_cl(2500)
ell = cl['ell'][2:]
dl_factor = (T_cmb * 1e6) ** 2 * ell * (ell + 1) / (2 * pi)   # -> muK^2
Dl_TT = dl_factor * cl['tt'][2:]

fig, ax = plt.subplots(figsize=(fig_dims[0] * 0.62, fig_dims[1]))
ax.plot(ell, Dl_TT, ls=LINESTYLES[0], color=COLORS[0], label=r'\texttt{CLASS} $\Lambda$CDM')
ax.set_xlim(2, 2500)
ax.set_xlabel(r'Multipolo $\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi\ \ (\mu\mathrm{K}^2)$')
ax.set_title('Espectro de potencia angular de temperatura del CMB')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cltt_lcdm.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'cltt_lcdm.png'), dpi=160, bbox_inches='tight')
print('wrote cltt_lcdm.{pdf,png}')

# ----------------------------------------------------------------------------
# (2) matter power spectrum: linear vs non-linear (HALOFIT)
# ----------------------------------------------------------------------------
k_h = np.logspace(-4, np.log10(40), 600)         # k in h/Mpc (extended to large scales)
k = k_h * h                                       # 1/Mpc for classy
P_lin = np.array([cosmo.pk_lin(ki, 0.0) for ki in k]) * h ** 3   # (Mpc/h)^3
P_nl = np.array([cosmo.pk(ki, 0.0) for ki in k]) * h ** 3

fig, ax1 = plt.subplots(figsize=(fig_dims[0] * 0.62, fig_dims[1]))
ax1.loglog(k_h, P_lin, ls=LINESTYLES[1], color=COLORS[1], label='lineal')
ax1.loglog(k_h, P_nl, ls=LINESTYLES[0], color=COLORS[0], label='no lineal (HALOFIT)')
ax1.set_xlim(k_h.min(), k_h.max())
ax1.set_xlabel(r'$k\ \ (h/\mathrm{Mpc})$')
ax1.set_ylabel(r'$P(k,\,z=0)\ \ (\mathrm{Mpc}/h)^3$')
ax1.set_title('Espectro de potencias de materia')
ax1.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pk_lcdm.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'pk_lcdm.png'), dpi=160, bbox_inches='tight')
print('wrote pk_lcdm.{pdf,png}')

cosmo.struct_cleanup()
cosmo.empty()
