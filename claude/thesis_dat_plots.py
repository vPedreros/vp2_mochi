#!/usr/bin/env python
"""Thesis-style plots of the stable_params INPUT .dat files used in
thesis_plots: the stable basis functions {Delta M*^2, D_kin, cs^2} and the
rho_de(a) expansion, for the three mochi models

  mochi LCDM       -> gr.dat                          (w0wa, w=-1 -> rho_de=const)
  f(R) Hu-Sawicki  -> hs_fr_stable_params_fr0_1em4_n_1 + rho_de_stable_hs_fr_...
  Cubic Galileon   -> cubic_galileon_stable_params    + rho_de_stable_cubic_galileon

Run from this directory:  ../.venv/bin/python thesis_dat_plots.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

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

thesis_width = 472.03123


def set_size(width, fraction=1, subplots=(1, 1)):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** .5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


fig_dims = set_size(thesis_width)

HERE = os.path.dirname(__file__)
SPI = os.path.join(HERE, '..', 'stable_params_input')
OUT = os.path.join(HERE, 'plots')
os.makedirs(OUT, exist_ok=True)

# (label, stable_params file, rho_de file or None, colour, linestyle)
MODELS = [
    (r'\texttt{mochi} $\Lambda$CDM', 'gr.dat', None, COLORS[0], LINESTYLES[0]),
    (r'$f(R)$ Hu-Sawicki', 'hs_fr_stable_params_fr0_1em4_n_1_mathematica.dat',
     'rho_de_stable_hs_fr_n_1_fr0_1em4_mathematica.dat', COLORS[1], LINESTYLES[1]),
    (r'Cubic Galileon', 'cubic_galileon_stable_params_mathematica.dat',
     'rho_de_stable_cubic_galileon_mathematica.dat', COLORS[2], LINESTYLES[2]),
]

# ----------------------------------------------------------------------------
# (1) stable basis functions  Delta M*^2, D_kin, cs^2  vs scale factor
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(fig_dims[0] * 1.9, fig_dims[1] * 0.95),
                         sharex=True)
for label, sfile, _rfile, color, lsty in MODELS:
    lna, dM2, Dkin, cs2 = np.loadtxt(os.path.join(SPI, sfile), unpack=True)
    a = np.exp(lna)
    axes[0].plot(a, dM2, color=color, ls=lsty, label=label)
    axes[1].plot(a, Dkin, color=color, ls=lsty)
    axes[2].plot(a, cs2, color=color, ls=lsty)

axes[0].set_ylabel(r'$\Delta M_*^2$')
axes[0].set_title(r'$\Delta M_*^2$', fontsize=11)
axes[1].set_yscale('log')
axes[1].set_ylim(1e-14, 3e1)
axes[1].set_title(r'$D_{\rm kin}$', fontsize=11)
axes[2].set_ylim(0, 1.5)
axes[2].set_title(r'$c_s^2$', fontsize=11)
for ax in axes:
    ax.set_xscale('log')
    ax.set_xlim(np.exp(-5), 1.0)
    ax.set_xlabel(r'scale factor $a$')
axes[0].legend(loc='lower left')
fig.suptitle(r'Stable-basis input functions', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'stable_params_inputs.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'stable_params_inputs.png'), dpi=160, bbox_inches='tight')
print('wrote stable_params_inputs.{pdf,png}')

# ----------------------------------------------------------------------------
# (2) rho_de(a)/rho_de(0) expansion input
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(fig_dims[0] * 0.62, fig_dims[1]))
for label, _sfile, rfile, color, lsty in MODELS:
    if rfile is None:                      # LCDM: w=-1 -> rho_de = const
        a = np.logspace(np.log10(np.exp(-5)), 0, 200)
        ax.plot(a, np.ones_like(a), color=color, ls=lsty, label=label)
    else:
        lna, rho = np.loadtxt(os.path.join(SPI, rfile), unpack=True)
        ax.plot(np.exp(lna), rho, color=color, ls=lsty, label=label)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(np.exp(-5), 1.0)
ax.set_xlabel(r'scale factor $a$')
ax.set_ylabel(r'$\rho_{\rm de}(a)/\rho_{\rm de}(0)$')
ax.set_title('Dark energy density (expansion input)')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rho_de_inputs.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'rho_de_inputs.png'), dpi=160, bbox_inches='tight')
print('wrote rho_de_inputs.{pdf,png}')
