# -*- coding: iso-8859-1 -*-

# ==================================================
# 	import modules
# ==================================================

import os
import sys

import inspect

import evaluation as ev
import evaluation.simpleplot as sp
import latextable as lt

# calc with uncertainties and arrays of uncertainties [[val, std], ...]
import uncertainties as uc
import uncertainties.unumpy as unp

# calc with arrays
import numpy as np

# plot engine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# for fitting curves
from scipy import optimize

# constants
import scipy.constants as const

# ==================================================
# 	settings
# ==================================================

# change path to script directory
os.chdir(sys.path[0])

sp.params["text.latex.preamble"] = sp.tex_fouriernc_preamble
plt.rcParams.update(sp.params)

ev.use_unitpkg("siunitx")

# ==================================================
# 	example
# ==================================================

"""
# Konstanten
c = const.c

# Geradenfunktion
def Gerade(x, m, b):
    return m*x + b

# Geschwindigkeit
def v(s, t):
    return s / t

# ===== Daten einlesen =============================

s, t = ev.get_data("messwerte.txt", unpack=True, index=0)

# ===== Berechnungen ===============================

# berechne Geschwindigkeiten in array
v_arr = v(s, t)

# mittlere Geschwindigkeit mit Fehler
v_uc = ev.get_uncert(v_arr)

# Geschwindigkeit als LateX-Code
v_tex = ev.to_tex(v_uc, unit="{m}{s}")

# linerare Ausgleichrechnung
val, cov = optimize.curve_fit(Gerade, t, s)
std = ev.get_std(cov)

# oder:

val, std = ev.fit(Gerade, t, s)

# latex-Gleichung der linearen Regression
lin_reg = ev.equation_linReg(
        val,
        std,
        unit=["{s}{m}", "{m}"],
        funcname="v(t)"
)

print_tex(ev.latexEq(lin_reg))

# ===== plot =======================================

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(t, s, linestyle = 'none', marker = '+', label = 'Messwerte')
lim = ax1.get_xlim()
x = np.linspace(lim[0], lim[1], 1000)
ax1.plot(x, Gerade(x, val[0], val[1]), label="Fit")

ax1.set_xlabel(r'$t / \unit{s}$')
ax1.set_ylabel(r'$s / \unit{m}$')

ax1.legend(loc = 'best')
ax1 = ev.plot_layout(ax1)

fig1.tight_layout()

# oder mit simpleplot:

p = sp.figure([r'$s / \unit{m}$', r'$s / \unit{m}$'])
p.add_plot(t, s, label="Messwerte")
p.add_plot(Gerade, val[0], val[1], label="Fit")
p.plot()

plt.show()

# ===== Tabellen ===================================

# see numpy doc! -> very useful function to put array's together!!!
M = np.concatenate([A, B, C])

# LateX-Tabelle erzeugen
t = lt.latextable(
    [t, v_arr],
    "file",
    alignment = 'CC',
    formatcolumn = '%.3f',
    unpack = True,
    comma = True,
    tableoption = 'ht',
    header = 	[r'$t / \unit{s}$',
                    r'$v / \unitfrac{m}{s}$'],
    caption = r'Just an amazing caption.'
)

# Tabellen ggf. zu minipage zsuammenfassen
m = lt.minipage([t, t], pageWidth = r'0.3\textwidth')

# ===== Daten/Plots speichern ======================

# tex schreiben
ev.write('file', v_tex)

# plots speichern
fig1.savefig('file.pdf')

"""

###############################################################################
#                              Brückenschaltung                               #
###############################################################################


def Gerade(x, m, b):
    return m*x + b

# ====[ 100 ohm ]===========================================

I_100, U_100 = np.loadtxt("messwerte/brueke_100.txt", unpack=True)

I_100_val, U_100_cov = optimize.curve_fit(Gerade, I_100, U_100)
I_100_std = ev.get_std(U_100_cov)

I_100_tex = ev.latexEnv(
    ev.tex_linreg(
        "U(I)",
        I_100_val,
        I_100_std,
        unit=[r"\milli\volt/\milli\ampere", r"\milli\volt"]
    )
)

R_100 = uc.ufloat(I_100_val[0], I_100_std[0])
R_100_tex = ev.tex_eq(R_100, name="R", unit="\ohm")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(
    I_100,
    U_100,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

fig_lim = ax.get_xlim()
fig_I = np.linspace(fig_lim[0], fig_lim[1], 1000)
ax.plot(
    fig_I,
    Gerade(fig_I, I_100_val[0], I_100_val[1]),
    color='k',
    label="Fit"
)

ax.annotate(
    I_100_tex,
    (0.4, 0.1),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.annotate(
    R_100_tex,
    (0.4, 0.2),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.set_xlabel(r'$I/\si{mA}$')
ax.set_ylabel(r'$U/\si{mV}$')
ax.set_title("Widerstand $R=\SI{100}{\ohm}", usetex=True)

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig.tight_layout()
fig.savefig("plots/brueken_100.pdf")


# ====[ 1k ohm ]============================================

I_1k, U_1k = np.loadtxt("messwerte/brueke_1k.txt", unpack=True)

I_1k_val, U_1k_cov = optimize.curve_fit(Gerade, I_1k, U_1k)
I_1k_std = ev.get_std(U_1k_cov)

R_1k = uc.ufloat(I_1k_val[0], I_1k_std[0])
R_1k_tex = ev.tex_eq(R_1k * 1e3, name="R", unit="\ohm")

fig = plt.figure()
ax = fig.add_subplot(111)

I_1k_tex = ev.latexEnv(
    ev.tex_linreg(
        "U(I)",
        I_1k_val,
        I_1k_std,
        unit=[r"\volt/\milli\ampere", r"\volt"]
    )
)

ax.plot(
    I_1k,
    U_1k,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

fig_lim = ax.get_xlim()
fig_I = np.linspace(fig_lim[0], fig_lim[1], 1000)
ax.plot(
    fig_I,
    Gerade(fig_I, I_1k_val[0], I_1k_val[1]),
    color='k',
    label="Fit"
)

ax.annotate(
    I_1k_tex,
    (0.4, 0.1),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.annotate(
    R_1k_tex,
    (0.4, 0.2),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.set_xlabel(r'$I/\si{mA}$')
ax.set_ylabel(r'$U/\si{V}$')
ax.set_title(r"Widerstand $R=\SI{1}{\kilo\ohm}$", usetex=True)

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig.tight_layout()
fig.savefig("plots/brueken_1k.pdf")

# ====[ 100k ohm ]==========================================

I_100k, U_100k = np.loadtxt("messwerte/brueke_100k.txt", unpack=True)

I_100k_val, U_100k_cov = optimize.curve_fit(Gerade, I_100k, U_100k)
I_100k_std = ev.get_std(U_100k_cov)

I_100k_tex = ev.latexEnv(
    ev.tex_linreg(
        "U(I)",
        I_100k_val,
        I_100k_std,
        unit=[r"\volt/\milli\ampere", r"\volt"]
    )
)

R_100k = uc.ufloat(I_100k_val[0], I_100k_std[0])
R_100k_tex = ev.tex_eq(R_100k, name="R", unit="\kilo\ohm")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(
    I_100k,
    U_100k,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

fig_lim = ax.get_xlim()
fig_I = np.linspace(fig_lim[0], fig_lim[1], 1000)
ax.plot(
    fig_I,
    Gerade(fig_I, I_100k_val[0], I_100k_val[1]),
    color='k',
    label="Fit"
)

ax.annotate(
    I_100k_tex,
    (0.4, 0.1),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.annotate(
    R_100k_tex,
    (0.4, 0.2),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.set_xlabel(r'$I/\si{mA}$')
ax.set_ylabel(r'$U/\si{V}$')
ax.set_title(r"Widerstand $R=\SI{100}{\kilo\ohm}$", usetex=True)

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig.tight_layout()
fig.savefig("plots/brueken_100k.pdf")

# plt.show()
