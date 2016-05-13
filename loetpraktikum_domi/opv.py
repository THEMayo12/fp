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
#                              Umkehrverstärker                               #
###############################################################################


def Gerade(x, m, b):
    return m*x + b

# ====[ 100 ohm ]===========================================

U_E, U_A = np.loadtxt("messwerte/umkehrverstaerker.txt", unpack=True)

U_E_fit = U_E[(U_E > -34.8) & (U_E < 54.4)]
U_A_fit = U_A[(U_E > -34.8) & (U_E < 54.4)]

val, cov = optimize.curve_fit(Gerade, U_E_fit, U_A_fit)
std = ev.get_std(cov)

U_A_tex = ev.latexEnv(
    ev.tex_linreg(
        "U_A(U_E)",
        val,
        std,
        unit=[r"\volt/\milli\volt", r"\volt"]
    )
)

# V_tex = ev.latexEnv("V = {}".format(val[0] * 1e-3))
V = uc.ufloat(val[0], std[0])
V_tex = ev.tex_eq(V * 1e3, name="V")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(
    U_E,
    U_A,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

fig_lim = ax.get_xlim()
fig_U_E = np.linspace(fig_lim[0], fig_lim[1], 1000)
ax.plot(
    fig_U_E,
    Gerade(fig_U_E, val[0], val[1]),
    color='k',
    label="Fit"
)

ax.set_xlabel(r'$U_E/\si{\milli\volt}$')
ax.set_ylabel(r'$U_A/\si{\volt}$')
ax.set_title(r"Umkehrverst\"arker", usetex=True)

ax.legend(loc='best')
ax = ev.plot_layout(ax)

ax.axhline(0, color='k')
ax.axvline(0, ymin=0.25, color='k')

ax.annotate(
    U_A_tex,
    (0.1, 0.1),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

ax.annotate(
    V_tex,
    (0.1, 0.2),
    xycoords='axes fraction',
    textcoords='axes fraction',
    usetex=True
)

fig.tight_layout()
fig.savefig("plots/opv.pdf")
# plt.show()
