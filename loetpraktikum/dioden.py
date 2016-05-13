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
#                               Diodenschaltung                               #
###############################################################################

# ====[ dicke Diode ]=======================================

I, U = np.loadtxt("messwerte/diode.txt", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(
    U,
    I,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$U/\si{\volt}$')
ax.set_ylabel(r'$I/\si{\milli\ampere}$')
ax.set_title("Diode")

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig.tight_layout()
fig.savefig("plots/diode.pdf")

# ====[ Zener-Diode ]=======================================

I_zener, U_zener = np.loadtxt("messwerte/zenerdiode.txt", unpack=True)

fig_zener = plt.figure()
ax = fig_zener.add_subplot(111)

ax.plot(
    U_zener,
    I_zener,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$U/\si{\volt}$')
ax.set_ylabel(r'$I/\si{\milli\ampere}$')
ax.set_title("Zener-Diode")

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_zener.tight_layout()
fig_zener.savefig("plots/zenerdiode.pdf")
