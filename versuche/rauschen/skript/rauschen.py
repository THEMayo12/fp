# -*- coding: iso-8859-1 -*-

# ==================================================
# 	import modules
# ==================================================

import sys
import os

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
# 	function to print equations with
# 	matplotlib
# ==================================================


def show(x):
    assert isinstance(x, str)
    print x + " = "
    print eval(x)
    print "\n"


def print_tex(s):
    assert isinstance(s, str)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.0, 0.5,
            s.replace('\t', ' ').replace('\n', ' ').replace('%', ''),
            color='k', size='x-large'
            )
    plt.show()

# ==================================================
#       example
# ==================================================

"""
# Konstanten
c = const.c

# Geradenfunktion
def G(x, m, b):
    return m*x + b

# Geschwindigkeit
def v(s, t):
    return s / t

# ===== Daten einlesen =============================

s, t = ev.get_data("messwerte.txt", unpack=True, index=0)

# oder, wenn nur eine Tabelle pro Datei

s, t = np.loadtxt("messwerte.txt", unpack=True)

# ===== Berechnungen ===============================

# berechne Geschwindigkeiten in array
v_arr = v(s, t)

# mittlere Geschwindigkeit mit Fehler
v_uc = ev.get_uncert(v_arr)

# Geschwindigkeit als LateX-Code
v_tex = ev.tex_eq(v_uc, form="({:L})", unit="\meter\per\second")

# linerare Ausgleichrechnung
val, cov = optimize.curve_fit(G, t, s)
std = ev.get_std(cov)

# oder:

val, std = ev.fit(G, t, s)

# latex-Gleichung der linearen Regression
lin_reg = ev.tex_linreg(
        "v(t)",
        val,
        std,
        unit = ["\second\per\meter", "\meter"]
)

print_tex(ev.latexEq(lin_reg))

# ===== plot =======================================

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(t, s, linestyle = 'none', marker = '+', label = 'Messwerte')
lim = ax1.get_xlim()
x = np.linspace(lim[0], lim[1], 1000)
ax1.plot(x, G(x, val[0], val[1]), label="Fit")

ax1.set_xlabel(r'$t / \si{\second}$')
ax1.set_ylabel(r'$s / \si{\meter}$')

ax1.legend(loc = 'best')
ax1 = ev.plot_layout(ax1)

fig1.tight_layout()

# oder mit simpleplot:

p = sp.figure([r'$t / \si{\second}$', r'$t / \si{\meter}$'])
p.add_plot(t, s, label="Messwerte")
p.add_plot(G, val[0], val[1], label="Fit")
p.plot()

# plt.show()

# ===== Tabellen ===================================

# LateX-Tabelle erzeugen
t = lt.latextable(
    [t, v_arr],
    "table",
    alignment = 'CC',
    form = '.3f',
)

# ===== Daten/Plots speichern ======================

# tex schreiben
ev.write('file', v_tex)

# plots speichern
fig1.savefig('file.pdf')


"""

###########################################################################
#                            Beginn Auswertung                            #
###########################################################################

# =========================================================
# 	Eichung
# =========================================================

f_eich, U_a_eich = np.loadtxt("../messwerte/eichung.txt", unpack=True)

fig_eich = plt.figure()
ax = fig_eich.add_subplot(111)

ax.plot(
    f_eich,
    U_a_eich,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_eich.tight_layout()

# ====[ R 1000 ]============================================

R_1000, U_a_R_1000 = np.loadtxt("../messwerte/R_1000.txt", unpack=True)

fig_R_100 = plt.figure()
ax = fig_R_100.add_subplot(111)

ax.plot(
    R_1000,
    U_a_R_1000,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R_1000$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_R_100.tight_layout()


# ====[ R 100k ]============================================

R_100k, U_a_R_100k = np.loadtxt("../messwerte/R_100k.txt", unpack=True)

fig_R_100k = plt.figure()
ax = fig_R_100k.add_subplot(111)

ax.plot(
    R_100k,
    U_a_R_100k,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R$ in $\si{\kilo\ohm}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_R_100k.tight_layout()
fig_R_100k.savefig("../tex/bilder/R_100k.pdf")

# ====[ Eichung selektiv ]==================================

f_sel, U_a_sel = np.loadtxt("../messwerte/eichung_selektiv.txt", unpack=True)

fig_sel = plt.figure()
ax = fig_sel.add_subplot(111)

ax.plot(
    f_sel,
    U_a_sel,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_sel.tight_layout()
fig_sel.savefig("../tex/bilder/f_sel.pdf")

# ====[ R_1000 selektiv ]===================================

R_1000_sel, U_a_R_1000_sel = np.loadtxt(
    "../messwerte/R_1000_sel.txt",
    unpack=True
)

# R_1000_sel[-4:] = 4 * R_1000_sel[-4:]
U_a_R_1000_sel[-5:] = 2.5**2 * U_a_R_1000_sel[-5:]

fig_R_1000_sel = plt.figure()
ax = fig_R_1000_sel.add_subplot(111)

ax.plot(
    R_1000_sel,
    U_a_R_1000_sel,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R$ in $\si{\ohm}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_R_1000_sel.tight_layout()
fig_R_1000_sel.savefig("../tex/bilder/R_1000_sel.pdf")

# ====[ R_100k selektiv ]===================================

R_100k_sel, U_a_R_100k_sel = np.loadtxt(
    "../messwerte/R_100k_sel.txt",
    unpack=True
)

fig_R_100k_sel = plt.figure()
ax = fig_R_100k_sel.add_subplot(111)

ax.plot(
    R_100k_sel,
    U_a_R_100k_sel,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R$ in $\si{\kilo\ohm}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_R_100k_sel.tight_layout()
fig_R_100k_sel.savefig("../tex/bilder/R_100k_sel.pdf")
