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

# ====[ Oxyd ]==============================================


f_ox, U_a_ox, delta_f, V_N_ox, delta_f = np.loadtxt(
    "../messwerte/oxyd_kathode.txt",
    unpack=True
)

R = 2200  # ohm


def W(U_a, nu, R):
    return U_a / (R**2 * nu)

U_a_norm = U_a_ox / V_N_ox**2

fig_ox = plt.figure()
ax = fig_ox.add_subplot(111)

ax.plot(
    f_ox,
    W(U_a_norm, delta_f, R),
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

# ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
# ax.set_ylabel(r'$U_a$ in $\si{\volt}$')
ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(loc='best')
# ax = ev.plot_layout(ax)

fig_ox.tight_layout()
fig_ox.savefig("../tex/bilder/U_a_ox.pdf")

# ====[ Reinmetall ]========================================

f_metall, U_a_metall, delta_f, V_N_metall, delta_f_metall = np.loadtxt(
    "../messwerte/kathode.txt",
    unpack=True
)

R = 4680  # ohm


def W(U_a, nu, R):
    return U_a / (R**2 * nu)

U_a_norm_metall = U_a_metall / V_N_metall**2

fig_metall = plt.figure()
ax = fig_metall.add_subplot(111)

ax.plot(
    f_metall,
    W(U_a_norm_metall, delta_f_metall, R),
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

# ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
# ax.set_ylabel(r'$U_a$ in $\si{\volt}$')
ax.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(loc='best')
# ax = ev.plot_layout(ax)

fig_metall.tight_layout()
fig_metall.savefig("../tex/bilder/U_a_metall.pdf")

# ====[ Kennlinien ]========================================

U_anode_1, U_a_1, dU_A_1, V_N_kennlinie_1 = np.loadtxt(
    "../messwerte/kennlinie_8.txt",
    unpack=True
)

U_a_norm_kennlinie_1 = U_a_1 / V_N_kennlinie_1**2 / R**2

fig_kennlinie_1 = plt.figure()
ax = fig_kennlinie_1.add_subplot(111)

ax.plot(
    U_anode_1,
    U_a_norm_kennlinie_1,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$U_\text{Anode}$ in $\si{\volt}$')
ax.set_ylabel(r'$I_a$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_kennlinie_1.tight_layout()
fig_kennlinie_1.savefig("../tex/bilder/kennlinie_1.pdf")

# ====[ Kennlinie 0.9 ]=====================================


U_anode_2, U_a_2, dU_A_2, V_N_kennlinie_2 = np.loadtxt(
    "../messwerte/kennlinie_9.txt",
    unpack=True
)

U_a_norm_kennlinie_2 = U_a_2 / V_N_kennlinie_2**2 / R**2

fig_kennlinie_2 = plt.figure()
ax = fig_kennlinie_2.add_subplot(111)

ax.plot(
    U_anode_2,
    U_a_norm_kennlinie_2,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$U_\text{Anode}$ in $\si{\volt}$')
ax.set_ylabel(r'$I_a$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_kennlinie_2.tight_layout()
fig_kennlinie_2.savefig("../tex/bilder/kennlinie_2.pdf")

# ====[ Elementarleidung ]==================================

I_Anode, U_a_e, dU_a_e, V_N_e = np.loadtxt(
    "../messwerte/elementarladung.txt",
    unpack=True
)

U_a_norm_e = U_a_e / V_N_e**2
I_a_e = U_a_norm_e / R**2

fig_e = plt.figure()
ax = fig_e.add_subplot(111)

ax.plot(
    I_Anode,
    I_a_e,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$I_\text{Anode}$ in $\si{\milli\ampere}$')
ax.set_ylabel(r'$I_a$ in $\si{\ampere}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_e.tight_layout()
fig_e.savefig("../tex/bilder/elementarladung.pdf")
