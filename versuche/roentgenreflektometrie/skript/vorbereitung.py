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


###########################################################################
#                                 Teil a)                                 #
###########################################################################


z1 = 0.0
z2 = -800.0e-10

k = 2.0*np.pi / 1.54e-10

n1 = 1.0
n2 = 1.0 - 1.0e-6
n3 = 1.0 - 3.0e-6

n_a = [n1, n2, n3]
z = [z1, z2]


# ==================================================
# 	Funktionen
# ==================================================


def k_z(j, n, k, a):
    return k*np.sqrt(n[j]**2 - np.cos(a)**2)


def r(j, n, k, a):
    k_j1 = k_z(j+1, n, k, a)
    k_j = k_z(j, n, k, a)
    divident = k_j - k_j1
    divisor = k_j + k_j1

    return divident / divisor


def X(j, n, k, a, z):
    if j == 2:
        return 0
    else:
        e_plus = np.exp(2*1j*k_z(j, n, k, a)*z[j])
        e_minus = np.exp(-2*1j*k_z(j, n, k, a)*z[j])
        divident = (r(j, n, k, a) + X(j+1, n, k, a, z)*e_plus)
        divisor = (1.0 + r(j, n, k, a)*X(j+1, n, k, a, z)*e_plus)

        return e_minus * divident / divisor


# ==================================================
# 	plot
# ==================================================


a_deg = np.linspace(0.2, 2.0, 1000)
a = np.deg2rad(a_deg)
X_a = np.abs(X(0, n_a, k, a, z))**2


fig = plt.figure()
ax = fig.add_subplot(111)

ax.semilogy(
    a_deg,
    X_a,
    color='k',
    linestyle='-',
    label='Messwerte'
)

ax.set_xlabel(r'$\alpha_i$ in $\si{\deg}$')
ax.set_ylabel(r'Reflectivity $|X|^2$')

ax.legend(loc='best')

leg = ax.get_legend()
ltext = leg.get_texts()
frame = leg.get_frame()
frame.set_facecolor('0.90')
frame.set_edgecolor('0.90')

fig.tight_layout()
fig.savefig("reflektivitaet.pdf")


###########################################################################
#                                 Teil b)                                 #
###########################################################################


n_substrat = 1.0 - 1.0e-6
sig_1 = 0.0
sig_2 = 6.0e-10

sig = [sig_1, sig_2]
n_b = [n1, n2, n_substrat]

# ==================================================
# 	Funktionen
# ==================================================


def r_sig(j, n, k, a, sig):
    k_j = k_z(j, n, k, a)
    k_j1 = k_z(j+1, n, k, a)
    divident = k_j - k_j1
    divisor = k_j + k_j1
    exponent = -2.0*k_j*k_j1*sig**2

    return divident / divisor * np.exp(exponent)


def X_sig(j, n, k, a, z, sig):
    if j == 2:
        return 0
    else:
        e_plus = np.exp(2*1j*k_z(j, n, k, a)*z[j])
        e_minus = np.exp(-2*1j*k_z(j, n, k, a)*z[j])
        divident = (
            r_sig(j, n, k, a, sig[j]) + X_sig(j+1, n, k, a, z, sig)*e_plus
        )
        divisor = (
            1.0 + r_sig(j, n, k, a, sig[j])*X_sig(j+1, n, k, a, z, sig)*e_plus
        )

        return e_minus * divident / divisor

# ==================================================
# 	plot
# ==================================================


a_deg = np.linspace(0.2, 2.0, 5000)
a = np.deg2rad(a_deg)
X_a_sig = np.abs(X_sig(0, n_a, k, a, z, sig))**2

fig = plt.figure()
ax = fig.add_subplot(111)

ax.semilogy(
    a_deg,
    X_a_sig,
    color='k',
    linestyle='-',
    label='Rauigkeit'
)

ax.set_xlabel(r'$\alpha_i$ in $\si{\deg}$')
ax.set_ylabel(r'Reflectivity $|X|^2$')

ax.legend(loc='best')

leg = ax.get_legend()
ltext = leg.get_texts()
frame = leg.get_frame()
frame.set_facecolor('0.90')
frame.set_edgecolor('0.90')

fig.tight_layout()
fig.savefig("reflektivitaet_rau.pdf")

plt.show()
