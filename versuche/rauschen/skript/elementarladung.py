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

I, U_a, delta, V_N = np.loadtxt("../messwerte/elementarladung.txt", unpack=True)

# LateX-Tabelle erzeugen
t = lt.latextable(
    [I, U_a, delta, V_N],
    "../tex/tabellen/elementarladung_messwerte.tex",
    alignment = 'CCCC',
    form =['.1f', '.2f', '.2f', 'g'],
)


#Bestimmung von e_0

delta_nu = 20.*10.**3
R = 4680.
v = (1000.*V_N)**2 * 10. # vor, nach, gleich
I_anode = I/1000.
I_a = (U_a/v)/(R**2)



# Geradenfunktion
def G(x, m, b):
    return m*x + b

# linerare Ausgleichrechnung
val, cov = optimize.curve_fit(G, I_anode, I_a)
std = ev.get_std(cov)


# latex-Gleichung der linearen Regression
lin_reg = ev.tex_linreg(
        "G(I_0)",
        val,
        std,
        unit = [r"\ampere", r"\ampere^2"]
)

ev.write('../tex/tabellen/elementarladung_gleichung.tex', lin_reg)


#plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(I_anode, I_a, linestyle = 'none', marker = '+', label = 'Messwerte')
lim = ax1.get_xlim()
x = np.linspace(lim[0], lim[1], 1000)
ax1.plot(x, G(x, val[0], val[1]), label="Fit")

ax1.set_xlabel(r'$I_0 / \si{\ampere}$')
ax1.set_ylabel(r'$I^2 / \si{\ampere}^2$')

ax1.legend(loc = 'best')
ax1 = ev.plot_layout(ax1)

fig1.tight_layout()
fig1.savefig('../tex/bilder/elementarladung.pdf')
#


e = uc.ufloat(val[0],std[0])/(2.*delta_nu)

print(e)
ev.write('../tex/tabellen/elementarladung.tex', str(e*10**19) )


























