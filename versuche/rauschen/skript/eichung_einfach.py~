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


#=========================================================
#     Eigenrauschen
#=========================================================

V_N_eigen, U_a_eigen = np.loadtxt("../messwerte/kurzschluss.txt", unpack=True)


tab_eigenrauschen_einfach = lt.latextable(
    [V_N_eigen, U_a_eigen],
    "../tex/tabellen/eichung_eigenrauschen_einfach.tex",
    alignment = 'CC',
    form = ['g', '.1f'],
)


U_a_eigen = U_a_eigen/(10.*(1000.*V_N_eigen)**2)












# =========================================================
# 	Eichung
# =========================================================

f_eich, U_a_eich = np.loadtxt("../messwerte/eichung.txt", unpack=True)

tab_einfach = lt.latextable(
    [f_eich, U_a_eich],
    "../tex/tabellen/eichung_einfach.tex",
    alignment = 'CC',
    form = '.3f',
    split = 2
)




U_ein = 120./1001.*10**(-3) #V * Abschwächer

V_gleich = 10.
V_vor = 1000.
V_nach = 20.

v = V_gleich * (V_vor * V_nach)**2 #Verstärkungsfaktor

U_a_eich = U_a_eich/(v*U_ein**2) - U_a_eigen[4]/(U_ein**2) #korriegierte Messwerte



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
ax.set_ylabel(r'$\beta $')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig_eich.tight_layout()



fig_eich.savefig("../tex/bilder/eichung_einfach.pdf")





#---Integration--->
#ober/untersummenintegration und mittelwert

J_ober=0
J_unter=0

for i in range (1,len(f_eich)):
	if f_eich[i]<=24:
		J_ober += U_a_eich[i]* (f_eich[i]-f_eich[i-1])
	else:
		J_ober += U_a_eich[i-1]* (f_eich[i]-f_eich[i-1])

for i in range (1,len(f_eich)):
	if f_eich[i]<=24:
		J_unter += U_a_eich[i-1]* (f_eich[i]-f_eich[i-1])
	else:
		J_unter += U_a_eich[i]* (f_eich[i]-f_eich[i-1])

print(J_ober)
print(J_unter)

# mittleres Integral mit Fehler
A = ev.get_uncert([J_ober,J_unter])

print(A)


# tex schreiben
ev.write('../tex/tabellen/eichfaktor_einfach.tex', str(A))

#<---Integration---
