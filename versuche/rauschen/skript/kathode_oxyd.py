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


f_ox, U_a_ox, delta_U, V_N_ox, delta_f = np.loadtxt(
    "../messwerte/oxyd_kathode.txt",
    unpack=True
)



# LateX-Tabelle erzeugen
t = lt.latextable(
    [f_ox, U_a_ox, delta_f, V_N_ox, delta_f],
    "../tex/tabellen/kathode_oxyd.tex",
    alignment = 'CCCCC',
    form = ['.3f','.3f','.3f','g', '.3f']
)



R = 2200.  # ohm

f_ox    *= 1000
delta_f *= 1000


def W(U_a, nu, R):
    return U_a / (R**2 * nu)

U_a_norm = U_a_ox / V_N_ox**2

# LateX-Tabelle erzeugen
t = lt.latextable(
    [f_ox/1000, U_a_ox, delta_U, V_N_ox, delta_f/1000, np.log(f_ox), np.log(W(U_a_norm, delta_f, R)),np.log(W(delta_U, delta_f, R))],
    "../tex/tabellen/kathode_oxyd.tex",
    alignment='CCCCCCCC',
    form=['.3f', '.3f', '.3f', 'g', '.3f', '.3f', '.3f','.3f'],
)







W_norm =  W(U_a_norm, delta_f, R)


# mittleres Schrotrauschen
W_schrot = ev.get_uncert(W_norm[:11])
#korrigiertes Funkelrauschen
W_korr = W_norm-W_schrot.nominal_value


# Geradenfunktion
def G(x, m, b):
    return m*x + b


# linerare Ausgleichrechnung
val, cov = optimize.curve_fit(G, np.log(W_korr[-8:]), np.log(f_ox[-8:]) )
std = ev.get_std(cov)

# latex-Gleichung der linearen Regression
lin_reg = ev.tex_linreg(
        r"G(f)",
        val,
        std,
#        unit = ["\second\per\meter", "\meter"]
)

ev.write('../tex/tabellen/schrotanteil.tex', str(W_schrot*10**15) )
ev.write('../tex/tabellen/kathode_oxyd_gleichung.tex', str(lin_reg) )
ev.write('../tex/tabellen/kathode_oxyd_exponent.tex', str( uc.ufloat(-val[0],std[0]) ) )















#--plot---------------

fig_ox = plt.figure()
ax = fig_ox.add_subplot(111)

#ax.set_xscale('log')
#ax.set_yscale('log')


ax.plot(np.log(f_ox[-8:]), (G(np.log(f_ox[-8:]), val[0], val[1] )-0.7 ), label="Fit")



ax.plot(
    np.log(f_ox[-8:])  ,
    np.log(W_korr[-8:]),
    color='k',
    linestyle='none',
    marker='o',
    label='Um den Schrotanteil korrigierter Funkelanteil'
)

ax.plot(
    np.log(f_ox[:11])  ,
    np.log(W_norm[:11]),
    color='k',
    linestyle='none',
    marker='x',
    label='Reines Schrotrauschen'
)

ax.plot(
    np.log(f_ox[11:])  ,
    np.log(W_norm[11:]),
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)


ax.set_xlabel(r'$\ln(\{\nu\})$')
ax.set_ylabel(r'$\ln(\{W\})$')


ax.legend(loc='best')
# ax = ev.plot_layout(ax)

fig_ox.tight_layout()
fig_ox.savefig("../tex/bilder/kathode_oxyd.pdf")

