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

# Radius
r_vert = 11.735e-2
r_hor = 15.790e-2
r_sweep = 16.39e-2

# Windungen
N_vert = 20
N_hor = 154
N_sweep = 11

# Umrechnungsfaktoren Strom/Umdrehungen
I_per_U_vert = 0.1
I_per_U_hor = 0.3
I_per_U_sweep = 0.1


# B Feldstärke
def B(I, N, r):
    return const.mu_0 * 8 / np.sqrt(125) * I * N / r

# =========================================================
# 	Vertikal komponente
# =========================================================

U_vert = 2.36
I_vert = U_vert * I_per_U_vert

B_vert = B(I_vert, N_vert, r_vert)
B_vert_tex = r"\SI{{{:.1f}}}{{\micro\tesla}}".format(B_vert * 1e6)
print(B_vert_tex)

ev.write("../tex/gleichungen/B_vert.tex", B_vert_tex)

# =========================================================
# 	Horizontalkomponente
# =========================================================


def G(x, m, b):
    return m*x + b

# ====[ Isotop 1 ]==========================================

f_1, U_sweep_1, U_hor_1 = np.loadtxt("../messwerte/teil_c.txt", unpack=True)

B_1 = B(U_sweep_1 * I_per_U_sweep, N_sweep, r_sweep) + \
      B(U_hor_1 * I_per_U_hor, N_hor, r_hor)

B_1 = B_1 * 1000  # in milli tesla

# Ausreißer gefunden -.-
B_1_fit = B_1[:-1]
B_1_ausreisser = B_1[-1]
f_1_fit = f_1[:-1]
f_1_ausreisser = f_1[-1]

# Fit
val_1, cov_1 = optimize.curve_fit(G, f_1_fit, B_1_fit)
std_1 = ev.get_std(cov_1)

# Plot
# fig_1 = plt.figure()
# ax_1 = fig_1.add_subplot(111)
#
# ax_1.plot(
#     f_1_fit,
#     B_1_fit,
#     color='k',
#     linestyle='none',
#     marker='+',
#     label='Messwerte'
# )
#
# ax_1.plot(
#     f_1_ausreisser,
#     B_1_ausreisser,
#     color='k',
#     linestyle='none',
#     marker='*',
#     label='Ausreisser'
# )
#
# fig_lim = ax_1.get_xlim()
# fig_x = np.linspace(fig_lim[0], fig_lim[1], 1000)
# ax_1.plot(
#     fig_x,
#     G(fig_x, val_1[0], val_1[1]),
#     color='k',
#     label="Fit"
# )
#
# ax_1.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
# ax_1.set_ylabel(r'$B$ in $\si{\milli\tesla}$')
#
# ax_1.legend(loc='best')
# ax_1 = ev.plot_layout(ax_1)
#
# fig_1.tight_layout()
# fig_1.savefig("../tex/bilder/isotop_1.pdf")

ev.write(
    "../tex/gleichungen/gerade_1.tex",
    ev.tex_linreg(
        r"G_1(\nu)",
        val_1 * 1000,
        std_1 * 1000,
        form=["({:L})", "({:L})"],
        unit=[r"\micro\tesla\per\kilo\hertz", r"\micro\tesla"]
    )
)

# Korrektur -> zu einer Tabelle zusammensfügen
# lt.latextable(
#     [f_1, U_sweep_1, U_hor_1, B_1],
#     "../tex/tabellen/tab_1.tex",
#     form=["g", ".2f", ".2f", ".3f"]
# )

# ====[ Isotop 2 ]==========================================


f_2, U_sweep_2, U_hor_2 = np.loadtxt("../messwerte/teil_c_2.txt", unpack=True)

B_2 = B(U_sweep_2 * I_per_U_sweep, N_sweep, r_sweep) + \
      B(U_hor_2 * I_per_U_hor, N_hor, r_hor)

B_2 = B_2 * 1000  # in milli tesla

B_2_fit = B_2
f_2_fit = f_2

# Fit
val_2, cov_2 = optimize.curve_fit(G, f_2_fit, B_2_fit)
std_2 = ev.get_std(cov_2)

# Plot
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111)

ax_2.plot(
    f_1_fit,
    B_1_fit,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte Isotop 1'
)

ax_2.plot(
    f_1_ausreisser,
    B_1_ausreisser,
    color='k',
    linestyle='none',
    marker='*',
    label='Ausreisser Isotop 1'
)

ax_2.plot(
    f_2_fit,
    B_2_fit,
    color='k',
    linestyle='none',
    marker='x',
    label='Messwerte Isotop 2'
)

fig_lim = ax_2.get_xlim()
fig_x = np.linspace(fig_lim[0], fig_lim[1], 1000)

ax_2.plot(
    fig_x,
    G(fig_x, val_1[0], val_1[1]),
    color='k',
    label="Fit Isotop 1"
)

ax_2.plot(
    fig_x,
    G(fig_x, val_2[0], val_2[1]),
    color='k',
    linestyle="--",
    label="Fit Isotop 2"
)

ax_2.set_xlabel(r'$\nu$ in $\si{\kilo\hertz}$')
ax_2.set_ylabel(r'$B$ in $\si{\milli\tesla}$')

ax_2.legend(loc='best')
ax_2 = ev.plot_layout(ax_2)

fig_2.tight_layout()
fig_2.savefig("../tex/bilder/isotop.pdf")

ev.write(
    "../tex/gleichungen/gerade_2.tex",
    ev.tex_linreg(
        r"G_2(\nu)",
        val_2 * 1000,
        std_2 * 1000,
        form=["({:L})", "({:L})"],
        unit=[r"\micro\tesla\per\kilo\hertz", r"\micro\tesla"]
    )
)

lt.latextable(
    [f_1, U_sweep_1, U_hor_1, B_1, U_sweep_2, U_hor_2, B_2],
    "../tex/tabellen/tab.tex",
    form=["g", ".2f", ".2f", ".3f", ".2f", ".2f", ".3f"]
)

# ====[ B-Felder ]==========================================


B_1_uc = uc.ufloat(val_1[1]*1000, std_1[1]*1000)
B_2_uc = uc.ufloat(val_2[1]*1000, std_2[1]*1000)

B_1 = ev.tex_eq(B_1_uc, form="({:L})", unit=r"\micro\tesla")
B_2 = ev.tex_eq(B_2_uc, form="({:L})", unit=r"\micro\tesla")
B_mean = ev.tex_eq(
    np.mean([B_1_uc, B_2_uc]),
    form="({:L})",
    unit=r"\micro\tesla"
)

ev.write("../tex/gleichungen/B1.tex", B_1)
ev.write("../tex/gleichungen/B2.tex", B_2)
ev.write("../tex/gleichungen/B_mean.tex", B_mean)

# =========================================================
# 	Lande-Faktor
# =========================================================

# B = 4 * pi * m_0 / (e_0 * g_J) * nu
#     \________  ______________/
#              \/
#           Steigung m

g_1_uc = 4 * np.pi * \
    const.m_e / (const.e * uc.ufloat(val_1[0], std_1[0])) * 1e6
g_2_uc = 4 * np.pi * \
    const.m_e / (const.e * uc.ufloat(val_2[0], std_2[0])) * 1e6

g_1 = ev.tex_eq(g_1_uc)
g_2 = ev.tex_eq(g_2_uc)

ev.write("../tex/gleichungen/g1.tex", g_1)
ev.write("../tex/gleichungen/g2.tex", g_2)

# =========================================================
# 	Kernspin
# =========================================================

g_J = 2.0023


def I(g_J, g_F):
    a = g_J / g_F
    return -1 + 1/4. * a + unp.sqrt((1 - 1/4. * a)**2 - (1 - a))

I_1_uc = I(g_J, g_1_uc)
I_2_uc = I(g_J, g_2_uc)

I_1 = ev.tex_eq(I_1_uc)
I_2 = ev.tex_eq(I_2_uc)

ev.write("../tex/gleichungen/I1.tex", I_1)
ev.write("../tex/gleichungen/I2.tex", I_2)
