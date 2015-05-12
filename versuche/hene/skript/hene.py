#-*- coding: iso-8859-1 -*-

# ==================================================
#	import modules
# ==================================================

import os
import sys

# append paths
# sys.path.append("/home/mario12/dokumente/physik/module/fp/python/")

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

sp.params["text.latex.preamble"] = sp.tex_mathpazo_preamble
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
            s.replace('\t',' ').replace('\n',' ').replace('%',''),
            color='k', size = 'x-large'
    )
    plt.show()

# ==================================================
#	example
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

# ===== Berechnungen ===============================

# berechne Geschwindigkeiten in array
v_arr = v(s, t)

# mittlere Geschwindigkeit mit Fehler
v_uc = ev.get_uncert(v_arr)

# Geschwindigkeit als LateX-Code
v_tex = ev.to_tex(v_uc, unit="{m}{s}")

# linerare Ausgleichrechnung
val, cov = optimize.curve_fit(G, t, s)
std = ev.get_std(cov)

# oder:

val, std = ev.fit(G, t, s)

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
ax1.plot(x, G(x, val[0], val[1]), label="Fit")

ax1.set_xlabel(r'$t / \unit{s}$')
ax1.set_ylabel(r'$s / \unit{m}$')

ax1.legend(loc = 'best')
ax1 = ev.plot_layout(ax1)

fig1.tight_layout()

# oder mit simpleplot:

p = sp.figure([r'$s / \unit{m}$', r'$s / \unit{m}$'])
p.add_plot(t, s, label="Messwerte")
p.add_plot(G, val[0], val[1], label="Fit")
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
    transpose = True,
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

# ==================================================
#	start evaluation
# ==================================================

# ===== Messdaten ==================================


# Stabilitätsbedingung
# l in cm, I in mA
l_S1, I_S1 = ev.get_data("../messwerte/messwerte", unpack=True, index=1)
l_S2, I_S2 = ev.get_data("../messwerte/messwerte", unpack=True, index=2)

# Moden
# a in mm, I in nA
a_M1, I_M1 = ev.get_data("../messwerte/messwerte", unpack=True, index=3)
# a in mm, I in muA
a_M2, I_M2 = ev.get_data("../messwerte/messwerte", unpack=True, index=4)

# Polaristion
# phi in grad, I in muA
phi, I_phi = ev.get_data("../messwerte/messwerte", unpack=True, index=5)

# Gitter
# Abstand in cm
order, a_r, a_l = ev.get_data("../messwerte/messwerte", unpack=True, index=6)

# ==================================================
#       Stabilitätsbedingung
# ==================================================


def g(L, r): return 1.0 - L/r
def gg(L, r1, r2): return g(L, r1) * g(L, r2)

I_S1_norm = I_S1*0.2 / max(I_S1)
I_S2_norm = I_S2*0.2 / max(I_S2)

# p_S1 = sp.figure([r'$l / \unit{cm}$', r'$\tilde{I}$'])
# p_S1.add_plot(
#     l_S1,
#     I_S1_norm,
#     label="Messwerte, $r_1 = \unit[1400]{mm}, r_2 = \unit[1400]{mm}",
#     marker="+",
#     color="k"
# )
# p_S1.add_plot(gg, 140, 140, c="k", label="Theoretische Kurve")
# fig_S1 = p_S1.plot()

fig_S1 = plt.figure()
ax1 = fig_S1.add_subplot(111)

lns1 = ax1.plot(
    l_S1,
    I_S1,
    ls = 'none',
    marker = '+',
    label="Messwerte, $r_1 = \unit[1400]{mm}, r_2 = \unit[1400]{mm}",
    c = "k"
)

x = np.linspace(100, 160, 1000)
ax2 = ax1.twinx()
lns2 = ax2.plot(
    x,
    gg(x, 140, 140),
    c = "k",
    label="theoretische Kurve"
)
ax1.set_xlabel(r'$l / \unit{cm}$')
ax1.set_ylabel(r'$I / \unit{mA}$')
ax2.set_ylabel(r'$g_1g_2$')

lns = lns1+lns2
labs = [l.get_label() for l in lns]

ax1 = ev.plot_layout(ax1)
leg = ax1.legend(lns, labs, loc="best")

ltext = leg.get_texts()
frame = leg.get_frame()
frame.set_facecolor('0.90')
frame.set_edgecolor('0.90')

ax2.yaxis.set_minor_locator(ev.AutoMinorLocator())
ax2.spines['top'].set_color('none')
fig_S1.tight_layout()

# p_S2 = sp.figure([r'$l / \unit{cm}$', r'$\tilde{I}$'])
# p_S2.add_plot(
#     l_S2 - 10.0,
#     I_S2_norm,
#     label="Messwerte, $r_1 = \unit[1000]{mm}, r_2 = \unit[1400]{mm}",
#     marker="+",
#     color="k"
# )
# p_S2.add_plot(gg, 100, 140, color="k", label="Theoretische Kurve")
# p_S2.add_plot(lambda x: x*0.0, ls="--", color="k")
# fig_S2 = p_S2.plot()
fig_S2 = plt.figure()
ax1 = fig_S2.add_subplot(111)

lns1 = ax1.plot(
    l_S2 - 10.0,
    I_S2,
    ls = 'none',
    marker = '+',
    label="Messwerte, $r_1 = \unit[1000]{mm}, r_2 = \unit[1400]{mm}",
    c = "k"
)

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

x = np.linspace(80, 180, 1000)
ax2 = ax1.twinx()
lns2 = ax2.plot(
    x,
    gg(x, 100, 140),
    c = "k",
    label="theorteischer Verlauf"
)
ax2.plot(
    x,
    x*0.0,
    c = "k",
    ls = "--",
    label=None
)
ax1.set_xlabel(r'$l / \unit{s}$')
ax1.set_ylabel(r'$I / \unit{mA}$')
ax2.set_ylabel(r'$g_1g_2$')

align_yaxis(ax2, 0, ax1, 0)

lns = lns1+lns2
labs = [l.get_label() for l in lns]

ax1 = ev.plot_layout(ax1)
leg = ax1.legend(lns, labs, loc="best")

ltext = leg.get_texts()
frame = leg.get_frame()
frame.set_facecolor('0.90')
frame.set_edgecolor('0.90')

ax2.yaxis.set_minor_locator(ev.AutoMinorLocator())
ax2.spines['top'].set_color('none')
fig_S2.tight_layout()

# ==================================================
#	Moden
# ==================================================


def gauss(r, I0, r0, w): return I0 * np.exp(-((r - r0)/w)**2)
# def TEM10(r, I0, r0, w, a, b): return (a*(r-r0) + b)**2 * gauss(r, I0, r0, w)
def TEM10(r, r0, w, a, b):
    return (a*(r-r0) + b)**2 * np.exp(-((r - r0)/w)**2)

# ===== Gauss/TEM00 ================================

val_M2, std_M2 = ev.fit(gauss, a_M2, I_M2)

p_M2 = sp.figure([r'$r / \unit{mm}$', r'$I / \si{\nA}$'])
p_M2.add_plot(
    a_M2,
    I_M2,
    label="Messwerte",
    marker="+",
    color="k"
)
p_M2.add_plot(
    gauss,
    val_M2[0],
    val_M2[1],
    val_M2[2],
    color="k",
    label="Fit $I_{00}$"
)
fig_M2 = p_M2.plot()

# tex parameter
I0_M2_tex = ev.tex_eq(
    uc.ufloat(val_M2[0], std_M2[0]),
    # name=r"I_0",
    unit=r"\nA",
    form="({:0L})"
)
r0_M2_tex = ev.tex_eq(
    uc.ufloat(val_M2[1], std_M2[1]),
    # name=r"r_0",
    unit=r"\mm",
    form="({:0L})"
)
w_M2_tex = ev.tex_eq(
    uc.ufloat(val_M2[2], std_M2[2]),
    # name=r"w",
    unit=r"\mm",
    form="({:0L})"
)

# ===== TEM10 ======================================

val_M1, std_M1 = ev.fit(TEM10, a_M1, I_M1)

p_M1 = sp.figure([r'$r / \unit{mm}$', r'$I / \unit{uA}$'])
p_M1.add_plot(
    a_M1,
    I_M1,
    label="Messwerte",
    marker="+",
    color="k"
)
p_M1.add_plot(
    TEM10,
    val_M1[0],
    val_M1[1],
    val_M1[2],
    val_M1[3],
    # val_M1[4],
    color="k",
    label="Fit $I_{10}$"
)
fig_M1 = p_M1.plot()

# tex parameter
# I0_M1_tex = ev.tex_eq(
#     uc.ufloat(val_M1[0], std_M1[0]),
#     # name=r"I_0",
#     unit=r"\nA",
#     form="({:0L})"
# )
r0_M1_tex = ev.tex_eq(
    uc.ufloat(val_M1[0], std_M1[0]),
    # name=r"r_0",
    unit=r"\mm",
    form="({:0L})"
)
w_M1_tex = ev.tex_eq(
    uc.ufloat(val_M1[1], std_M1[1]),
    # name=r"w",
    unit=r"\mm",
    form="({:0L})"
)
a_tex = ev.tex_eq(
    uc.ufloat(val_M1[2], std_M1[2]),
    # name=r"w",
    unit=r"\sqrt{\upmu\text{A}}",
    form="({:0L})"
)
b_tex = ev.tex_eq(
    uc.ufloat(val_M1[3], std_M1[3]),
    # name=r"w",
    unit=r"\sqrt{\upmu\text{A}}",
    form="({:0L})"
)

# ==================================================
#	Polarisation
# ==================================================


def pol(phi, A, phi0): return A * np.sin(phi*np.pi/180. - phi0)**2

val_P, std_P = ev.fit(pol, phi, I_phi)

p_P = sp.figure([r'$\varphi / \si{\degree}$', r'$I / \si{\nano\ampere}$'])
p_P.add_plot(
    phi,
    I_phi,
    label="Messwerte",
    marker="+",
    color="k"
)
p_P.add_plot(
    pol,
    val_P[0],
    val_P[1],
    # val_P[2],
    color="k",
    label=r"Fit $I_\text{P}$"
)
fig_P = p_P.plot()

# Paramter zu tex
A_tex = ev.tex_eq(
    uc.ufloat(val_P[0], std_P[0]),
    form="({:1L})",
    unit=r"\nA"
)

phi0_tex = ev.tex_eq(
    uc.ufloat(val_P[1], std_P[1]),
    form="({:1L})",
    unit=r"\degree"
)

print A_tex
print phi0_tex

# plt.show()

# ==================================================
#	Gitter
# ==================================================

def l(n, L, a, g): return g / (n * np.sqrt(1. + (L/a)**2))

L_G = 47.7
g_G = 1.0 / 1000.0

l_r = l(order, L_G, a_r, g_G) * 1e7
l_l = l(order, L_G, a_l, g_G) * 1e7

l = ev.get_uncert(list(l_r) + list(l_l))

l_tex = ev.tex_eq(l, name="\lambda", form="({:2L})", unit=r"\nano\meter")
# print l_tex

# ==================================================
#	Speichern
# ==================================================

# ===== Plots ======================================

# fig_S1.savefig("../tex/bilder/fig_S1.pdf")
# fig_S2.savefig("../tex/bilder/fig_S2.pdf")
fig_M1.savefig("../tex/bilder/fig_M1.pdf")
fig_M2.savefig("../tex/bilder/fig_M2.pdf")
fig_P.savefig("../tex/bilder/fig_P.pdf")

# ===== Gleichungen ================================

ev.write("../tex/gleichungen/l", l_tex)
ev.write("../tex/gleichungen/A", A_tex)
ev.write("../tex/gleichungen/phi0", phi0_tex)
ev.write("../tex/gleichungen/I0_M2", I0_M2_tex)
ev.write("../tex/gleichungen/r0_M2", r0_M2_tex)
ev.write("../tex/gleichungen/w_M2", w_M2_tex)
# ev.write("../tex/gleichungen/I0_M1", I0_M1_tex)
ev.write("../tex/gleichungen/r0_M1", r0_M1_tex)
ev.write("../tex/gleichungen/w_M1", w_M1_tex)
ev.write("../tex/gleichungen/a_M1", a_tex)
ev.write("../tex/gleichungen/b_M1", b_tex)

# ===== Tabellen ===================================

temp1 = np.split(l_S1, 3)
temp2 = np.split(I_S1, 3)

lt.latextable(
    [temp1[0], temp2[0], temp1[1], temp2[1],
    temp1[2], temp2[2]],
    "../tex/tabellen/S1.tex",
    alignment = 'CCCCCC',
    form = ['.0f', '.1f', '.0f', '.1f', '.0f', '.1f'],
    transpose = True,
)

temp1 = np.split(l_S2, 3)
temp2 = np.split(I_S2, 3)

lt.latextable(
    [temp1[0], temp2[0], temp1[1], temp2[1], temp1[2], temp2[2]],
    "../tex/tabellen/S2.tex",
    alignment = 'CCCCCC',
    form = ['.0f', '.1f', '.0f', '.1f', '.0f', '.1f'],
    transpose = True,
)

temp1 = np.split(a_M1, 3)
temp2 = np.split(I_M1, 3)

lt.latextable(
    [temp1[0], temp2[0], temp1[1], temp2[1], temp1[2], temp2[2]],
    "../tex/tabellen/M1.tex",
    alignment = 'CCCCCC',
    form = ['.0f', '.2f', '.0f', '.2f', '.0f', '.2f'],
    transpose = True,
)

temp1 = np.split(np.append(a_M2, ["-", "-"]), 3)
temp2 = np.split(np.append(I_M2, ["-", "-"]), 3)

lt.latextable(
    [temp1[0], temp2[0], temp1[1], temp2[1], temp1[2], temp2[2]],
    "../tex/tabellen/M2.tex",
    alignment = 'CCCCCC',
    form = ['.0f', '.2f', '.0f', '.2f', '.0f', '.2f'],
    transpose = True,
)

temp1 = np.split(phi, 3)
temp2 = np.split(I_phi, 3)

lt.latextable(
    [temp1[0], temp2[0], temp1[1], temp2[1], temp1[2], temp2[2]],
    "../tex/tabellen/P.tex",
    alignment='CCCCCC',
    form=['.0f', '.2f', '.0f', '.2f', '.0f', '.2f'],
    transpose=True,
)

temp1 = np.split(np.append(order, "-"), 2)
temp2 = np.split(np.append(a_r, "-"), 2)
temp3 = np.split(np.append(a_l, "-"), 2)

lt.latextable(
    # [order, a_r, a_l],
    [temp1[0], temp2[0], temp3[0], temp1[1], temp2[1], temp3[1]],
    "../tex/tabellen/G.tex",
    alignment='CCCCCC',
    form=['.0f', '.1f', '.1f', '.0f', '.1f', '.1f'],
    transpose=True,
)
