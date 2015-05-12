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
# 	function to print equations with
# 	matplotlib
# ==================================================


def show(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = [
        var_name for var_name,
        var_val in callers_local_vars if var_val is var
    ]
    var_format = " VARIABLE: {} ".format(var_name[0])
    print "{:=^50}".format(var_format)

    if isinstance(var, (int, float, uc.UFloat)):
        print "{} = {}".format(var_name[0], var)
    else:
        print var

    print ""  # newline


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
# 	example
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

###########################################################################
#                            Beginn Auswertung                            #
###########################################################################


# ==================================================
# 	- R, C, L in Abhängingkeit von f und
# 	  deren Theoriekurve plotten
# 	- G bestimmen.
# ==================================================


def teil_a():

    # f / kHz, C / pF, R / ohm, L / muH
    f_50k, C_50k, R_50k, L_50k = np.loadtxt(
        "../messwerte/leitungskonstanten_50_kurz.txt",
        unpack=True
    )

    # f / kHz, C / nF, R / ohm, L / muH
    f_50l, C_50l, R_50l, L_50l = np.loadtxt(
        "../messwerte/leitungskonstanten_50_lang.txt",
        unpack=True
    )

    # f / kHz, C / pF, R / ohm, L / muH
    f_75k, C_75k, R_75k, L_75k = np.loadtxt(
        "../messwerte/leitungskonstanten_75_kurz.txt",
        unpack=True
    )

    def Leitwert(R, C, L):
        return R*C/L

    G_50k = Leitwert(R_50k, C_50k, L_50k)
    G_50l = Leitwert(R_50l, C_50l, L_50l)
    G_75k = Leitwert(R_75k, C_75k, L_75k)

    lt.latextable(
        [f_50k, C_50k, R_50k, L_50k, G_50k],
        "../tex/tabellen/Leitungskonstanten_50k.tex",
        form=[".1f", ".2f", ".4f", ".2f", ".1f"],
        alignment="CCCCC"
    )

    lt.latextable(
        [f_50l, C_50l, R_50l, L_50l, G_50l],
        "../tex/tabellen/Leitungskonstanten_50l.tex",
        form=[".1f", ".4f", ".4f", ".1f", ".1f"],
        alignment="CCCCC"
    )

    lt.latextable(
        [f_75k, C_75k, R_75k, L_75k, G_75k],
        "../tex/tabellen/Leitungskonstanten_75k.tex",
        form=[".2f", ".4f", ".4f", ".1f", ".1f"],
        alignment="CCCCC"
    )

    plot_dict = {
        "R_50k": [R_50k, r'$R / \si{\ohm}$'],
        "R_50l": [R_50l, r'$R / \si{\ohm}$'],
        "R_75k": [R_75k, r'$R / \si{\ohm}$'],
        "C_50k": [C_50k, r'$C / \si{\pico\farad}$'],
        "C_50l": [C_50l, r'$C / \si{\micro\farad}$'],
        "C_75k": [C_75k, r'$C / \si{\pico\farad}$'],
        "L_50k": [L_50k, r'$L / \si{\micro\henry}$'],
        "L_50l": [L_50l, r'$L / \si{\micro\henry}$'],
        "L_75k": [L_75k, r'$L / \si{\micro\henry}$'],
        "G_50k": [G_50k, r'$G / \si{\milli\siemens}$'],
        "G_50l": [G_50l, r'$G / \si{\siemens}$'],
        "G_75k": [G_75k, r'$G / \si{\milli\siemens}$'],
    }

    for key in plot_dict:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(
            f_50k,  # Frequenz ist bei allen gleich
            plot_dict[key][0],
            color='k',
            linestyle='none',
            marker='+',
            label='Messwerte'
        )

        ax1.set_xlabel(r'$\nu / \si{\kHz}$')
        ax1.set_ylabel(plot_dict[key][1])

        ax1.legend(loc='best')
        ax1 = ev.plot_layout(ax1)

        fig.tight_layout()
        fig.savefig("../tex/bilder/{}.pdf".format(key))

    pass


# ==================================================
# 	Dämpfungskonstante
# ==================================================


def teil_b():

    # f / MHz, A0 / dBm, A / dBm
    f_D, A0, A = np.loadtxt(
        "../messwerte/werte_daempfung.txt",
        unpack=True
    )

    # Berechne Dämpfung, A0 = -10*log(U0/1mW) <-- dBm! ...
    alpha = A - A0

    lt.latextable(
        [f_D, A0, A, alpha],
        "../tex/tabellen/alpha.tex",
        form=["g", ".1f", ".1f", ".1f"],
        alignment="CCCC"
    )

    fig_alpha = plt.figure()
    ax1 = fig_alpha.add_subplot(111)

    ax1.plot(
        f_D,
        alpha,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    ax1.set_xlabel(r'$\nu / \si{\MHz}$')
    ax1.set_ylabel(r'$\alpha / \si{dB}$')

    ax1.legend(loc='best')
    ax1 = ev.plot_layout(ax1)

    fig_alpha.tight_layout()
    fig_alpha.savefig("../tex/bilder/alpha.pdf")

    pass

# ==================================================
# 	Länge der Kabel
# ==================================================


def teil_c():

    # Ausbreitunsgeschwindigkeit als frequenzunabhängig angenommen.
    epslison_r = 2.25
    v = const.c / np.sqrt(epslison_r)

    def laenge(v, t1, t2, unit=True, scale=1e-9):
        l = 0.5*v*(t2 - t1)*scale
        print l

        if unit:
            return ev.tex_eq(
                l,
                form="({:L})",
                unit="\meter"
            )
        else:
            return l

    # ===== 50, kurz ===================================

    # Alles in ns
    t_50k_offen_1 = uc.ufloat(104.0, 5.0)
    t_50k_offen_2 = uc.ufloat(307.0, 5.0)
    t_50k_kurz_1 = uc.ufloat(105.0, 5.0)
    t_50k_kurz_2 = uc.ufloat(307.0, 5.0)

    ev.write(
        "../tex/gleichungen/teil_c/laenge_50k_offen.tex",
        laenge(v, t_50k_offen_1, t_50k_offen_2)
    )

    ev.write(
        "../tex/gleichungen/teil_c/laenge_50k_kurz.tex",
        laenge(v, t_50k_kurz_1, t_50k_kurz_2)
    )

    # ===== 75, kurz ===================================

    # Alles in ns
    t_75k_offen_1 = uc.ufloat(105.0, 5.0)
    t_75k_offen_2 = uc.ufloat(308.0, 5.0)
    t_75k_kurz_1 = uc.ufloat(104.0, 5.0)
    t_75k_kurz_2 = uc.ufloat(306.0, 5.0)

    ev.write(
        "../tex/gleichungen/teil_c/laenge_75k_offen.tex",
        laenge(v, t_75k_offen_1, t_75k_offen_2)
    )

    ev.write(
        "../tex/gleichungen/teil_c/laenge_75k_kurz.tex",
        laenge(v, t_75k_kurz_1, t_75k_kurz_2)
    )

    # ===== 50, lang ===================================

    # Alles in ns
    t_50l_offen_1 = uc.ufloat(530.0, 50.0)
    t_50l_offen_2 = uc.ufloat(1400.0, 50.0)
    t_50l_kurz_1 = uc.ufloat(530.0, 50.0)
    t_50l_kurz_2 = uc.ufloat(1400.0, 50.0)

    ev.write(
        "../tex/gleichungen/teil_c/laenge_50l_offen.tex",
        laenge(v, t_50l_offen_1, t_50l_offen_2)
    )

    ev.write(
        "../tex/gleichungen/teil_c/laenge_50l_kurz.tex",
        laenge(v, t_50l_kurz_1, t_50l_kurz_2)
    )

    # ===== Alles in einer Tabelle =====================

    # Tabelle Zeiten
    col_1 = [t_50k_offen_1, t_75k_offen_1, t_50l_offen_1]
    col_2 = [t_50k_offen_2, t_75k_offen_2, t_50l_offen_2]
    col_3 = [t_50k_kurz_1, t_75k_kurz_1, t_50l_kurz_1]
    col_4 = [t_50k_kurz_2, t_75k_kurz_2, t_50l_kurz_2]

    table = [
            [
                r"\CU, kurz",
                r"\BU",
                r"\CU, lang",
            ],
                col_1,
                col_2,
                col_3,
                col_4
            ]

    lt.latextable(
        table,
        "../tex/tabellen/zeiten.tex",
        form=[
            "",
            "0.3gL",
            ["0.3gL", "0.3gL", "0.4gL"],
            "0.3gL",
            ["0.3gL", "0.3gL", "0.4gL"],
        ],
    )

    # Tabelle Längen
    col_1 = [
        laenge(v, t_50k_offen_1, t_50k_offen_2, False),
        laenge(v, t_50l_offen_1, t_50l_offen_2, False),
        laenge(v, t_75k_offen_1, t_75k_offen_2, False)
    ]
    col_2 = [
        laenge(v, t_50k_kurz_1, t_50k_kurz_2, False),
        laenge(v, t_50l_kurz_1, t_50l_kurz_2, False),
        laenge(v, t_75k_kurz_1, t_75k_kurz_2, False)
    ]

    table = [
            [
                r"\CU, kurz",
                r"\CU, lang",
                r"\BU",
            ],
                col_1,
                col_2
            ]

    lt.latextable(
        table,
        "../tex/tabellen/laengen.tex",
        form=[
            "",
            "0.3gL",
            "0.3gL"
        ],
    )

    pass

# ==================================================
# 	Leitungskonstanten aus Eingangsimpedanzen
# ==================================================


def teil_d():

    # ===== Konstanen ==================================

    Z0 = 50.0  # ohm, Wellenwiderstand des Kabels
    epslison_r = 2.25

    # ===== R bestimmen ================================

    t_R, U_R = np.loadtxt(
        "../messwerte/kurz.csv",
        unpack=True,
        delimiter=","
    )

    # Werte selektieren
    U_R_off_values = U_R[t_R < 500]
    U1_R_values = U_R[(t_R > 750) & (t_R < 1360)]
    U0_R_values = U_R[(t_R > 2000) & (t_R < 2250)]

    # Uncertainties
    U_R_off = ev.get_uncert(U_R_off_values)
    U1_R = ev.get_uncert(U1_R_values)
    U0_R = ev.get_uncert(U0_R_values)

    # Werte speichern
    ev.write(
        "../tex/gleichungen/teil_d/U_R_off.tex",
        ev.tex_eq(U_R_off, unit=r"\milli\volt")  # , form="({:L})")
    )
    ev.write(
        "../tex/gleichungen/teil_d/U1_R.tex",
        ev.tex_eq(U1_R, unit=r"\milli\volt")  # , form="({:L})")
    )
    ev.write(
        "../tex/gleichungen/teil_d/U0_R.tex",
        ev.tex_eq(U0_R, unit=r"\milli\volt")  # , form="({:L})")
    )

    # R bestimmen
    G = U0_R / U1_R - 1.0
    R = - Z0*(G + 1.0)/(G - 1.0)
    R_tex = ev.tex_eq(R, unit=r"\ohm", form="({:L})")

    ev.write("../tex/gleichungen/teil_d/R.tex", R_tex)

    # Daten in Tabelle
    # lt.latextable([t_R, U_R], "../tex/tabellen/R.tex", alignment="CC", split=3)

    # Plot
    fig_R = plt.figure()
    ax1 = fig_R.add_subplot(111)

    ax1.plot(
        t_R,
        U_R,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    ax1.set_xlabel(r'$t / \si{\nano\second}$')
    ax1.set_ylabel(r'$U / \si{\milli\volt}$')

    ax1.legend(loc='best')
    ax1 = ev.plot_layout(ax1)

    fig_R.tight_layout()
    fig_R.savefig("../tex/bilder/R.pdf")
    # plt.show()

    # ===== L bestimmen ================================

    t_L_roh, U_L_roh = np.loadtxt(
        "../messwerte/kurz.csv",
        unpack=True,
        delimiter=","
    )

    # Werte selektieren
    t_L = t_L_roh[(t_L_roh > 1430) & (t_L_roh < 2250)]
    U_L = U_L_roh[(t_L_roh > 1430) & (t_L_roh < 2250)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Daten plotten
    ax.plot(
        t_L,
        U_L,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    ax.set_xlabel(r'$t / \si{\nano\second}$')
    ax.set_ylabel(r'$U / \si{\milli\volt}$')

    ax.legend(loc='best')
    ax = ev.plot_layout(ax)

    fig.tight_layout()
    fig.savefig("../tex/bilder/L.pdf")

    # Fit-Funktion
    def Geradenfkt(x, m, b):
        return m*x + b

    # Offset als Mittelwert der letzten paar Werte
    U_L_off = np.mean(U_L[t_L > 2100])
    U_L_off_tex = r"\SI{{{:.2f}}}{{\milli\volt}}".format(U_L_off)

    ev.write(
        "../tex/gleichungen/teil_d/U_L_off.tex",
        U_L_off_tex
    )

    # Fitdaten vorbeiten
    # -> Negative Werte rausfiltern (Logarithmus), Offset
    check_positive = np.where(U_L - U_L_off > 0.0)
    t_L_fit = t_L[check_positive]
    U_L_fit = U_L[check_positive] - U_L_off

    # Fit
    U_L_log = np.log(U_L_fit)
    val_L, std_L = ev.fit(Geradenfkt, t_L_fit, U_L_log)

    ev.write(
        "../tex/gleichungen/teil_d/L_fit.tex",
        ev.tex_linreg(
            "G_L(t)",
            val_L,
            std_L,
            unit=[r"\per\nano\second", ""],
            # form=["({:L})", "({:L})"]
        )
    )

    # L bestimmen
    m_L = uc.ufloat(val_L[0], std_L[0])  # m=-1/tau, tau in ns
    # L = - (Z0 - R) / uc.ufloat(val_L[0], std_L[0])  # L in nH
    L = - Z0 / uc.ufloat(val_L[0], std_L[0])  # L in nH
    L = L*1e-3  # L in muH
    L_tex = ev.tex_eq(L, unit=r"\micro\henry")

    ev.write("../tex/gleichungen/teil_d/L.tex", L_tex)

    # den Fit plotten
    fig_L = plt.figure()
    ax1 = fig_L.add_subplot(111)

    ax1.plot(
        t_L_fit,
        U_L_log,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    lim = ax1.get_xlim()
    x = np.linspace(lim[0], lim[1], 1000)
    ax1.plot(x, Geradenfkt(x, val_L[0], val_L[1]), color='k', label="Fit")

    ax1.set_xlabel(r'$t / \si{\nano\second}$')
    ax1.set_ylabel(r'$\ln(U - U_\text{off})$')

    ax1.legend(loc='best')
    ax1 = ev.plot_layout(ax1)

    fig_L.tight_layout()
    fig_L.savefig("../tex/bilder/L_fit.pdf")

    # Daten für Plot in Tabelle schreiben
    lt.latextable(
        [
            t_L,
            U_L,
            U_L - U_L_off,
            np.log(U_L - U_L_off)
        ],
        "../tex/tabellen/L.tex",
        alignment="CCCC",
        form=[".0f", ".1f", "0.2f", "0.2f"]
    )

    # ===== Kapazität ==================================

    t_C, U_C = np.loadtxt(
        "../messwerte/offen.csv",
        unpack=True,
        delimiter=","
    )

    # Werte selektieren
    t_C_fit = t_C[(t_C > 1370) & (t_C < 1650)]
    U_C_fit = U_C[(t_C > 1370) & (t_C < 1650)]

    # U0
    U0 = np.mean(U_C[t_C > 2000])
    U0_tex = r"\SI{{{:.2f}}}{{\milli\volt}}".format(U0)

    ev.write(
        "../tex/gleichungen/teil_d/U0.tex",
        U0_tex
    )

    # Daten plotten
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        t_C,
        U_C,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    ax.set_xlabel(r'$t / \si{\nano\second}$')
    ax.set_ylabel(r'$U / \si{\milli\second}$')

    ax.legend(loc='best')
    ax = ev.plot_layout(ax)

    fig.tight_layout()
    fig.savefig("../tex/bilder/C.pdf")

    # Tabelle der Rohdaten schreiben
    lt.latextable(
        [t_C, U_C],
        "../tex/tabellen/C_roh.tex",
        form=[".0f", ".3f"],
        alignment="CC",
        split=3
    )

    # Fit vorbereiten
    U_C_fit_log = np.log(U0 - U_C_fit)

    # Fit-Funktion
    def Geradenfkt(x, m, b):
        return m*x + b

    # Fit
    val_C, std_C = ev.fit(Geradenfkt, t_C_fit, U_C_fit_log)

    ev.write(
        "../tex/gleichungen/teil_d/C_fit.tex",
        ev.tex_linreg("G_C(t)", val_C, std_C, unit=[r"\per\nano\second", ""])
    )

    # Tabelle der Fitdaten
    lt.latextable(
        [t_C_fit, U_C_fit, U0 - U_C_fit, U_C_fit_log],
        "../tex/tabellen/C_fit.tex",
        form=[".0f", ".3f", ".3f", ".3f"],
        alignment="CCCC"
    )

    # C berechnen
    m_C = uc.ufloat(val_C[0], std_C[0])  # m=-1/tau, tau in ns
    C = - 1.0/(m_C*Z0)  # in nF
    C_tex = ev.tex_eq(C, unit=r"\nano\farad")
    # print C_tex

    ev.write("../tex/gleichungen/teil_d/C.tex", C_tex)

    # Plot
    fig_C = plt.figure()
    ax_C = fig_C.add_subplot(111)

    ax_C.plot(
        t_C_fit,
        U_C_fit_log,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    fig_C_lim = ax_C.get_xlim()
    fig_C_x = np.linspace(fig_C_lim[0], fig_C_lim[1], 1000)
    ax_C.plot(
            fig_C_x,
            Geradenfkt(fig_C_x, val_C[0], val_C[1]),
            color='k',
            label="Fit"
    )

    ax_C.set_xlabel(r'$t / \si{\nano\second}$')
    ax_C.set_ylabel(r'$\ln(U)$')

    ax_C.legend(loc='best')
    ax_C = ev.plot_layout(ax_C)

    fig_C.tight_layout()
    fig_C.savefig("../tex/bilder/C_fit.pdf")
    # plt.show()

    # ===== Länge ======================================

    f = 1000.0  # Hz
    f = 117000.0  # Hz

    def laenge(f, phi, e_r):
        l = const.c / (f*np.sqrt(e_r))
        return l*phi / (4.0*np.pi)

    def reflektion(Z, Z0):
        return (Z - Z0) / (Z + Z0)

    # bestimme Z_L
    Z_L = np.complex(R.nominal_value, 2.0*np.pi*f*L.nominal_value*1e-6)

    Z_L_tex = r"\SI{{({:.2f} + {:.2f}i)}}{{\ohm}}".format(
        Z_L.real,
        Z_L.imag
    )

    ev.write("../tex/gleichungen/teil_d/Z_L_L.tex", Z_L_tex)

    # bestimme G_L
    G_L = reflektion(Z_L, Z0)

    G_L_tex = r"{:.2f} + {:.2f}i".format(
        G_L.real,
        G_L.imag
    )

    ev.write("../tex/gleichungen/teil_d/G_L_L.tex", G_L_tex)

    # bestimme Winkel zwischen G_L und G
    phi_L = np.angle(np.complex(G.nominal_value)/G_L)

    # bestimme Länge des Kabels
    laenge_L = laenge(f, phi_L, epslison_r)

    laenge_L_tex = r"\SI{{{:.1f}}}{{\meter}}".format(laenge_L)
    ev.write("../tex/gleichungen/teil_d/laenge_L.tex", laenge_L_tex)

    # bestimme Z_C
    Z_C = np.complex(0, -1.0/(2.0*np.pi*f*C.nominal_value*1e-9))
    G_C = reflektion(Z_C, Z0)
    phi_C = np.angle(np.complex(1.0)/G_C)
    laenge_C = laenge(f, phi_C, epslison_r)

    laenge_C_tex = r"\SI{{{:.1f}}}{{\meter}}".format(laenge_C)
    ev.write("../tex/gleichungen/teil_d/laenge_C.tex", laenge_C_tex)

    pass

# ==================================================
# 	Mehrfachreflektion
# ==================================================


def teil_e():
    data_M = np.loadtxt("../messwerte/mehrfach.csv", delimiter=",")
    t_M, U_M = data_M.transpose()

    U_off_values = U_M[t_M < 80]
    U0_values = U_M[(t_M > 110) & (t_M < 180)]
    U1_values = U_M[(t_M > 210) & (t_M < 285)]
    U2_values = U_M[(t_M > 322) & (t_M < 380)]
    U3_values = U_M[t_M > 432]

    U_off = ev.get_uncert(U_off_values)
    U0_roh = ev.get_uncert(U0_values)
    U1_roh = ev.get_uncert(U1_values)
    U2_roh = ev.get_uncert(U2_values)
    U3_roh = ev.get_uncert(U3_values)

    ev.write(
        "../tex/gleichungen/teil_e/U_off.tex",
        ev.tex_eq(U_off, unit=r"\milli\volt", form="{:L}")
    )
    ev.write(
        "../tex/gleichungen/teil_e/U0_roh.tex",
        ev.tex_eq(U0_roh, unit=r"\milli\volt", form="{:L}")
    )
    ev.write(
        "../tex/gleichungen/teil_e/U1_roh.tex",
        ev.tex_eq(U1_roh, unit=r"\milli\volt", form="{:L}")
    )
    ev.write(
        "../tex/gleichungen/teil_e/U2_roh.tex",
        ev.tex_eq(U2_roh, unit=r"\milli\volt", form="{:L}")
    )
    ev.write(
        "../tex/gleichungen/teil_e/U3_roh.tex",
        ev.tex_eq(U3_roh, unit=r"\milli\volt", form="{:L}")
    )

    U0 = U0_roh - U_off
    U1 = U1_roh - U_off
    U2 = U2_roh - U_off
    U3 = U3_roh - U_off

    dU1 = U1 - U0
    dU2 = U2 - U1
    dU3 = U3 - U2

    G_L = dU1/U0
    G_E = dU3/dU2 + dU2/(U0*(1.0 - G_L))
    G_R = dU3/(dU2*G_E)

    ev.write("../tex/gleichungen/teil_e/G_L.tex", ev.tex_eq(G_L, form="{:L}"))
    ev.write("../tex/gleichungen/teil_e/G_E.tex", ev.tex_eq(G_E, form="{:L}"))
    ev.write("../tex/gleichungen/teil_e/G_R.tex", ev.tex_eq(G_R, form="{:L}"))

    # Plot
    fig_M = plt.figure()
    ax_M = fig_M.add_subplot(111)

    ax_M.plot(
        t_M,
        U_M,
        color='k',
        linestyle='none',
        marker='+',
        label='Messwerte'
    )

    ax_M.set_xlabel(r'$t / \si{\nano\second}$')
    ax_M.set_ylabel(r'$U / \si{\milli\volt}$')

    ax_M.legend(loc='best')
    ax_M = ev.plot_layout(ax_M)

    fig_M.tight_layout()
    # plt.show()

# ==================================================
# 	Ausführen der Versuchsteile
# ==================================================


# teil_a()
# teil_b()
# teil_c()
# teil_d()
# teil_e()
