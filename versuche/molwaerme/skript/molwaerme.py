# -*- coding: iso-8859-1 -*-

# ==================================================
# 	import modules
# ==================================================

from __future__ import print_function

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
from matplotlib.ticker import AutoMinorLocator

# for fitting curves
from scipy import optimize

# constants
import scipy.constants as const

import datetime
import re

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
    print(x + " = ")
    print(eval(x))
    print("\n")


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

fig1 = plt.Figure()
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


# Globale Größen
M_ = 63.55         # g/mol
m_ = 342           # g
kappa_ = 1.378e11  # N/m^3
rho_ = 8.92        # g/cm^3
V0_ = M_ / rho_ * 1e-6    # m^3/mol


def Temp(R_):
    """Temperatur in Abhängigkeit der Widerstandes des Pt100 """
    return 0.00134*R_**2 + 2.296*R_ - 243.02


def Temp_inv(T_):
    """Temp invertiert """
    b = 2.296
    a = 0.00134
    T_C = T_ - 273.15
    return - 0.5 * b/a + np.sqrt((0.5 * b/a)**2 + (243.02 + T_C)/a)


def Energie(U_, I_, dt_):
    """Elektrische Energie """
    return U_*I_*dt_


def Calc_C_p(E_, dT_):
    """Spezifische molare Wärmekapazität """
    return M_/m_ * np.divide(E_, dT_, out=np.zeros_like(E), where=dT_ != 0)


def time2sec(s_):
    """Convert hh:mm:ss to seconds for loadtxt()"""
    h, m, s = re.split(":", s_)
    return int(datetime.timedelta(
        hours=int(h),
        minutes=int(m),
        seconds=int(s)
    ).total_seconds())


def Calc_alpha(T, T_arr, a_arr):
    """Interpolate alpha """
    return np.interp(T, T_arr, a_arr) * 1e-6


def Celsius2kelvin(T):
    """Convert C to K """
    return T + 273.15


def Calc_C_V(C_p, a, k, V0, T):
    """Calculate C_V """
    return C_p - 9*a**2 * k*V0*T

# ====[ Read in data ]======================================


# R = np.loadtxt("../messwerte/", unpack=True)
t, R_rar, I_rar, U_rar, R_Geh, I_Geh, U_Geh = np.loadtxt(
    "../messwerte/messwerte.txt",
    unpack=True,
    converters={0: time2sec}
)

# ====[ Add uncertainties ]=================================

# I_rar = np.array(I_rar[1:])
# U_rar = np.array(U_rar[1:])

R = unp.uarray(R_rar, 0.1)
I = unp.uarray(I_rar, 0.1) * 1e-3  # convert from mA to A
U = unp.uarray(U_rar, 0.01)

# Calc delta time and remove first entry
dt = np.array(np.concatenate(([0], t[1:] - t[:-1])))

# Table of Ausdehnungskoeffizient
T_alpha = np.arange(70, 310, 10)
alpha = np.array([
    7.00,
    8.50,
    9.75,
    10.70,
    11.50,
    12.10,
    12.65,
    13.15,
    13.60,
    13.90,
    14.25,
    14.50,
    14.75,
    14.95,
    15.20,
    15.40,
    15.60,
    15.75,
    15.90,
    16.10,
    16.25,
    16.35,
    16.50,
    16.65
])

# ====[ Energy, Temp, C_p ]=================================


E = Energie(U, I, dt)

T_Celsius = Temp(R)
T_Kelvin = Celsius2kelvin(T_Celsius)
# TODO: delta problem Do 2016/11/03
dT = np.array(np.concatenate(([0], T_Kelvin[1:] - T_Kelvin[:-1])))

C_p = Calc_C_p(E, dT)

# ====[ C_V ]===============================================

alpha_interp = Calc_alpha(unp.nominal_values(T_Kelvin), T_alpha, alpha)
C_V = Calc_C_V(C_p, alpha_interp, kappa_, V0_, T_Kelvin)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(
    unp.nominal_values(T_Kelvin[1:]),
    unp.nominal_values(C_V[1:]),
    yerr=unp.std_devs(C_V[1:]),
    color='k',
    linestyle='none',
    marker='.',
    label='Messwerte'
)

ax.set_xlabel(r'$T~/~\si{\kelvin}$')
ax.set_ylabel(r'$C_V~/~\si{\joule\per\mol\kelvin}$')

# ax.legend(loc='best')
ax.set_axisbelow(True)
lgd = ax.legend(bbox_to_anchor=(1.05, 1.05))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which='major', ls=":", color="0.65")

# fig.tight_layout()
fig.savefig(
    "../tex/bilder/cv.pdf",
    bbox_extra_artists=(lgd,),
    bbox_inches='tight'
)

lt.latextable(
    [R, T_Celsius, T_Kelvin, C_p, alpha_interp*1e5, C_V],
    "../tex/tabellen/C_V",
    form=".3f"
)

# ====[ Debye ]=============================================


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return np.array([array[idx], idx])


def get_debye(A, C_V):
    y = []

    for i in range(len(A)):
        y.append(find_nearest(A[i], C_V))

    y = np.array(y)
    x = find_nearest(y[:, 0], C_V)[1]

    return float("{}.{}".format(int(x), int(y[x, 1])))


debye_array = np.array(
    [
        [24.9430, 24.9310, 24.8930, 24.8310, 24.7450, 24.6340, 24.5000,
         24.3430, 24.1630, 23.9610],
        [23.7390, 23.4970, 23.2360, 22.9560, 22.6600, 22.3480, 22.0210,
         21.6800, 21.3270, 20.9630],
        [20.5880, 20.2050, 19.8140, 19.4160, 19.0120, 18.6040, 18.1920,
         17.7780, 17.3630, 16.9470],
        [16.5310, 16.1170, 15.7040, 15.2940, 14.8870, 14.4840, 14.0860,
         13.6930, 13.3050, 12.9230]
    ]
)

C_V_debye = unp.nominal_values(C_V[(T_Kelvin > 80) & (T_Kelvin < 170)][1:])
T_Kelvin_debye = unp.nominal_values(
    T_Kelvin[(T_Kelvin > 80) & (T_Kelvin < 170)][1:]
)
debye = np.array([get_debye(debye_array, cv) for cv in C_V_debye])
debye_temp = debye * T_Kelvin_debye

lt.latextable(
    [
        T_Kelvin_debye,
        C_V_debye,
        debye,
        debye_temp
    ],
    "../tex/tabellen/debye.tex",
)

ev.write(
    "../tex/gleichungen/debye_average.tex",
    ev.tex_eq(
        uc.ufloat(np.mean(debye_temp), np.std(debye_temp)),
        form="{:.3gL}",
        unit="\kelvin"
    )
)
