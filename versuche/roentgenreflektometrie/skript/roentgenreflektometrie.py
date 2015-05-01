# -*- coding: iso-8859-1 -*-

# ==================================================
# 	import modules
# ==================================================

import sys
import os

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

theta_rho, psd_rho = np.loadtxt(
    "../messwerte/messung.txt",
    unpack=True
)

theta_diff_rho, psd_diff_rho = np.loadtxt(
    "../messwerte/diffus.txt",
    unpack=True
)


###########################################################################
#                                Aufgabe 8                                #
###########################################################################

# ==================================================
# 	Diffusen Scan von Reflektivitätsscan
# 	abziehen
# ==================================================

psd1 = psd_rho - psd_diff_rho


def plot_data(theta, psd, psd_diff):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(
        theta,
        psd,
        color='k',
        linestyle='none',
        marker='+',
        markersize=2,
        label='Reflexionsscan'
    )

    ax.semilogy(
        theta,
        psd_diff,
        color='k',
        linestyle='none',
        marker='x',
        markersize=2,
        label='Diffuser Scan'
    )

    ax.set_xlabel(r'$\alpha$ in \si{\deg}')
    ax.set_ylabel(r'Intensit\"at')

    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig("../tex/bilder/data.pdf")


plot_data(theta_rho, psd_rho, psd_diff_rho)

# ==================================================
# 	Geometriewinkel
# ==================================================

d0 = 0.1  # mm
D = 30  # mm

alpha_g = np.arcsin(d0/D)
alpha_g_deg = np.rad2deg(alpha_g)

show(alpha_g_deg)


def Geom(a):
    return D*np.sin(np.deg2rad(a))/d0


sel = theta_rho < alpha_g_deg
sel[0] = False  # durch Null teilen verhindern
psd2 = np.array(psd1)
psd2[sel] = psd1[sel] / Geom(theta_rho[sel])

# ===== Plot =======================================


def plot_geom(theta_rho, psd1, psd2):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(
        theta_rho,
        psd1,
        color='k',
        linestyle='none',
        marker=4,
        markersize=2,
        label='ohne Geometriefaktor'
    )

    ax.semilogy(
        theta_rho,
        psd2,
        color='k',
        linestyle='none',
        marker='+',
        markersize=2,
        label='mit Geometriefaktor'
    )

    ax.set_xlabel(r'$\alpha$ in \si{\deg}')
    ax.set_ylabel(r'Intensit\"at')

    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig("../tex/bilder/geometriefaktor.pdf")

    plt.show()


plot_geom(theta_rho, psd1, psd2)

# ==================================================
# 	Normierung
# ==================================================

# Schneide alles unterhalb des Plateaus ab
sel = theta_rho > 0.082

theta = theta_rho[sel]
psd3 = psd2[sel]

# Mittlere höhe des Plateaus
sel = theta < 0.168

N = np.mean(psd3[sel])

# Normierung
psd3 = psd3 / N


def plot_norm(theta, psd3):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(
        theta,
        psd3,
        color='k',
        linestyle='-',
        marker='+',
        markersize=2,
        label='Messwerte'
    )

    ax.set_xlabel(r'$\alpha$ in \si{\deg}')
    ax.set_ylabel(r'psd')

    ax.legend(loc='best')

    fig.tight_layout()

    plt.show()


# plot_norm(theta, psd3)

###########################################################################
#                                Aufgabe 9                                #
###########################################################################


def k_z(j, n, k, a):
    Return = k*np.sqrt(n[j]**2 - np.cos(a)**2)
    return Return
    # return k*np.sqrt(n[j]**2 - np.cos(a)**2)


def r_sig(j, n, k, a, sig):
    k_j = k_z(j, n, k, a)
    k_j1 = k_z(j+1, n, k, a)
    divident = k_j - k_j1
    divisor = k_j + k_j1
    exponent = -2.0*k_j*k_j1*sig**2

    return divident / divisor * np.exp(exponent)


def X_sig(a, j, k, n, z, sig):
    if j == 2:
        return 0
    else:
        e_plus = np.exp(2*1j*k_z(j, n, k, a)*z[j])
        e_minus = np.exp(-2*1j*k_z(j, n, k, a)*z[j])
        divident = (
            r_sig(j, n, k, a, sig[j]) +
            X_sig(a, j+1, k, n, z, sig)*e_plus
        )
        divisor = (
            1.0 + r_sig(j, n, k, a, sig[j]) *
            X_sig(a, j+1, k, n, z, sig)*e_plus
        )

        return e_minus * divident / divisor


def X_sig_squared(a, j, k, n, z, sig):
    return np.abs(X_sig(a, j, k, n, z, sig))**2


k = 2.0*np.pi / 1.54e-10
z1 = 0.0
z2 = -209.0e-10
n1 = 1.0

#  Vielleicht
# n1 = 9.99999669e-01
# n2 = 9.99997764e-01
# n3 = 9.99991790e-01
# z2 = -2.10454391e-08
# sig1 = 3.65257401e-10
# sig2 = 6.87502927e-10

n1 = 9.99991619e-01
n2 = 9.99995656e-01
n3 = 1.00000481e+00
z2 = -2.10454391e-08
sig1 = 3.10215472e-10
sig2 = 3.98169509e-10

n = [n1, n2, n3]
z = [z1, z2]
sig = [sig1, sig2]

a = np.deg2rad(theta)
X_a = X_sig_squared(a, 0, k, n, z, sig)


def plot_X_variiert(theta, X_a, psd3):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(
        theta,
        X_a,
        color='k',
        linestyle='-',
        # marker='+',
        label='Messwerte'
    )

    ax.semilogy(
            theta,
            psd3,
            color='k',
            # linestyle='-',
            marker='+',
            markersize=2,
            label='Messwerte'
        )

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.legend(loc='best')

    fig.tight_layout()
    plt.show()

# ==================================================
# 	Fit
# ==================================================


def X_sig_fit(X, n1, n2, n3, z2, sig1, sig2):
    a, j, k = X
    z = [0.0, z2]
    n = [n1, n2, n3]
    sig = [sig1, sig2]

    if j == 2:
        return 0
    else:
        e_plus = np.exp(2*1j*k_z(j, n, k, a)*z[j])
        e_minus = np.exp(-2*1j*k_z(j, n, k, a)*z[j])
        divident = (
            r_sig(j, n, k, a, sig[j]) +
            X_sig_fit((a, j+1, k), n1, n2, n3, z2, sig1, sig2)*e_plus
        )
        divisor = (
            1.0 + r_sig(j, n, k, a, sig[j]) *
            X_sig_fit((a, j+1, k), n1, n2, n3, z2, sig1, sig2)*e_plus
        )

        return e_minus * divident / divisor


def X_sig_fit_squared(X, n1, n2, n3, z2, sig1, sig2):
    return np.abs(X_sig_fit(X, n1, n2, n3, z2, sig1, sig2))**2


z1 = 0.0
z2 = -209.0e-10
n1 = 1.0
n2 = 1.0
n3 = 1.0
sig1 = 3.0e-10
sig2 = 6.0e-10

n = [n1, n2, n3]
z = [z1, z2]
sig = [sig1, sig2]

sel = (theta > 1.0) & (theta < 2.0)
theta_fit = theta[sel]
psd3_fit = psd3[sel]

val, cor = optimize.curve_fit(
    X_sig_fit_squared,
    (np.deg2rad(theta_fit), 0, k),
    psd3_fit,
    p0=[n1, n2, n3, z2, sig1, sig2],
    maxfev=10000,
)
std = ev.get_std(cor)


def plot_X_fitted(val, theta, psd3):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.semilogy(
        theta,
        X_sig_fit_squared(
            (np.deg2rad(theta), 0, k),
            val[0],
            val[1],
            val[2],
            val[3],
            val[4],
            val[5]
        ),
        color='k',
        linestyle='-',
        # marker='+',
        label='Fit'
    )

    ax.semilogy(
            theta,
            psd3,
            color='k',
            linestyle='none',
            marker='+',
            markersize=2,
            label='Messwerte'
        )

    ax.set_xlabel(r'$\alpha$ in \si{\deg}')
    ax.set_ylabel(r'Intensit\"at')

    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig("../tex/bilder/fit.pdf")
    plt.show()


plot_X_fitted(val, theta, psd3)

# ==================================================
# 	Daten schreiben
# ==================================================

n1_tex = ev.tex_eq(1.0 - uc.ufloat(val[0], std[0]))
n2_tex = ev.tex_eq(1.0 - uc.ufloat(val[1], std[1]))
n3_tex = ev.tex_eq(1.0 - uc.ufloat(val[2], std[2]))
sig1_tex = ev.tex_eq(uc.ufloat(-val[4], std[4]))
sig2_tex = ev.tex_eq(uc.ufloat(val[5], std[5]))
z2_tex = ev.tex_eq(uc.ufloat(val[3], std[3])*1e10, unit=r"\angstrom")

show(n1_tex)
show(n1_tex)
show(n2_tex)
show(n3_tex)
show(sig1_tex)
show(sig2_tex)
show(z2_tex)

ev.write("../tex/gleichungen/n1.tex", n1_tex)
ev.write("../tex/gleichungen/n2.tex", n2_tex)
ev.write("../tex/gleichungen/n3.tex", n3_tex)
ev.write("../tex/gleichungen/sig1.tex", sig1_tex)
ev.write("../tex/gleichungen/sig2.tex", sig2_tex)
ev.write("../tex/gleichungen/z2.tex", z2_tex)
