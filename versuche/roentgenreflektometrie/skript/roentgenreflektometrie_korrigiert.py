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
from matplotlib.widgets import Slider, Button

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


# plot_data(theta_rho, psd_rho, psd_diff_rho)

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


# plot_geom(theta_rho, psd1, psd2)

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


# ==================================================
# 	Fit
# ==================================================


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


def X_sig_fit(X, z2, n2, n3, sig1, sig2):
    a, j = X
    z = [0.0, z2]
    n = [1.0, n2, n3]
    sig = [sig1, sig2]

    if j == 2:
        return 0
    else:
        e_plus = np.exp(2*1j*k_z(j, n, k, a)*z[j])
        e_minus = np.exp(-2*1j*k_z(j, n, k, a)*z[j])
        divident = (
            r_sig(j, n, k, a, sig[j]) +
            X_sig_fit((a, j+1), z2, n2, n3, sig1, sig2)*e_plus
        )
        divisor = (
            1.0 + r_sig(j, n, k, a, sig[j]) *
            X_sig_fit((a, j+1), z2, n2, n3, sig1, sig2)*e_plus
        )

        return e_minus * divident / divisor


def X_sig_fit_squared(X, z2, n2, n3, sig1, sig2):
    return np.abs(X_sig_fit(X, z2, n2, n3, sig1, sig2))**2


k = 2.0*np.pi / 1.54e-10
z1 = 0.0
z2 = -230.0e-10
n1 = 1.0
n2 = 1.0 - 1.5 * 1e-06
n3 = 1.0 - 3.0 * 1e-06
sig1 = 8.0 * 1e-10
sig2 = 3.0 * 1e-10
n = [n2, n3]
z = [z1, z2]
sig = [sig1, sig2]

sel = (theta > 0.25) & (theta < 0.7)
theta_fit = theta[sel]
psd3_fit = psd3[sel]
sigma_fit = len(theta_fit) * [0.00001]

val, cor = optimize.curve_fit(
    X_sig_fit_squared,
    (np.deg2rad(theta_fit), 0),
    psd3_fit,
    p0=[z2, n2, n3, sig1, sig2],
    maxfev=100000,
    sigma=sigma_fit
)
print val

# theta_plot = np.deg2rad(theta)
theta_plot = np.deg2rad(theta)
theta_test = np.linspace(0.1, 2.5, 200)
theta_test_rad = np.deg2rad(theta_test)

values = [z2, n2, n3, sig1, sig2]

fig = plt.figure()
ax = fig.add_axes([0.15, 0.4, 0.75, 0.6])

axcolor = 'lightgoldenrodyellow'
ax_z2 = plt.axes([0.25, 0.3, 0.60, 0.03], axisbg=axcolor)
ax_n2 = plt.axes([0.25, 0.25, 0.60, 0.03], axisbg=axcolor)
ax_n2.set_xscale("log")
ax_n3 = plt.axes([0.25, 0.2, 0.60, 0.03], axisbg=axcolor)
ax_n3.set_xscale("log")
ax_sig1 = plt.axes([0.25, 0.15, 0.60, 0.03], axisbg=axcolor)
ax_sig1.set_xscale("log")
ax_sig2 = plt.axes([0.25, 0.1, 0.60, 0.03], axisbg=axcolor)
ax_sig2.set_xscale("log")

ax_fit = plt.axes([0.1, 0.25, 0.05, 0.05])
b_fit = Button(ax_fit, "Fit")

s_z2 = Slider(ax_z2, "-z2", 120, 250, valinit=180)
s_n2 = Slider(
    ax_n2,
    "n2",
    1.0e-7,
    1.0e-4,
    valinit=4.59e-6,
    valfmt="1.0 - %.2e"
)
s_n3 = Slider(
    ax_n3,
    "n3",
    1.0e-7,
    1.0e-4,
    valinit=1.6e-6,
    valfmt="1.0 - %.2e"
)
s_sig1 = Slider(
    ax_sig1,
    "sig1",
    1.0e-11,
    5.0e-9,
    valinit=3.9e-10,
    valfmt="%.2e"
)
s_sig2 = Slider(
    ax_sig2,
    "sig2",
    1.0e-11,
    1.0e-09,
    valinit=1.4e-10,
    valfmt="%.2e"
)

l, = ax.semilogy(
    theta_test,
    X_sig_fit_squared(
        (theta_test_rad, 0),
        *values
    ),
    color='k',
    linestyle='-',
    label='Fit'
)

s_z2.set_val(-val[0]*1e10)
s_n2.set_val(1.0 - val[1])
s_n3.set_val(1.0 - val[2])
s_sig1.set_val(val[3])
s_sig2.set_val(val[4])

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


def update(val):
    z2 = -s_z2.val * 1e-10
    n2 = 1.0 - s_n2.val
    n3 = 1.0 - s_n3.val
    sig1 = s_sig1.val
    sig2 = s_sig2.val

    l.set_ydata(X_sig_fit_squared(
        (theta_test_rad, 0), z2, n2, n3, sig1, sig2)
    )
    plt.draw()

s_z2.on_changed(update)
s_n2.on_changed(update)
s_n3.on_changed(update)
s_sig1.on_changed(update)
s_sig2.on_changed(update)


def update_button(value):
    z2 = -s_z2.val * 1e-10
    n2 = 1.0 - s_n2.val
    n3 = 1.0 - s_n3.val
    sig1 = s_sig1.val
    sig2 = s_sig2.val

    val, cor = optimize.curve_fit(
        X_sig_fit_squared,
        (np.deg2rad(theta_fit), 0),
        psd3_fit,
        p0=[z2, n2, n3, sig1, sig2],
        maxfev=100000,
        sigma=sigma_fit
    )
    l.set_ydata(X_sig_fit_squared(
        (theta_plot, 0), *val)
    )
    print val
    s_z2.set_val(-val[0]*1e10)
    s_n2.set_val(1.0 - val[1])
    s_n3.set_val(1.0 - val[2])
    s_sig1.set_val(val[3])
    s_sig2.set_val(val[4])

b_fit.on_clicked(update_button)

# fig.tight_layout()
# fig.savefig("../tex/bilder/fit.pdf")
plt.show()


# [ -2.17355393e-08   9.99994907e-01   9.99991870e-01   3.19229034e-12
#   -7.83730593e-12]
# [ -2.14368032e-08   9.99995124e-01   9.99991885e-01   9.46574935e-11
#    1.59432838e-14]

# ==================================================
# 	Daten schreiben
# ==================================================

# n1_tex = ev.tex_eq(1.0 - uc.ufloat(val[0], std[0]))
# n2_tex = ev.tex_eq(1.0 - uc.ufloat(val[1], std[1]))
# sig1_tex = ev.tex_eq(uc.ufloat(-val[3], std[3]))
# sig2_tex = ev.tex_eq(uc.ufloat(val[4], std[4]))
# z2_tex = ev.tex_eq(uc.ufloat(val[2], std[2])*1e10, unit=r"\angstrom")

# show(n1_tex)
# show(n1_tex)
# show(n2_tex)
# show(sig1_tex)
# show(sig2_tex)
# show(z2_tex)

# ev.write("../tex/gleichungen/n1.tex", n1_tex)
# ev.write("../tex/gleichungen/n2.tex", n2_tex)
# ev.write("../tex/gleichungen/sig1.tex", sig1_tex)
# ev.write("../tex/gleichungen/sig2.tex", sig2_tex)
# ev.write("../tex/gleichungen/z2.tex", z2_tex)
