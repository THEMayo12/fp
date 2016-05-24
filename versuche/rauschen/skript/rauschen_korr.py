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

V_N_eigen, U_a_eigen = np.loadtxt("../messwerte/kurzschluss_selektiv.txt", unpack=True)
U_a_eigen = U_a_eigen/(10.*(1000.*V_N_eigen)**2)






#external input parameters
A = uc.ufloat(0.3868,0.01) *1000 #kHz in Hz
T = 300. #K
C_eigen = 0.1*100*10**(-12) #Kabellänge*pF/Meter
nu_mittel =25000 #Hz

#=========================================================
#     Rauschen R_1000
#=========================================================

R1, U1 = np.loadtxt("../messwerte/R_1000_sel.txt", unpack=True)

tab1 = lt.latextable(
    [R1, U1],
    "../tex/tabellen/rauschen_korr1.tex",
    alignment = 'CC',
    form = '.3f',
)

#Verstärkungsfaktoren rausrechnen
v11 = 10.*(1000.*10.*50.)**2 #Verstärkungsfaktor
v12 = 10.*(1000.*10.*20.)**2 #Verstärkungsfaktor

for i in range(0,len(U1)):
	if i<=(len(U1)-6):
		U1[i]=(1./(1.+2.*const.pi*R1[i]*nu_mittel*C_eigen)*U1[i])/v11-U_a_eigen[3]
	else:
		U1[i]=(1./(1.+2.*const.pi*R1[i]*nu_mittel*C_eigen)*U1[i])/v12-U_a_eigen[3]






#Ausgleichrechnung
# latex-Gleichung der linearen Regression
# Geradenfunktion
def G(x, m, b):
    return m*x + b

val1, cov1 = optimize.curve_fit(G, R1[7:], U1[7:])
std1 = ev.get_std(cov1)

lin_reg1 = ev.tex_linreg(
        "U_a",
        val1,
        std1,
        unit = ["\volt^2\per\ohm", "\meter"]
)


#print_tex(ev.latexEq(lin_reg1))  			 ### <---


#steigung
m1=uc.ufloat(val1[0],std1[0])
print(m1)
k_korr1 = m1/(4*T*A)
print(k_korr1)
# tex schreiben
ev.write('../tex/tabellen/k_korr1', str(k_korr1))		### <---






#Plotten
fig1 = plt.figure()
ax = fig1.add_subplot(111)

ax.plot(
    R1,
    U1,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R_1000$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig1.tight_layout()
fig1.savefig('../tex/bilder/rauschen_korr1.pdf')






#=========================================================
#     Rauschen R_100k
#=========================================================

R2, U2 = np.loadtxt("../messwerte/R_100k_sel.txt", unpack=True)

tab2 = lt.latextable(
    [R2, U2],
    "../tex/tabellen/rauschen_korr2.tex",
    alignment = 'CC',
    form = '.3f',
)


#Verstärkungsfaktoren rausrechnen
v2 = 10.*(1000.*200.)**2 #Verstärkungsfaktor
U2 = (1./(1.+2.*const.pi*R2*nu_mittel*C_eigen)*U2)/v2-U_a_eigen[4]





#Ausgleichrechnung
# latex-Gleichung der linearen Regression
# Geradenfunktion
def G(x, m, b):
    return m*x + b

val2, cov2 = optimize.curve_fit(G, 1000*R2[:14], U2[:14]) #  kOhm to Ohm, only linear part
std2 = ev.get_std(cov1)

lin_reg2 = ev.tex_linreg(
        "U_a",
        val2,
        std2,
        unit = ["\volt^2\per\ohm", "\meter"]
)

#print_tex(ev.latexEq(lin_reg2))

#steigung
m2=uc.ufloat(val2[0],std2[0])
print(m2)
k_korr2 = m2/(4*T*A)
print(k_korr2)
# tex schreiben
ev.write('../tex/tabellen/k_korr2', str(k_korr2))




#Plotten
fig2 = plt.figure()
ax = fig2.add_subplot(111)

ax.plot(
    R2,
    U2,
    color='k',
    linestyle='none',
    marker='+',
    label='Messwerte'
)

ax.set_xlabel(r'$R_100k$')
ax.set_ylabel(r'$U_a$ in $\si{\volt}$')

ax.legend(loc='best')
ax = ev.plot_layout(ax)

fig2.tight_layout()
fig2.savefig('../tex/bilder/rauschen_korr2.pdf')



