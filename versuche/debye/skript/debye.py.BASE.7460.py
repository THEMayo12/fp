#-*- coding: iso-8859-1 -*-

# ==================================================
#	import modules
# ==================================================

# import sys

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
#	settings
# ==================================================


sp.params["text.latex.preamble"] = sp.tex_fouriernc_preamble
plt.rcParams.update(sp.params)

ev.use_unitpkg("siunitx")

# ==================================================
#	function to print equations with
#	matplotlib
# ==================================================


def show(x) :
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

# ==================================================
#	start evaluation
# ==================================================
#werte einlesen
r1 = ev.get_data("../messwerte/messwerte1.txt", unpack=True, index=0)# in mm
r2 = ev.get_data("../messwerte/messwerte2.txt", unpack=True, index=0)# in mm

l1 =1.54093*1e-10 #in m
l2 =1.54478*1e-10 #in m
R=57.3 #in mm
#winkel berechnen, Faktor 2 wegen Bragg
t1=r1/(2.*R)
t2=r2/(2.*R)

print str(t1) + "t1" 
print str(t2) + "t2"
#Bragg: nl=2dsint , Ann:n=_1 ? => d= l/2sint
d11=l1/(2.*np.sin(t1))
d12=l1/(2.*np.sin(t2))
d21=l2/(2.*np.sin(t1))
d22=l2/(2.*np.sin(t2))

#Es gilt: sint(t)^2 propto h^2+k^2+l^2 , def s=sin(t)**2
s1=np.sin(t1)**2
s2=np.sin(t2)**2

s1=4.*s1/(s1[0])
s2=4.*s2/(s2[0])

print "h**2+k**2+l**2 =:s"
print "s1" + str(s1) + "passt zu bcc"
print "s2" + str(s2) + "passt zu diamant"
#exp. Berechnung mit theor. Werten vergleichen, um abzuschätzen welchen Wert für s man nehmen muss

print "Die Netzebenenabstände ergeben:"
print "d1" + str(d11)
print "d2" + str(d12)

sth1=[4,6,8,10,16,20,22]
sth2=[4,11]
rs1=np.sqrt(sth1)
rs2=np.sqrt(sth2)

a1=d11*rs1
a2=d12*rs2
print "daraus ergeben sich die Gitterkonstanten (sollten zu einer Probe je konstant sein):"
print "a1" + str(a1)
print "a2" + str(a2)



#vergleich mit liste h^2+k^2+l^2 => Vermutung: 1 ist bcc, 2 ist diamant

# es gilt exakt l=2 a/sqrt(s) sin(t) <=> a=l*sqrt(s)

a1_uc = ev.get_uncert(a1)
a2_uc = ev.get_uncert(a2)

print a1_uc
print a2_uc

#Regression a gegen cos^2 t 


def G(x, m, b):
    return m*x + b

val1, std1 = ev.fit(G, np.cos(t1)**2,a1 )
val2, std2 = ev.fit(G, np.cos(t1)**2,a1 )  ###

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(np.cos(t1)**2, a1, linestyle = 'none', marker = '+', label = 'Messwerte')
lim = ax1.get_xlim()
x = np.linspace(lim[0], lim[1], 1000)
ax1.plot(x, G(x, val1[0], val1[1]), label="Fit")

ax1.set_xlabel(r'$\cos^2(\vartheta)$')
ax1.set_ylabel(r'$a / \si{\meter}$')

ax1.legend(loc = 'best')
ax1 = ev.plot_layout(ax1)

fig1.tight_layout()

#######

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

ax2.plot(np.cos(t1)**2, a1, linestyle = 'none', marker = '+', label = 'Messwerte')
lim = ax2.get_xlim()
x = np.linspace(lim[0], lim[1], 1000)
ax2.plot(x, G(x, val2[0], val2[1]), label="Fit")

ax2.set_xlabel(r'$\cos^2(\vartheta)$')
ax2.set_ylabel(r'$a / \si{\meter}$')

ax2.legend(loc = 'best')
ax2 = ev.plot_layout(ax2)

fig2.tight_layout()


#Bilder speichern
fig1.savefig('../tex/bilder/fig1.pdf')
fig2.savefig('../tex/bilder/fig2.pdf')



# LateX-Tabelle erzeugen
table1 = lt.latextable(
    [r1,t1,s1,sth1, a1],
    "table",
    alignment = 'CCCCC',
    form = '.3f',
)
table2 = lt.latextable(
    [r1,t1,s1,sth1, a1],
    "table",
    alignment = 'CCCCC',
    form = '.3f',
)

ev.write('../tex/tabellen/tabelle1.tex', table1)
ev.write('../tex/tabellen/tabelle2.tex', table2)

# latex-Gleichung der linearen Regression
lin_reg1 = ev.tex_linreg(
        "G_1(t)",
        val1,
        std1,
        unit = ["", "\meter"]
)

ev.write('../tex/gleichungen/gerade1.tex', lin_reg1)

lin_reg2 = ev.tex_linreg(
        "G_2(t)",
        val2,
        std2,
        unit = ["", "\meter"]
)

ev.write('../tex/gleichungen/gerade2.tex', lin_reg2)


