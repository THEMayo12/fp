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
from scipy.special import erf

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

# Geradenfunktion
def G(x, m, b):
    return m*x + b
# Konstantenfunktion
def C(x,Konst):
	return Konst 
#complementäre Errorfunktion mit limes-inf =a,lim+inf =c, mittelwert bei b 
def erfc(x,a,b,c,d):
	return ((a-c)/2.)*(1.-erf(d*(x-b)))+c
# Angepasste Heaviside
def H(x,a,b,c):
#	elif (x<minil):
#		return 0
	if (x<b):
		return a

#	elif (x>400):
#		return 0
	else:
		return c
# Dreiecksfunktion, normiert
def Dr(x,a,b):
	if (x<a):
		return 0
	elif (x>b):
		return 0
	else:
		c=(a+b)/2.
		y=x-c
		A=a-c
		B=b-c
		n=1./B
		if (y<0):
			return (-n/A*y+n)
		else:
			return (n-n/B*y)
# Falschrummer Topf
def Topf(x,a,b,c):
	if (x<a):
		return 0
	elif (x>c):
		return 0
	else:
		return b
#Diskrete Faltung
def Falt(A,B):
	return np.convolve(A, B,'same')

#Daten einlesen , Index^=Wuerfel ,T=livetimes, C=counts

T0,C0 = ev.get_data("../messwerte/Peak0.txt", unpack=True, index=0)
T1,C1 = ev.get_data("../messwerte/Peak1.txt", unpack=True, index=0)
T2,C2 = ev.get_data("../messwerte/Peak2.txt", unpack=True, index=0)
T3,C3 = ev.get_data("../messwerte/Peak3.txt", unpack=True, index=0)

Tinit=45.14	#ohne Wuerfel
Cinit=10236. 	#ohne Wuerfel


#Spektren einlesen , K:"Kanal", C:"Counts"
K02,C02 = ev.get_data("../messwerte/C02.dat", unpack=True, index=0)
K08,C08 = ev.get_data("../messwerte/C08.dat", unpack=True, index=0)
K09,C09 = ev.get_data("../messwerte/C09.dat", unpack=True, index=0)

K11,C11 = ev.get_data("../messwerte/C11.dat", unpack=True, index=0)
#K12,C12 = ev.get_data("../messwerte/C12.dat", unpack=True, index=0) fehlt
K13,C13 = ev.get_data("../messwerte/C13.dat", unpack=True, index=0)
K14,C14 = ev.get_data("../messwerte/C14.dat", unpack=True, index=0)
K15,C15 = ev.get_data("../messwerte/C15.dat", unpack=True, index=0)
K16,C16 = ev.get_data("../messwerte/C16.dat", unpack=True, index=0)
K17,C17 = ev.get_data("../messwerte/C17.dat", unpack=True, index=0)
K18,C18 = ev.get_data("../messwerte/C18.dat", unpack=True, index=0)
K19,C19 = ev.get_data("../messwerte/C19.dat", unpack=True, index=0)
K110,C110 = ev.get_data("../messwerte/C110.dat", unpack=True, index=0)
K111,C111 = ev.get_data("../messwerte/C111.dat", unpack=True, index=0)
K112,C112 = ev.get_data("../messwerte/C112.dat", unpack=True, index=0)

K21,C21 = ev.get_data("../messwerte/C21.dat", unpack=True, index=0)
K22,C22 = ev.get_data("../messwerte/C22.dat", unpack=True, index=0)
K23,C23 = ev.get_data("../messwerte/C23.dat", unpack=True, index=0)
K24,C24 = ev.get_data("../messwerte/C24.dat", unpack=True, index=0)
K25,C25 = ev.get_data("../messwerte/C25.dat", unpack=True, index=0)
K26,C26 = ev.get_data("../messwerte/C26.dat", unpack=True, index=0)
K27,C27 = ev.get_data("../messwerte/C27.dat", unpack=True, index=0)
K28,C28 = ev.get_data("../messwerte/C28.dat", unpack=True, index=0)
K29,C29 = ev.get_data("../messwerte/C29.dat", unpack=True, index=0)
K210,C210 = ev.get_data("../messwerte/C210.dat", unpack=True, index=0)
K211,C211 = ev.get_data("../messwerte/C211.dat", unpack=True, index=0)
K212,C212 = ev.get_data("../messwerte/C212.dat", unpack=True, index=0)

K31,C31 = ev.get_data("../messwerte/C31.dat", unpack=True, index=0)
K32,C32 = ev.get_data("../messwerte/C32.dat", unpack=True, index=0)
K33,C33 = ev.get_data("../messwerte/C33.dat", unpack=True, index=0)
K34,C34 = ev.get_data("../messwerte/C34.dat", unpack=True, index=0)
K35,C35 = ev.get_data("../messwerte/C35.dat", unpack=True, index=0)
K36,C36 = ev.get_data("../messwerte/C36.dat", unpack=True, index=0)
K37,C37 = ev.get_data("../messwerte/C37.dat", unpack=True, index=0)
K38,C38 = ev.get_data("../messwerte/C38.dat", unpack=True, index=0)
K39,C39 = ev.get_data("../messwerte/C39.dat", unpack=True, index=0)
K310,C310 = ev.get_data("../messwerte/C310.dat", unpack=True, index=0)
K311,C311 = ev.get_data("../messwerte/C311.dat", unpack=True, index=0)
K312,C312 = ev.get_data("../messwerte/C312.dat", unpack=True, index=0)

#sinnvoll zusammenfassen
X0=[K02,K08,K09]
X1=[K11,K13,K14,K15,K16,K17,K18,K19,K110,K111,K112]
X2=[K21,K22,K23,K24,K25,K26,K27,K28,K29,K210,K211,K212]
X3=[K31,K32,K33,K34,K35,K36,K37,K38,K39,K310,K311,K312]
Y0=[C02,C08,C09]
Y1=[C11,C13,C14,C15,C16,C17,C18,C19,C110,C111,C112]
Y2=[C21,C22,C23,C24,C25,C26,C27,C28,C29,C210,C211,C212]
Y3=[C31,C32,C33,C34,C35,C36,C37,C38,C39,C310,C311,C312]
#Geometriematrix
s=np.sqrt(2)
A= np.matrix((
(0,0,0,0,0,0,1,1,1),
(0,0,0,1,1,1,0,0,0),
(1,1,1,0,0,0,0,0,0),
(1,0,0,1,0,0,1,0,0),
(0,1,0,0,1,0,0,1,0),
(0,0,1,0,0,1,0,0,1),
(0,s,0,s,0,0,0,0,0),
(0,0,s,0,s,0,s,0,0),
(0,0,0,0,0,s,0,s,0),
(0,s,0,0,0,s,0,0,0),
(s,0,0,0,s,0,0,0,s),
(0,0,0,s,0,0,0,s,0)
))
#Und weil C12 vergessen wurde:
B=np.matrix((
(0,0,0,0,0,0,1,1,1),
(1,1,1,0,0,0,0,0,0),
(1,0,0,1,0,0,1,0,0),
(0,1,0,0,1,0,0,1,0),
(0,0,1,0,0,1,0,0,1),
(0,s,0,s,0,0,0,0,0),
(0,0,s,0,s,0,s,0,0),
(0,0,0,0,0,s,0,s,0),
(0,s,0,0,0,s,0,0,0),
(s,0,0,0,s,0,0,0,s),
(0,0,0,s,0,0,0,s,0)
))

A=A*1. #in cm
B=B*1.

#Bem: A.transpose(), np.linalg,inv(A)

#Grenzen des Strahlungspeaks
minil=220
maxil=257
minir=315



#Eigener Untergrund-Fit (K02, C02)
val0l=[]
val1l=[]
val2l=[]
val3l=[]
val0r=[]
val1r=[]
val2r=[]
val3r=[]


for i in range (0,len(X0)):
	vall, covl = optimize.curve_fit(C, X0[i][(X0[i]>minil)&(X0[i]<maxil)], Y0[i][(X0	[i]>minil)&(X0[i]<maxil)])
	valr, covr = optimize.curve_fit(C, X0[i][X0[i]>minir], Y0[i][X0[i]>minir])
	val0l.append(vall[0])
	val0r.append(valr[0])

for i in range (0,len(X1)):
	vall, covl = optimize.curve_fit(C, X1[i][(X1[i]>minil)&(X1[i]<maxil)], Y1[i][(X1	[i]>minil)&(X1[i]<maxil)])
	valr, covr = optimize.curve_fit(C, X1[i][X1[i]>minir], Y1[i][X1[i]>minir])
	val1l.append(vall[0])
	val1r.append(valr[0])

for i in range (0,len(X2)):
	vall, covl = optimize.curve_fit(C, X2[i][(X2[i]>minil)&(X2[i]<maxil)], Y2[i][(X2	[i]>minil)&(X2[i]<maxil)])
	valr, covr = optimize.curve_fit(C, X2[i][X2[i]>minir], Y2[i][X2[i]>minir])
	val2l.append(vall[0])
	val2r.append(valr[0])

for i in range (0,len(X3)):
	vall, covl = optimize.curve_fit(C, X3[i][(X3[i]>minil)&(X3[i]<maxil)], Y3[i][(X3	[i]>minil)&(X3[i]<maxil)])
	valr, covr = optimize.curve_fit(C, X3[i][X3[i]>minir], Y3[i][X3[i]>minir])
	val3l.append(vall[0])
	val3r.append(valr[0])

'''
He=[]
D=[]
for i in range (0,len(K02)):
	He.append(H(i,val02l,K02[np.argmax(C02)],val02r))
	D.append(Dr(i,maxil, minir))
X=Falt(D,He)
'''
#Untergrund rausrechnen:
C0=[] # C0[i] :"Korrigierte Counts in i-Richtung"
C1=[]
C2=[]
C3=[]

for j in range(0,len(X0)):
	RE=0.
	for i in range(maxil-1,minir):
		RE=RE+Y0[j][i]-erfc(X0[j][i],val0l[j],X0[j][np.argmax(Y0[j])],val0r[j],(3.*1./(minir-maxil)/2.))
	
	C0.append(RE)

for j in range(0,len(X1)):
	RE=0.
	for i in range(maxil-1,minir):
		RE=RE+Y1[j][i]-erfc(X1[j][i],val1l[j],X1[j][np.argmax(Y1[j])],val1r[j],(3.*1./(minir-maxil)/2.))
	
	C1.append(RE)
for j in range(0,len(X2)):
	RE=0.
	for i in range(maxil-1,minir):
		RE=RE+Y2[j][i]-erfc(X2[j][i],val2l[j],X2[j][np.argmax(Y2[j])],val2r[j],(3.*1./(minir-maxil)/2.))
	
	C2.append(RE)
for j in range(0,len(X3)):
	RE=0.
	for i in range(maxil-1,minir):
		RE=RE+Y3[j][i]-erfc(X3[j][i],val3l[j],X3[j][np.argmax(Y3[j])],val3r[j],(3.*1./(minir-maxil)/2.))
	
	C3.append(RE)



#Intensität=counts/second
Iinit=Cinit/Tinit
I0=C0/T0

I1=[]
'''
for i in range (0,len(T1)):
	if T1[i]==0:
		I1.append(0)
	else:
		I1.append(C1[i]/T1[i])
'''
I1=C1/T1
I2=C2/T2
I3=C3/T3

# LateX-Tabelle erzeugen
#Richtungen 
R0=[2,8,9]
R1=[1,3,4,5,6,7,8,9,10,11,12]
R2=[1,2,3,4,5,6,7,8,9,10,11,12]

t0 = lt.latextable(
    [R0,T0, C0],
    "table",
    alignment = 'CCC',
    form = '.2f', )
# tex schreiben
ev.write('tabelle0', t0)

t1 = lt.latextable(
    [R1,T1, C1],
    "table",
    alignment = 'CCC',
    form = '.2f', )
# tex schreiben
ev.write('tabelle1', t1)

t2 = lt.latextable(
    [R2,T2, C2],
    "table",
    alignment = 'CCC',
    form = '.2f', )
# tex schreiben
ev.write('tabelle2', t2)

t3 = lt.latextable(
    [R2,T3, C3],
    "table",
    alignment = 'CCC',
    form = '.2f', )
# tex schreiben
ev.write('tabelle3', t3)




#Um den Alumantel korrigieren
F1=Iinit/I0[0]#Frontal
F2=Iinit/I0[1]#Hauptdiagonale
F3=Iinit/I0[2]#Nebendiagonale

#print I1
#print I2

for i in range(0,len(I1)):
	if ((i==1)|(i==2)|(i==3)|(i==4)|(i==5)):
		I1[i]=I1[i]*F1
	elif ((i==7)|(i==10)):
		I1[i]=I1[i]*F2
	else:
		I1[i]=I1[i]*F3
for i in range(0,len(I2)):
	if ((i==1)|(i==2)|(i==3)|(i==4)|(i==5)|(i==6)):
		I2[i]=I2[i]*F1
	elif ((i==8)|(i==11)):
		I2[i]=I2[i]*F2
	else:
		I2[i]=I2[i]*F3
for i in range(0,len(I3)):
	if ((i==1)|(i==2)|(i==3)|(i==4)|(i==5)|(i==6)):
		I3[i]=I3[i]*F1
	elif ((i==8)|(i==11)):
		I3[i]=I3[i]*F2
	else:
		I3[i]=I3[i]*F3


#print I1
#print I2
J1=np.insert(I1,1,0)

#erste Tabelle
t5 = lt.latextable(
    [R2,J1, I2,I3],
    "table",
    alignment = 'CCCC',
    form = '.2f', )
# tex schreiben
ev.write('korr', t5)



I1=np.log(Iinit/I1)
I2=np.log(Iinit/I2)
I3=np.log(Iinit/I3)



#Berechne die Absorptionskoeffizienten
mu1=np.dot(np.dot(np.linalg.inv(np.dot(B.transpose(),B)),B.transpose()),I1)

mu2=np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),I2)
mu3=np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),I3)

print mu1
print mu2
print mu3

#Ausgabe
N0=["C02.pdf","C08.pdf","C09.pdf"]
N1=["C11.pdf","C13.pdf","C14.pdf","C15.pdf","C16.pdf","C17.pdf","C18.pdf","C19.pdf","C110.pdf","C111.pdf","C112.pdf",]
N2=["C21.pdf","C22.pdf","C23.pdf","C24.pdf","C25.pdf","C26.pdf","C27.pdf","C28.pdf","C29.pdf","C210.pdf","C211.pdf","C212.pdf",]
N3=["C31.pdf","C32.pdf","C33.pdf","C34.pdf","C35.pdf","C36.pdf","C37.pdf","C38.pdf","C39.pdf","C310.pdf","C311.pdf","C312.pdf",]

x=[X0,X1,X2,X3]
y=[Y0,Y1,Y2,Y3]
L=[val0l,val1l,val2l,val3l]
R=[val0r,val1r,val2r,val3r]
N=[N0,N1,N2,N3]

#PLOTS
'''
fig2 = plt.figure()
for j in range(0,4):

	for i in range(0,len(N[j])):
		
		ax2 = fig2.add_subplot(111)

		ax2.plot(x[j][i], y[j][i], linestyle = 'none', marker = '+', label = 'Messwerte')
		ax2.plot(x[j][i], erfc(x[j][i],L[j][i],x[j][i][np.argmax(y[j][i])],R[j][i],(4.*1./(minir-maxil)/2.)) , label="Fit")

		ax2.set_xlabel('Kanal')
		ax2.set_ylabel('Counts')

		ax2.legend(loc = 'best')
		ax2 = ev.plot_layout(ax2)

		fig2.tight_layout()
		fig2.savefig(N[j][i])
		plt.clf()
###

'''
#PLOTS FERTIG


#Mittelwerte für Messing? und Blei
mumess_uc = ev.get_uncert(mu1)
mublei_uc = ev.get_uncert(mu2)

print mumess_uc
print mublei_uc 

#Fehler
sigma1=0.03
sigma2=0.03
sigma3=0.01
 
f1=sigma1**2 * np.linalg.inv(np.dot(B.transpose(),B))
f2=sigma2**2 * np.linalg.inv(np.dot(A.transpose(),A))
f3=sigma3**2 * np.linalg.inv(np.dot(A.transpose(),A))

'''
print "Fehler1"
for i in range(0,9):
	print f1[i,i]
print "Fehler2"
for i in range(0,9):
	print f2[i,i]
print "Fehler3"
for i in range(0,9):
	print f3[i,i]
'''






