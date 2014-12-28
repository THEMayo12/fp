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


sp.params["text.latex.preamble"] = sp.tex_mathpazo_preamble
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
#	start evaluation
# ==================================================

def g(L, r): return 1.0 - L/r

def gg(L, r1, r2): return g(L, r1) * g(L, r2)

# ===== Los gehts ==================================

r1 = 100 # cm
r2 = 140 # cm

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
#
# L = np.linspace(0, 200, 1000)
# ax1.plot(L, gg(L, r1, r1), label="Fit")
#
# ax1.set_xlabel(r'$L / \unit{cm}$')
# ax1.set_ylabel(r'$g_1g_2$')
#
# ax1.legend(loc = 'best')
# ax1 = ev.plot_layout(ax1)
#
# fig1.tight_layout()

min_r1r1 = optimize.fmin(gg, 110.0, args=(r1, r1))
min_r1r2 = optimize.fmin(gg, 110.0, args=(r1, r2))

p = sp.figure([r'$L / \unit{cm}$', r'$g_1g_2$'])
p.set_xlim([0, 200.])
p.add_plot(gg, r1, r2, label="r1 = \unit[100]{cm}, r2 = \unit[140]{cm}")
p.add_plot(gg, r1, r1, label="r1 = \unit[100]{cm}, r2 = \unit[100]{cm}")
fig = p.plot()
# plt.show()

# ax = fig.add_subplot(111)
# ax.

fig.savefig("gg.pdf")
