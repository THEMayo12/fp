#!/usr/bin/env python2
# -*- coding: iso-8859-1 -*-


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

d = 0.9
D = 2.95
e = 2.25
print e, D, d

