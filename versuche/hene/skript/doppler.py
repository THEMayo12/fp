import scipy.constants as const
import numpy as np

def doppler(f0, T, m):
    return f0 / (1. - np.sqrt((8*const.k*T) / (np.pi*m)) / const.c) - f0

def deltav(L):
    return const.c / (2.*L)

f0 = 622*1e9
L = 0.5
m = 20.18 * 1.6e-27
T = 300.

print "doppler: {}, df: {}".format(doppler(f0, T, m)*1e-6, deltav(L)*1e-6)
