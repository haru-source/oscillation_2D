import config
from config import *

from Domain import *

import numpy as np
import math as math
import tensorflow as tf


class Interface():
    def __init__(self, coeff):

        self.coeff = coeff
        self.coeff_Pj = 0.1
        self.coeff_Tj = 0.1

    ##########################
    def S(self, theta):
        val = 3.0 * tf.cos(theta) ** 2 - 1.0

        return self.coeff * val
    
    def F(self, x, y):
        VSMALL = 1e-13
        rr = tf.sqrt(x*x + y*y )
        theta = tf.math.acos(y / (rr + VSMALL))
        R_0 = 1.0
        F = rr - R_0

        return F
    #########################
    
    def P_jet(self, theta):
        p_jet = self.coeff_Pj * tf.math.cos(np.pi - theta)

        return p_jet
    
    ##################################
    def Tau_jet(self, theta):
        tau_jet = -self.coeff_Tj * tf.math.sin(theta)

        return tau_jet

    
    ##########################
    def normal(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            F = self.F(x, y, t)
        Fx = tape.gradient(F, x)
        Fy = tape.gradient(F, y)

        denom = tf.math.sqrt(Fx**2 + Fy**2)
        Fx /= denom
        Fy /= denom

        return Fx, Fy
    
    ##########################
    def curvature(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            nx, ny = self.normal(x, y, t)
        nxx = tape.gradient(nx, x)
        nyy = tape.gradient(ny, y)

        kappa = nxx + nyy

        return kappa
    
    

    
    

