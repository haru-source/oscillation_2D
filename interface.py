import config
from config import *

from Domain import *

import numpy as np
import math as math
import tensorflow as tf


class Interface():
    def __init__(self, coeff):

        self.coeff_deform = coeff
        self.coeff_Pj = 1.0
        self.coeff_Tj = 1.0

    ##########################
    def S(self, theta):
        val = 3.0 * tf.cos(theta) ** 2 - 1.0

        return self.coeff_deform * val
    

    ##########################
    def F(self, x, y, z, t):
        eps = 1e-12
        r = tf.sqrt(x**2 + y**2 + z**2 + eps)
        r = tf.sqrt(r)
        # cos_theta = z / r
        # P2 = (3.0 * (cos_theta)**2 - 1.0)
        P2 = 3.0 * (z**2 / r) - 1.0
        a2 = 0.4
        Rmax_theta = 1.0 + a2 * P2 * tf.cos(t)
        F = r - Rmax_theta
        return F
    
    
#        theta = tf.math.acos(z / (rr + VSMALL))
#        r2 = tf.sqrt(x*x + y*y)
#        phi   = tf.math.sign(y) * tf.math.acos(x / (r2 + VSMALL))  # phi [-pi:pi]
#        phi = tf.atan2(y, x)
#        rs = 1.0 + self.S(theta)
    

    ##################################
    def P_jet(self, theta):
        p_jet = self.coeff_Pj * tf.math.cos(np.pi - theta)

        return p_jet
    
    ##################################
    def Tau_jet(self, theta):
        tau_jet = self.coeff_Tj * tf.math.sin(theta)

        return tau_jet

    ########################## outward normal
    def normal(self, x, y, z, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            F = self.F(x, y, z, t)
        Fx = tape.gradient(F, x)
        Fy = tape.gradient(F, y)
        Fz = tape.gradient(F, z)
        
        denom = tf.math.sqrt(Fx**2 + Fy**2 + Fz**2)
        nx = Fx / denom
        ny = Fy / denom
        nz = Fz / denom

        del tape

        return nx, ny, nz

    
    ##########################
    def curvature(self, x, y, z, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            nx, ny, nz = self.normal(x, y, z, t)
        nxx = tape.gradient(nx, x)
        nyy = tape.gradient(ny, y)
        nzz = tape.gradient(nz, z)

        del tape

#        print(nxx.numpy(), nyy.numpy(), nzz.numpy())

        kappa = (nxx + nyy + nzz)

        return kappa
    
    

    
    

