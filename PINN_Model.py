import config
from config import *

from Domain import *

from interface import Interface
from interface import *


import numpy as np
import math as math
import tensorflow as tf
import time
import os

from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks


class PINN_Model_steady(tf.keras.Model):
    
    def __init__(self, numHiddenLayers, numNeurons, domain):
        super(PINN_Model_steady, self).__init__(name="PINN_Model_steady")

        self.numHiddenLayers = numHiddenLayers
        self.numNeurons      = numNeurons
        self.domain = domain

        self.interface = Interface(coeff = 1e-5)
        
        
        self.Ga = tf.constant(98.9, dtype=config.real(tf))    # 98.0
        self.We = tf.constant(4.62e-4, dtype=config.real(tf)) # 4.62e-4
        
        self.lower_bounds = tf.constant([domain.xmin, domain.ymin, domain.zmin, domain.tmin], dtype=config.real(tf))
        self.upper_bounds = tf.constant([domain.xmax, domain.ymax, domain.zmax, domain.tmax], dtype=config.real(tf))

        
        self.hiddenLayers = []
        
        for l in range(0, self.numHiddenLayers):
            self.hiddenLayers.append(tf.keras.layers.Dense(self.numNeurons, activation=None, name='Dense%s'%(l), dtype=config.real(tf)))
        self.outputLayer = tf.keras.layers.Dense(4, activation=None, name='Output', dtype=config.real(tf))
    
        self.act_coeff = tf.keras.Variable(1.0, trainable=True, dtype=config.real(tf))
   
    def build(self):
        ins=(None, 4)
        for l in range(0, self.numHiddenLayers):
            self.hiddenLayers[l].build(input_shape=ins)
            ins = (None, self.numNeurons)
           
        self.outputLayer.build(ins)

   
    def scale(self, X):
        aa = tf.constant(2.0, dtype=config.real(tf))
        bb = tf.constant(1.0, dtype=config.real(tf))
        ll = self.lower_bounds
        rr = self.upper_bounds
        return aa * (X - ll) / (rr - ll) - bb

   
    def call(self, X):
        X = self.scale(X)
        for layer in self.hiddenLayers:
            Y = layer(X)
            X = tf.math.tanh(self.act_coeff * Y)
        Y = self.outputLayer(X)
        return Y

   
    def net_field(self, x, y, z, t):
        X = tf.concat([x,y,z,t], axis=1) 
        Y = self.call(X)
        u = Y[:,0:1]
        v = Y[:,1:2]
        w = Y[:,2:3]
        p = Y[:,3:4]
        
        return u, v, w, p


   
    def Equations(self, x,y,z,t):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
            tape1.watch(z)
            tape1.watch(t)
            
        
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                tape2.watch(z)
                tape2.watch(t)
                
                u, v, w, p = self.net_field(x, y, z, t)
        
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            u_z = tape2.gradient(u, z)
        
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            v_z = tape2.gradient(v, z)
        
            w_x = tape2.gradient(w, x)
            w_y = tape2.gradient(w, y)
            w_z = tape2.gradient(w, z)

            u_t = tape2.gradient(u, t)
            v_t = tape2.gradient(v, t)
            w_t = tape2.gradient(w, t)
        
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)
            p_z = tape2.gradient(p, z)

            t_xx = u_x + u_x
            t_xy = u_y + v_x
            t_xz = u_z + w_x
            t_yy = v_y + v_y
            t_yz = v_z + w_y
            t_zz = w_z + w_z

        tau_x = tape1.gradient(t_xx, x) + tape1.gradient(t_xy, y) + tape1.gradient(t_xz, z)
        tau_y = tape1.gradient(t_xy, x) + tape1.gradient(t_yy, y) + tape1.gradient(t_yz, z)
        tau_z = tape1.gradient(t_xz, x) + tape1.gradient(t_yz, y) + tape1.gradient(t_zz, z)
            
        
        Cnt = u_x + v_y + w_z
        Nsx = u_t + (u*u_x + v*u_y + w*u_z) + p_x - tau_x
        Nsy = v_t + (u*v_x + v*v_y + w*v_z) + p_y - tau_y
        Nsz = w_t + (u*w_x + v*w_y + w*w_z) + p_z - tau_z + self.Ga

        return Cnt, Nsx, Nsy, Nsz


    
    def call_loss_phys(self, resPoints, sw_sobolev : bool):
        
        xx_f   = resPoints[:,0:1]
        yy_f   = resPoints[:,1:2]
        zz_f   = resPoints[:,2:3]
        tt_f   = resPoints[:,3:4]
        

        loss_phys = 0.0
        loss_sobo = 0.0

        if sw_sobolev:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(xx_f)
                tape.watch(yy_f)
                tape.watch(zz_f)
                tape.watch(tt_f)
                Eqns = self.Equations(xx_f, yy_f, zz_f, tt_f)
            for eqn in Eqns:
                loss_phys += tf.reduce_mean(tf.square(eqn))
                gf_x = tape.gradient(eqn, xx_f)
                gf_y = tape.gradient(eqn, yy_f)
                gf_z = tape.gradient(eqn, zz_f)
                gf_t = tape.gradient(eqn, tt_f)

                loss_sobo += tf.reduce_mean(tf.square(gf_x))
                loss_sobo += tf.reduce_mean(tf.square(gf_y))
                loss_sobo += tf.reduce_mean(tf.square(gf_z))
                loss_sobo += tf.reduce_mean(tf.square(gf_t))
                

            del tape
        
        else:
            Eqns = self.Equations(xx_f, yy_f, zz_f, tt_f)

            for eqn in Eqns:
                loss_phys += tf.reduce_mean(tf.square(eqn))

        return loss_phys, loss_sobo
        # return loss_phys    

    
    def call_loss_BC(self, BC_Points):
        x   = BC_Points[:,0:1]
        y   = BC_Points[:,1:2]
        z   = BC_Points[:,2:3]
        t   = BC_Points[:,3:4]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            tape.watch(t)
            u, v, w, p = self.net_field(x, y, z, t)
        u_x = tape.gradient(u,x)
        u_y = tape.gradient(u,y)
        u_z = tape.gradient(u,z)
        u_t = tape.gradient(u,t)

        v_x = tape.gradient(v,x)
        v_y = tape.gradient(v,y)
        v_z = tape.gradient(v,z)
        v_t = tape.gradient(v,t)
        
        w_x = tape.gradient(w,x)
        w_y = tape.gradient(w,y)
        w_z = tape.gradient(w,z)
        w_t = tape.gradient(w,t)

        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        p_z = tape.gradient(p, z)

        del tape

        VSMALL = 1e-13
        rr = tf.math.sqrt(x*x + y*y + z*z)
        theta = tf.math.acos(z / (rr + VSMALL))
        r2 = tf.math.sqrt(x*x + y*y)
        phi   = tf.math.sign(y) * tf.math.acos(x / (r2 + VSMALL))  # phi [-pi:pi]

        p_jet = self.interface.P_jet(theta) * self.Ga
        tau_jet = self.interface.Tau_jet(theta)

        nx, ny, nz = self.interface.normal(x, y, z, t)
        kappa = self.interface.curvature(x, y, z, t)

        t_xx = u_x + u_x
        t_xy = u_y + v_x
        t_xz = u_z + w_x
        t_yy = v_y + v_y
        t_yz = v_z + w_y
        t_zz = w_z + w_z

        Snx = t_xx*nx + t_xy*ny + t_xz*nz
        Sny = t_xy*nx + t_yy*ny + t_yz*nz
        Snz = t_xz*nx + t_yz*ny + t_zz*nz

        Snx -= p*nx
        Sny -= p*ny
        Snz -= p*nz

        Tx = (1.0-nx*nx)*Snx      -nx*ny *Sny      -nx*nz *Snz
        Ty =     -ny*nx *Snx +(1.0-ny*ny)*Sny      -ny*nz *Snz
        Tz =     -nz*nx *Snx      -nz*ny *Sny +(1.0-nz*nz)*Snz

        # unit vector negative theta
        ex = -nx*nz
        ey = -ny*nz
        ez = 1.0-nz*nz

        Rx = tau_jet*ex - (p_jet + kappa / self.We) * nx
        Ry = tau_jet*ey - (p_jet + kappa / self.We) * ny
        Rz = tau_jet*ez - (p_jet + kappa / self.We) * nz

        F = self.interface.F(x,y,z,t)

        

        BC4 = u*nx + v*ny + w*nz
        
        BC1 = Snx - Rx
        BC2 = Sny - Ry
        BC3 = Snz - Rz
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x,y,z,t])
            F = self.interface.F(x,y,z,t)
            u,v,w,p = self.net_field(x,y,z,t)

        F_x = tape.gradient(F, x)
        F_y = tape.gradient(F, y)
        F_z = tape.gradient(F, z)
        F_t = tape.gradient(F, t)
        
        del tape

        BC_kinematic = F_t + u*F_x + v*F_y + w*F_z
        
        # loss_BC1 = tf.reduce_mean(tf.square(BC1))
        loss_BC1 = tf.reduce_mean(tf.square(BC1)) 
        loss_BC2 = tf.reduce_mean(tf.square(BC2)) 
        loss_BC3 = tf.reduce_mean(tf.square(BC3)) 
        loss_BC4 = tf.reduce_mean(tf.square(BC4)) 
        loss_BC_kin = tf.reduce_mean((tf.square(BC_kinematic)))


        loss_BC = loss_BC1 + loss_BC2 + loss_BC3 + loss_BC4 + loss_BC_kin

        return loss_BC, [loss_BC1, loss_BC2, loss_BC3, loss_BC4, loss_BC_kin]

    
    # def cal_loss_pRef(self):
        
        x0 = np.full((1,1), 0.0, dtype=config.real(np))
        y0 = np.full((1,1), 0.0, dtype=config.real(np))
        z0 = np.full((1,1), 0.0, dtype=config.real(np))
        
        x0 = tf.convert_to_tensor(x0, dtype=config.real(tf))
        y0 = tf.convert_to_tensor(y0, dtype=config.real(tf))
        z0 = tf.convert_to_tensor(z0, dtype=config.real(tf))

        sol = self.net_field(x0, y0, z0)
        pp = sol[3]

        loss_Pref = tf.reduce_mean(tf.square(pp))
        loss_Pref =0.0

        return loss_Pref
    
    def loss_fn(self, dataList):
        dataGE = dataList[0]
        dataBC = dataList[1]        
        
        # loss_phys = self.call_loss_phys(dataGE, True)
        loss_phys, loss_sobo = self.call_loss_phys(dataGE, True)
        BC, BC_sub = self.call_loss_BC(dataBC)
        
        
        loss_value = loss_phys + BC + loss_sobo 
        # loss_value =  BC 
        
        return loss_value
    
    
    def sub_loss_labels(self):        
        return ["GE", "BC", "BC1", "BC2", "BC3", "BC4", "act_coeff"]
        # return ["GE", "BC", "BC1", "BC2", "BC3", "BC4", "Sob", "act_coeff"]
    


    def loss_eval(self, dataList):
        dataGE = dataList[0]
        dataBC = dataList[1]        
        
        loss_phys, loss_sobo = self.call_loss_phys(dataGE, True)        
        # loss_phys = self.call_loss_phys(dataGE, True)        
        BC, BC_sub = self.call_loss_BC(dataBC)

        
        loss_value = loss_phys + BC + loss_sobo 
        # loss_value = loss_phys + BC 

        # return loss_value, [loss_phys, BC, BC_sub[0], BC_sub[1], BC_sub[2], BC_sub[3], loss_sobo, self.act_coeff]
        return loss_value, [loss_phys, BC, BC_sub[0], BC_sub[1], BC_sub[2], BC_sub[3], self.act_coeff]
