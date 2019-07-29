# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:32:11 2016

@author: MindLab
"""

import numpy as NP
import numpy.linalg as NLA
from tracking.util.data.Sampler import Sampler

class TranslationSampler(Sampler):

    def generateTheta(self, position, samples, transRange):      
        thetaC = NP.zeros((samples, 3, 3), dtype='float32')
        objCenter = NP.sum(position.reshape((-1, 2)).T, axis=1, keepdims=True) / 2.0
        tx = objCenter[0,0]
        ty = objCenter[1,0]
        sx = 1.0 # Scale x
        sy = 1.0 # Scale y
        
        # Setting transformation
        thetaC[:, 0, 0] = sx
        thetaC[:, 1, 1] = sy
        thetaC[:, 0, 2] = tx
        thetaC[:, 1, 2] = ty
        thetaC[:, 2, 2] = 1.0
        
        thetaT = NP.zeros((samples, 3, 3), dtype='float32')
        tx = NP.random.uniform(-transRange, transRange, size=(samples))
        ty = NP.random.uniform(-transRange, transRange, size=(samples))
        sx = 1.0 # Scale x
        sy = 1.0 # Scale y
        
        # Setting transformation
        thetaT[:, 0, 0] = sx
        thetaT[:, 1, 1] = sy
        thetaT[:, 0, 2] = tx
        thetaT[:, 1, 2] = ty
        thetaT[:, 2, 2] = 1.0
        
        theta = NP.matmul(thetaC, thetaT)
        
        return theta, NLA.inv(theta)