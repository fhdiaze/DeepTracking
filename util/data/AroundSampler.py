# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:06:12 2016

@author: MindLab
"""

import numpy as NP
import numpy.linalg as NLA
from tracking.util.data.Sampler import Sampler

class AroundSampler(Sampler):
    
    def generateTheta(self, position, samples, transRange):      
        theta = NP.zeros((samples, 3, 3), dtype='float32')
        tx = NP.random.uniform(-transRange, transRange, size=(samples))
        ty = NP.random.uniform(-transRange, transRange, size=(samples))
        sx = 1.0 - abs(tx) # Scale x
        sy = 1.0 - abs(ty) # Scale y
        
        # Setting transformation
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        theta[:, 2, 2] = 1.0
        
        return theta, NLA.inv(theta)