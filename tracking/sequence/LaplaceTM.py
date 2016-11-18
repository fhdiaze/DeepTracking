# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:29:48 2016

@author: MindLAB
"""

import numpy as NP
from tracking.util.model.CentroidHWPM import CentroidHWPM

class LaplaceTM:
    
    def __init__(self, muT, lambdaT, muR, lambdaR):
        self.samplerT = lambda size : NP.random.laplace(loc=muT, scale=lambdaT, size=size)
        self.samplerR = lambda size : NP.random.laplace(loc=muR, scale=lambdaR, size=size)
        self.centroidHWPM = CentroidHWPM()
       
    
    def generateTrajectory(self, length):
        theta = NP.zeros((length, 3, 3), dtype='float32')
        
        # Initialize the trajectory
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0
        theta[0, 2, 2] = 1.0
        
        for t in range(1, length):
            deltaT = self.samplerT((1, 2))
            deltaR = self.samplerR((1, 2))
            
            thetaT = NP.zeros((3, 3))
            thetaT[[0, 1], [2, 2]] = deltaT
            thetaT[[0, 1], [0, 1]] = 1.0 - abs(deltaT)
            thetaT[[0, 1], [1, 0]] = deltaR
            thetaT[2, 2] = 1.0
            
            theta[t, ...] = NP.dot(theta[t-1, ...], thetaT)
            
        return theta