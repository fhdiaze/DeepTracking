# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:21:03 2016

@author: MindLab
"""

import numpy as NP

class RandomTrajectory:
    
    def __init__(self, stepScale, velScale, accScale):
        self.stepScale = stepScale
        self.velScale = velScale
        self.accScale = accScale
        
        
    def getTrajectory(self, batchSize, length):
        position = NP.empty((batchSize, length, 2))
        
        # Initial position uniform random inside the box.
        p = NP.random.uniform(low=-1.0, high=1.0, size=(batchSize, 2))

        # Choose a random velocity.
        theta = NP.random.rand(batchSize) * 2 * NP.pi
        initialVelocity = NP.random.normal(0, self.velScale, size=(batchSize, 2))
        velocity = initialVelocity * NP.array([NP.sin(theta), NP.cos(theta)]).T
        position[:, 0, :] =  p
        
        for t in range(1, length):
            p += self.stepScale * velocity
            
            clip = -1 * (NP.abs(p) >= 1.0) + (NP.abs(p) < 1.0)
            p = NP.clip(p, -1.0, 1.0)
            velocity *= clip
            
            # Set the new position.
            position[:, t, :] = p
            
            # Update the velocity
            velocity += NP.zeros(shape=(batchSize, 2)) if self.accScale == 0 else NP.random.normal(0, self.accScale, size=(batchSize, 2))
        
        return position