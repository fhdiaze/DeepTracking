# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:49:17 2016

@author: MindLab
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess

class PositionSampler(object):
    
    def __init__(self, transRange):
        self.transRange = transRange
        
        
    def generateSamples(self, frame, position, samples):
        # Generating the sampled positions
        frameDims = frame.shape
        position = Preprocess.scalePosition(position, frameDims[:2])
        targetDim = position.shape[0]
        trans = NP.random.uniform(-self.transRange, self.transRange, size=(samples, targetDim))
        sampledPosition = position + trans
        
        sampledPosition = Preprocess.rescalePosition(sampledPosition, frameDims[:2])
        
        return NP.tile(frame, (samples, 1, 1, 1)), sampledPosition