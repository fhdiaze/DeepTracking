# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:51 2016

@author: MindLAB
"""

import numpy as NP
from tracking.data.Processor import Processor

class CropperProcessor(Processor):
    
    def __init__(self, cropper, positionModel, tRange):
        self.cropper = cropper
        self.theta = None
        self.positionModel = positionModel
        self.tRange = tRange
        self.cRange = NP.array([[-1.0, 1.0], [-1.0, 1.0]])
        
        
    def preprocess(self, frame, position):        
        bF = []
        bP = []
        bT = []
        self.oRange = []
        
        for f, p in zip(frame, position):        
            frameSize = f[0].size[0]
            self.oRange.append(NP.array([[0, frameSize], [0, frameSize]]))
            cropPosition = NP.copy(p)
            
            if cropPosition.shape[1] > 1:
                cropPosition[-1, :] = cropPosition[-2, :] # We want to predict last frame position
            
            f, p, t = self.cropper.crop(f, cropPosition, p)
            bF.append(f)
            bP.append(p)
            bT.append(t)
            
        self.theta = NP.stack(bT)
        
        return NP.stack(bF), NP.stack(bP)

    
    def postprocess(self, frame, position):
        p = NP.empty(position.shape)
        
        for i in range(position.shape[0]):
            # Transforming the positions        
            tempP = self.positionModel.scale(position[i, ...], self.tRange, self.cRange)
            tempP = self.positionModel.transform(self.theta[i, ...], tempP)
            p[i, ...] = self.positionModel.scale(tempP, self.cRange, self.oRange[i])

        return frame, p