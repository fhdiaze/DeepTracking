# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:40:26 2017

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class SingleCropperProcessor(Processor):
    
    def __init__(self, cropper, positionModel, processor, tRange, timeLength):
        self.cropper = cropper
        self.positionModel = positionModel
        self.processor = processor
        self.tRange = tRange
        self.timeLength = timeLength
        self.cRange = NP.array([[-1.0, 1.0], [-1.0, 1.0]])
        self.frame = None
        self.position = None
        self.theta = None
        self.lastLength = 0
        
        
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        position = self.positionModel.scale(position, self.tRange, self.cRange)
        
        self.updateCache(frame, position)
        cropPosition = NP.copy(self.position)
        
        if cropPosition.shape[1] > 1:
            cropPosition[:, -1, :] = cropPosition[:, -2, :] # All objects are centered inside search region except the last one
        
        frame, position, self.theta = self.cropper.crop(self.frame, cropPosition, self.position)
        position = self.positionModel.scale(position, self.cRange, self.tRange)

        return frame, position[:, -1, :]

    
    def postprocess(self, frame, position):
        # Transforming the positions
        position = NP.concatenate((self.position, NP.expand_dims(position, axis=1)), axis=1)
        position = self.positionModel.scale(position[:, 1:, :], self.tRange, self.cRange)
        position = self.positionModel.transform(self.theta[:, -self.lastLength:, ...], position)
        position = self.positionModel.scale(position, self.cRange, self.tRange)
        frame, position = self.processor.postprocess(frame[:, -self.lastLength:, ...], position)

        return frame, position
        
        
    def updateCache(self, frame, position):
        
        if(self.frame is not None and self.position is not None):
            self.frame = NP.concatenate((self.frame, frame), axis=1)[:, -self.timeLength:, ...]
            self.position = NP.concatenate((self.position, position), axis=1)[:, -self.timeLength:, :]
        else:
            self.frame = frame
            self.position = position
            
            
    def clean(self):
        self.frame = None
        self.position = None
        
    
    def setStateful(self, stateful):
        if stateful:
            