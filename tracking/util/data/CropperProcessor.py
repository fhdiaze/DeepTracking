# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:51 2016

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class CropperProcessor(Processor):
    
    def __init__(self, cropper, positionModel, processor, tRange):
        self.cropper = cropper
        self.theta = None
        self.positionModel = positionModel
        self.processor = processor
        self.tRange = tRange
        self.cRange = NP.array([[-1.0, 1.0], [-1.0, 1.0]])
        
        
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        position = self.positionModel.scale(position, self.tRange, self.cRange)
        cropPosition = NP.roll(position, 1, axis=1) # Shift the time
        
        if cropPosition.shape[1] > 1:
            cropPosition[:, 0, :] = cropPosition[:, 1, :] # First frame is ground truth
        
        frame, position, self.theta = self.cropper.crop(frame, cropPosition, position)
        position = self.positionModel.scale(position, self.cRange, self.tRange)

        return frame, position

    
    def postprocess(self, frame, position):
        # Transforming the positions        
        position = self.positionModel.scale(position, self.tRange, self.cRange)
        position = self.positionModel.transform(self.theta, position)
        position = self.positionModel.scale(position, self.cRange, self.tRange)
        frame, position = self.processor.postprocess(frame, position)

        return frame, position