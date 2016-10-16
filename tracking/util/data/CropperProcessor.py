# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:51 2016

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class CropperProcessor(Processor):
    
    def __init__(self, cropper, positionModel, processor):
        self.cropper = cropper
        self.theta = None
        self.positionModel = positionModel
        self.processor = processor
        
        
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        cropPosition = NP.roll(position, 1, axis=1) # Shift the time
        
        if position.shape[1] > 1:
            cropPosition[:, 0, :] = cropPosition[:, 1, :] # First frame is ground truth
        
        frame, position, self.theta = self.cropper.crop(frame, cropPosition, position)

        return frame, position

    
    def postprocess(self, frame, position):
        
        # Transforming the positions
        position = self.positionModel.transform(self.theta, position)
        frame, position = self.processor.postprocess(frame, position)

        return frame, position