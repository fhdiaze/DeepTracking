# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:36:32 2016

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class AttProcessor(Processor):
    
    def __init__(self, processor):
        self.processor = processor
    
    
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        cropPosition = NP.roll(position, 1, axis=1) # Shift the time
        
        if position.shape[1] > 1:
            cropPosition[:, 0, :] = cropPosition[:, 1, :] # First frame is ground truth
        
        return [frame, cropPosition], position
        
        
    def postprocess(self, frame, position):
        frame, position = self.processor.postprocess(frame[0], position)
        
        return frame, position