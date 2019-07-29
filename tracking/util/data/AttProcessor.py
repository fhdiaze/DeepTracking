# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:36:32 2016

@author: MindLAB
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess
from tracking.model.core.Processor import Processor

class AttProcessor(Processor):
    
    def __init__(self, processor):
        self.processor = processor
    
    
    def preprocess(self, frame, position):
        attPosition = NP.roll(position, 1, axis=1) # Shift the time
        frame, position = self.processor.preprocess(frame, position)
        frameDims = frame.shape[-2:]
        
        if position.shape[1] > 1:
            attPosition[:, 0, :] = attPosition[:, 1, :] # First frame is ground truth
            
        attPosition = Preprocess.scalePosition(attPosition, frameDims)
        
        return [frame, attPosition], position


    def postprocess(self, frame, position):
        frame, position = self.processor.postprocess(frame[0], position)
        
        return frame, position