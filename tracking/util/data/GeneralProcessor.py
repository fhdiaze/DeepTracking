# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:12:33 2016

@author: MindLAB
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess
from tracking.model.core.Processor import Processor

class GeneralProcessor(Processor):
    
    def __init__(self, positionModel):
        self.positionModel = positionModel
    
    
    def preprocess(self, frame, position):
        frame = NP.copy(frame).transpose(0, 1, 4, 2, 3)
        #frame = frame[:,:,::-1,:,:] # Make BGR
        frame = Preprocess.scaleFrame(frame)
        frameDims = frame.shape[-2:]
        
        position = Preprocess.scalePosition(position, frameDims)
        position = self.positionModel.fromTwoCorners(position)

        return frame, position

    
    def postprocess(self, frame, position):
        frame = Preprocess.rescaleFrame(NP.copy(frame))
        #frame = frame[:,:,::-1,:,:] # Make RGB
        frameDims = frame.shape[-2:]
        frame = frame.transpose(0, 1, 3, 4, 2)
        
        position = self.positionModel.toTwoCorners(position)
        position = Preprocess.rescalePosition(position, frameDims)

        return frame, position