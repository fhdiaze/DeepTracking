# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:18:09 2016

@author: MindLab
"""

import theano.tensor as THT
from tracking.model.theano.Tracker import Tracker

class TrackerCR(Tracker):         
    
    """
    Execute a forward.

    @type    frame: numpy.ndarray
    @param   frame: The frames (batchSize, channels, height, width)
    @type    prevPosition: theano.tensor.matrix
    @param   prevPosition: The positions in the previous frame (batchSize, targetDim)
    @type    prevState: theano.tensor.matrix
    @param   prevState: The states in the previous time (batchSize, stateDim)
    @rtype:  (theano.tensor.matrix, theano.tensor.matrix)
    @return: The new position and the new state
    """
    def step(self, frame, prevPosition, prevState):
        features = self.cnn.forward(frame)
        flat = THT.reshape(features, (features.shape[0], THT.prod(features.shape[1:])))
        newPosition = self.regressor.forward(prevPosition, flat)
        
        return newPosition, prevState
        
        
    def getParams(self):
        
        return self.regressor.getParams() + self.cnn.getParams()