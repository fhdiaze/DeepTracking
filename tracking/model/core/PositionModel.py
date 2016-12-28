# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:35:26 2016

@author: MindLab
"""

import numpy as NP

class PositionModel(object):
    
    def __init__(self, targetDim):
        self.targetDim = targetDim
        
    
    def getTargetDim(self):
        
        return self.targetDim
    
    
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def fromTwoCorners(self, position):
        pass
    
    
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def toTwoCorners(self, position):
        pass
    
    
    """
    Plot (inside) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: object
    @param position: The objects's position representation
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        pass
    

    # theta.shape = (batchSize, 3, 3)
    def transform(self, theta, position):
        positionShape = position.shape
        theta = theta.reshape((-1, 3, 3))
        samples = theta.shape[0]
        position = position.reshape((samples, -1, 2)).transpose(0, 2, 1)
    
        # Reshaping the positions
        position = NP.concatenate((position, NP.ones((samples, 1, position.shape[2]))), axis=1)
        
        # Applying the transformation
        position = NP.matmul(theta, position)[:, :2, :]
    
        # Reshaping the result
        position = position.transpose(0, 2, 1)
        position = position.reshape(positionShape)
    
        return position
    
    
    # oRange = [[xMin, xMax], [yMin, yMax]]
    # tRange = [[xMin, xMax], [yMin, yMax]]
    def scale(self, position, oRange, tRange):
        shape = position.shape
        oDiff = NP.abs(oRange[:, :1] - oRange[:, 1:]) # [[xDiff], [yDiff]]
        tDiff = NP.abs(tRange[:, :1] - tRange[:, 1:]) # [[xDiff], [yDiff]]
        position = position.reshape((-1, 2)).T
        position = tDiff * (position - oRange[:, :1]) / oDiff + tRange[:, :1]
        position = position.T.reshape(shape)
        
        return position