# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:53:22 2016

@author: MindLab
"""

from tracking.model.core.Objective import Objective

class ReconstructionObjective(Objective):
    
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
    def evaluate(self, gtPosition, predPosition, frame):
        batchSize = gtPosition.shape[0]
        frameSize = frame.shape[2]
        
        # Rescale the positions
        gtPosition = THT.round(Process.rescalePosition(gtPosition))
        predPosition = THT.round(Process.rescalePosition(predPosition))
        
        loss = ((gtPosition - predPosition) ** 2).sum() / batchSize
    
        return loss