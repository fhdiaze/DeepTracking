# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:50:16 2016

@author: MindLab
"""

import theano.tensor as Tensor
from tracking.model.core.LossFunction import LossFunction

class SmoothManhattanObjective(LossFunction):

    def __init__(self):
        pass
    
    
    def evaluate(self, target, prediction):
        diff = (target - prediction)
        loss = Tensor.switch(Tensor.lt(Tensor.abs_(diff), 1), 0.5 * diff**2, Tensor.abs_(diff) - 0.5)
        
        return loss