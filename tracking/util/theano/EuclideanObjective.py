# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:43:49 2016

@author: MindLab
"""

from tracking.model.core.Objective import Objective

class EuclideanObjective(Objective):
    
    
    def evaluate(self, target, prediction):
        batchSize = target.shape[0]
        seqLength = target.shape[1]
        loss = ((target - prediction) ** 2).sum() / batchSize / seqLength
    
        return loss
