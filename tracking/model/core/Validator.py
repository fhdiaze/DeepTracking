# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:55:29 2016

@author: MindLAB
"""

import logging
import numpy as NP

class Validator(object):

    def __init__(self, frame, position, batchSize, measure):        
        self.frame = frame
        self.position = position
        self.batchSize = batchSize
        self.measure = measure
        
    
    def validateEpoch(self, tracker):
        valSetSize, seqLength, targetDim = self.position.shape
        position = NP.empty((0, seqLength-1, targetDim))
        
        for i in xrange(0, valSetSize, self.batchSize):
            start = i
            end = i + self.batchSize
            batchF = self.frame[start:end, ...]
            batchP = self.position[start:end, :1, :]
            tracker.reset()
            batchPredPosition = tracker.track(batchF, batchP)
            position = NP.append(position, batchPredPosition, axis=0)
        
        measureValue = self.measure.calculate(self.position[:, 1:, :], position).mean()
        logging.info("Validation Epoch: %s = %f", self.measure.name, measureValue)
        
    
    def validateBatch(self, tracker, frame, position):
        tracker.reset()
        predPosition = tracker.forward(frame, position[:, :1, :])
        measureValue = self.measure.calculate(position, predPosition).mean()
        logging.info("Validation Batch: %s = %f", self.measure.name, measureValue)
        

    def setValidationSet(self, frame, position):
        self.frame = frame
        self.position = position
        
        
    def test(self, tracker, seqs):
        result = {}

        for name, frame, position in seqs:
            tracker.reset()
            predP = tracker.track(frame, position[:, :1, :])
            measure = self.measure.calculate(position[:, 1:, :], predP)
    
            result[name] = measure
    
        return result
