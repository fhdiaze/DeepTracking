# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:55:29 2016

@author: MindLAB
"""

import logging
import numpy as NP

class Validator(object):

    def __init__(self, batchSize, measure):
        self.batchSize = batchSize
        self.measure = measure
        self.valSets = {}
        
    
    def validateEpoch(self, tracker):
                
        for name, data in self.valSets.iteritems():
            valF, valP = data
            valSetSize, seqLength, targetDim = valP.shape
            position = NP.empty((0, seqLength-1, targetDim))
        
            for i in xrange(0, valSetSize, self.batchSize):
                start = i
                end = i + self.batchSize
                batchF = valF[start:end, ...]
                batchP = valP[start:end, :1, :]
                tracker.reset()
                batchPredPosition = tracker.track(batchF, batchP)
                position = NP.append(position, batchPredPosition, axis=0)
            
            measureValue = self.measure.calculate(valP[:, 1:, :], position).mean()
            logging.info("Validation Epoch - %s: %s = %f", name, self.measure.name, measureValue)
        
    
    def validateBatch(self, tracker, frame, position):
        tracker.reset()
        predPosition = tracker.forward(frame, position)
        measureValue = self.measure.calculate(position, predPosition).mean()
        logging.info("Validation Batch: %s = %f", self.measure.name, measureValue)
        

    def setValidationSet(self, name, frame, position):
        self.valSets[name] = (frame, position)
        
        
    def test(self, tracker, seqs):
        result = {}

        for name, frame, position in seqs:
            tracker.reset()
            predP = tracker.track(frame, position[:, :1, :])
            measure = self.measure.calculate(position[:, 1:, :], predP)
    
            result[name] = measure
    
        return result
