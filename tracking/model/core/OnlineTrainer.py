# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:18:55 2016

@author: MindLab
"""

import logging
import numpy as NP


class OnlineTrainer(object):
    
    def __init__(self, tracker, processor, validator):
        self.tracker = tracker
        self.processor = processor
        self.validator = validator
        
    # frame.shape = (channels, height, width)
    # initPosition.shape = (targetDim)
    def fitFrame(self, sampler, batches, batchSize, lnr, frame, position):
        for batch in range(batches):
            sampledFrame, sampledPosition = sampler.generateSamples(frame, position, batchSize)
            sampledFrame = NP.expand_dims(sampledFrame, axis=1)
            sampledPosition = NP.expand_dims(sampledPosition, axis=1)
            loss = self.tracker.fit(sampledFrame, sampledPosition, lnr)
            self.validator.validateBatch(self.tracker, sampledFrame, sampledPosition)
            
            logging.info("Batch Loss: batch = %d, loss = %f", batch, loss)


    def forwardOnline(self, sampler, frame, initPosition, batches, batchSize, lnr):
        batchSize = frame.shape[0]
        seqLength = frame.shape[1]
        targetDim = initPosition.shape[1]
        predPosition = NP.zeros((batchSize, seqLength, targetDim))
        predPosition[0, 0, :] = initPosition
        
        for i in range(1, seqLength):
            self.fitFrame(sampler, batches, batchSize, lnr, frame[0, i-1, ...], predPosition[0, i-1:i, ...])
            predPosition[0, i, :] = self.tracker.forward(frame[:1, i:i+1, ...], predPosition[:, i-1:i, ...])

        return predPosition
        
    
    # frame [samples, channels, height, width]
    # position [samples, targetDim]
    def train(self, sampler, batches, batchSize, lnr, frame, position):
        for i in range(frame.shape[0]):
            self.fitFrame(sampler, batches, batchSize, lnr, frame[i, ...], position[i, ...])
        