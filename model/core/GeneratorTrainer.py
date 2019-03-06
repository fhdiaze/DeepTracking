# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:57:53 2016

@author: MindLab
"""

import logging
from tracking.model.core.Trainer import Trainer

class GeneratorTrainer(Trainer):

    """
    Create a Trainer for a tracker.

    @type  tracker: tracking.model.Tracker
    @param tracker: The tracker to be trained
    """
    def __init__(self, tracker, generator, processor, validator):
        self.tracker = tracker
        self.generator = generator
        self.processor = processor
        self.validator = validator
    
    
    """
    Train a tracker.

    @type  epochs: tracking.model.Tracker
    @param epochs: The tracker to be trained
    @type  batches: tracking.model.Tracker
    @param batches: The tracker to be trained
    @type  batchSize: tracking.model.Tracker
    @param batchSize: The tracker to be trained
    @type  lnr: tracking.model.Tracker
    @param lnr: The tracker to be trained
    @type  lnrdy: tracking.model.Tracker
    @param lnrdy: The tracker to be trained
    """
    def train(self, epochs, batches, batchSize, lnr, lnrdy):
        for epoch in range(epochs):
            for batch in range(batches):
                frame, position = self.getBatch(batchSize)
                loss = self.tracker.fit(frame, position, lnr)
                self.validator.validateBatch(self.tracker, frame, position)
                
                logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", epoch, batch, loss)
            
            # Updating the learning rate
            lnr = lnr * (1.0 / (1.0 + (lnrdy * epoch * batches + batch)))
            
            self.validator.validateEpoch(self.tracker)
            
            
    def getBatch(self, batchSize):
        frame, position = self.generator.getBatch(batchSize)
        #frame, position = self.processor.preprocess(frame, position)
        
        return frame, position