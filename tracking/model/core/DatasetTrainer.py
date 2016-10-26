# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:17:41 2016

@author: MindLab
"""

import logging
from tracking.model.core.Trainer import Trainer

class DatasetTrainer(Trainer):

    """
    Create a Trainer for a tracker.

    @type  tracker: tracking.model.Tracker
    @param tracker: The tracker to be trained
    """
    def __init__(self, tracker, processor, validator):
        self.tracker = tracker
        self.processor = processor
        self.validator = validator
    
    
    """
    Train a tracker.

    @type  f: tracking.model.Tracker
    @param f: The tracker to be trained
    """
    def train(self, frame, position, epochs, batchSize, lnr, lnrdy):
        size = frame.shape[0]
        batches = size / batchSize + (size % batchSize > 0)
        
        frame, position = self.processor.preprocess(frame, position)
        
        for epoch in range(epochs):
            for batch in range(batches):
                start = batch * batchSize
                end = start + batchSize
                batchFrame, batchPosition = frame[start:end, ...], position[start:end, ...]               
                loss = self.tracker.fit(batchFrame, batchPosition, lnr)
                
                logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", epoch, batch, loss)
                self.validator.validateBatch(self.tracker, batchFrame, batchPosition)
                
            # Updating the learning rate
            lnr *= lnrdy
            self.validator.validateEpoch(self.tracker) 