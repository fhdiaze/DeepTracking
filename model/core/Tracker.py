# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:12:34 2016

@author: MindLAB
"""

import logging
import dill

class Tracker(object):
    
    """
    Make a fit process using a set of frames and positions.

    @type    frame: numpy.ndarray
    @param   frame: The frames (batchSize, seqLength, channels, height, width)
    @type    position: numpy.ndarray
    @param   position: The positions over the frames (batchSize, seqLength, targetDim)
    @type    lnr: float
    @param   lnr: The learning rate
    @rtype:  (float, numpy.ndarray)
    @return: The cost in training
    """
    def fit(self, frame, position):
        pass
    
    
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
    def train(self, generator, epochs, batches, lnr, lnrdy, validator):
        initLnr = lnr
        
        for epoch in range(epochs):
            for batch in range(batches):
                frame, position = next(generator)
                loss = self.tracker.fit(frame, position, lnr)
                validator.validateBatch(self.tracker, frame, position)
                
                logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", epoch, batch, loss)
            
            # Updating the learning rate
            lnr = initLnr * (1.0 / (1.0 + lnrdy * epoch))
            
            validator.validateEpoch(self.tracker)
    
    
    """
    Make a forward process using a set of frames and positions.

    @type    frame: numpy.ndarray
    @param   frame: The frames (batchSize, seqLength, channels, height, width)
    @type    initPosition: numpy.ndarray
    @param   initPosition: The positions in the first frames (batchSize, targetDim)
    @rtype:  (float, numpy.ndarray)
    @return: The cost in training
    """
    def forward(self, frame, initPosition):
        pass
    
    
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        pass
    
    
    def reset(self):
        pass
    
    
    def save(self, path):
        logging.info("Saving tracker to %s", path)
        modelFile = open(path, 'wb')
        dill.dump(self.__dict__, modelFile)
        modelFile.close()
            
    
    def load(self, path):
        logging.info("Loading tracker from %s", path)
        modelFile = open(path, 'rb')
        tmpDict = dill.load(modelFile)
        modelFile.close()
        self.__dict__.update(tmpDict)