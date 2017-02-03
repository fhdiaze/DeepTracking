# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

import logging
import numpy as NP
from keras.callbacks import Callback
from keras.models import Model
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, input=None, modules=None, builder=None, optimizer=None, loss=None, processor=None, timeSize=None, metrics=None):
        self.input = input
        self.modules = modules
        self.builder = builder
        self.optimizer = optimizer
        self.loss = loss
        self.processor = processor
        self.timeSize = timeSize
        self.metrics = metrics
    
    
    def fit(self, frame, position, lnr):
        frame, position = self.processor.preprocess(frame, position)
        loss = self.model.train_on_batch(frame, position)
        
        return loss
        
    
    def forward(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        position = self.model.predict_on_batch(frame)
        frame, position = self.processor.postprocess(frame, position)
        
        return position
    
    
    def track(self, frame, initPosition):
        batchSize = frame.shape[0]
        seqLength = frame.shape[1]
        targetDim = initPosition.shape[-1]
        #iters = seqLength / self.timeSize + (seqLength % self.timeSize > 0)
        position = NP.empty((batchSize, 0, targetDim))
        predPosition = initPosition
        
        for i in range(seqLength):
            start = i
            end = i + self.timeSize
            pFrame = frame[:, start:end, ...]
            predPosition = self.step(pFrame, predPosition)
            position = NP.concatenate((position, predPosition), axis=1)
        
        return position
    
    
    def build(self):
        output = self.builder.build(self.input, self.modules)
        model = Model(input=self.input, output=output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model = model
        
        
    def train(self, generator, epochs, batches, batchSize, validator):
        history = LossHistory(validator, self)
        spe = batches * batchSize
        self.model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=spe, verbose=0, callbacks=[history])
        
    
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        
        for name, module in self.modules.items():
            module.setStateful(stateful, batchSize)
            
        self.build()
        
    
    def reset(self):
        self.model.reset_states()
        
        
    def getWeights(self):
        
        return self.model.get_weights()
        
    
    def setWeights(self, weights):
        self.model.set_weights(weights)
        
        
    def step(self, frame, position):
        input, position = self.processor.preprocess(frame, position)
        position = self.model.predict_on_batch(input)
        input, position = self.processor.postprocess(input, position)
        
        return position
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker


    def on_batch_end(self, batch, logs={}):
        loss = logs.get("loss")
        measure = logs.get("calculateGpu").mean()
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", 0, batch, loss)
        logging.info("Validation Batch: %s = %f", "Overlap", measure)
        
        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)
