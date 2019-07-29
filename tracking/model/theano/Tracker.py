# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:59:51 2016

@author: MindLab
"""

import theano as TH
import theano.tensor as THT
import tracking.util.theano.Data as Data
import numpy as NP
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker): 
    
    """
    Creates a Tracker based in some components.

    @type  rnn: Rnn instance
    @param rnn: The Rnn instance
    @type  cnn: Cnn instance
    @param cnn: The Cnn instance
    @type  regressor: Regressor instance
    @param regressor: The Regressor instance
    @type  objective: Objective instance
    @param objective: The loss function
    @type  stateDim: iterator(PIL.Image)
    @param stateDim: The frames iterator
    """
    def __init__(self, rnn, cnn, regressor, objective, optimizer):
        # Components
        self.rnn = rnn
        self.cnn = cnn
        self.regressor = regressor
        self.objective = objective
        self.optimizer = optimizer
        self.rnnState = None
        self.stateful = False
        
        # Functions
        self.buildModel()
        
    
    def fit(self, frame, position, lnr):
        batchSize = frame.shape[0]
        initPosition = position[:, 0, :]
        initRnnState = NP.zeros((batchSize, self.rnn.stateDim))
        loss = self.fitFunc(frame, initPosition, position, initRnnState, lnr)

        return loss
    

    def forward(self, frame, initPosition):
        batchSize = frame.shape[0]
        rnnState = self.getRnnState(batchSize, self.rnn.stateDim)
        predPosition, self.rnnState = self.forwardFunc(frame, initPosition, rnnState)

        return predPosition
    
    
    # TODO: Template method???
    def buildModel(self):
        # frame.shape = (batchSize, seqLen, nrChannels, frameHeight, frameWidth)
        frame = Data.getTensor("frame", TH.config.floatX, 5)
        
        # position.shape = (batchSize, targetDim)
        position = THT.matrix()

        # state.shape = (batchSize, stateDim)
        state = THT.matrix()
        
        lnr = THT.scalar()
        
        # Move the time (frame) axis to the top. Iterate over frames using all batch
        scanOut, _ = TH.scan(self.step, sequences=[frame.dimshuffle(1, 0, 2, 3, 4)], outputs_info=[position, state])

        # predPosition.shape = (batchSize, targetDim)        
        predPosition = scanOut[0].dimshuffle(1, 0, 2)
        
         # gtPosition: of shape (batchSize, seqLen, targetDim)
        gtPosition = Data.getTensor("gtPosition", TH.config.floatX, 3)
    
        loss = self.objective.evaluate(gtPosition, predPosition)
        
        params = self.getParams()
        
        fitFunc = TH.function([frame, position, gtPosition, state, lnr], loss, updates=self.optimizer(loss, params, lnr), allow_input_downcast=True)
        forwardFunc = TH.function([frame, position, state], [predPosition, state], allow_input_downcast=True)
        
        self.fitFunc, self.forwardFunc = fitFunc, forwardFunc
    
    
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
    def step(self, frame, prevPosition, prevState):
        features = self.cnn.forward(frame)
        flat = THT.reshape(features, (features.shape[0], THT.prod(features.shape[1:])))
        inData = THT.concatenate([flat, prevPosition], axis=1)
        state = self.rnn.forward(inData, prevState)        
        newPosition = self.regressor.forward(prevPosition, state)
        
        return newPosition, state
        
        
    def getParams(self):
        return self.rnn.getParams() + self.regressor.getParams() + self.cnn.getParams()
        
        
    def getRnnState(self, batchSize, stateDim):
        if self.rnnState is None or not self.stateful:
            self.rnnState = NP.zeros((batchSize, stateDim))
        
        return self.rnnState
        
    
    def reset(self):
        
        self.rnnState = None
        
    
    def setStateful(self, stateful, batchSize):
        self.stateful = stateful