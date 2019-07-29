# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:51:57 2016

@author: MindLab
"""

import numpy as NP
import theano as TH
import theano.tensor as THT
import theano.tensor.nnet as NNET
import tracking.util.data.Random as Random
from tracking.model.core.Rnn import Rnn


class SingleGru(Rnn):
    
    """
    Create a GRU unit.

    @type  inputDim: Integer
    @param inputDim: GRU input tensor's shape
    """
    def __init__(self, inputDim, stateDim):
        self.stateDim = stateDim
        self.Wr, self.Ur, self.br, self.Wz, self.Uz, self.bz, self.Wg, self.Ug, self.bg = self.initGru(inputDim, stateDim)
        self.trainable = True
    
    
    def forward(self, data, h):
        z = NNET.sigmoid(THT.dot(data, self.Wz) + THT.dot(h, self.Uz) + self.bz)
        r = NNET.sigmoid(THT.dot(data, self.Wr) + THT.dot(h, self.Ur) + self.br)
        c = THT.tanh(THT.dot(data, self.Wg) + THT.dot(r * h, self.Ug) + self.bg)
        out = (1 - z) * h + z * c
        
        return out
        
    
    def initGru(self, inputDim, stateDim):
        Wr = TH.shared(Random.glorotUniform((inputDim, stateDim)), name='Wr')
        Ur = TH.shared(Random.orthogonal((stateDim, stateDim)), name='Ur')
        br = TH.shared(NP.zeros((stateDim,), dtype=TH.config.floatX), name='br')
        Wz = TH.shared(Random.glorotUniform((inputDim, stateDim)), name='Wz')
        Uz = TH.shared(Random.orthogonal((stateDim, stateDim)), name='Uz')
        bz = TH.shared(NP.zeros((stateDim,), dtype=TH.config.floatX), name='bz')
        Wg = TH.shared(Random.glorotUniform((inputDim, stateDim)), name='Wg')
        Ug = TH.shared(Random.orthogonal((stateDim, stateDim)), name='Ug')
        bg = TH.shared(NP.zeros((stateDim,), dtype=TH.config.floatX), name='bg')
        
        return Wr, Ur, br, Wz, Uz, bz, Wg, Ug, bg
        
    def getParams(self):
        params = []
        
        if self.trainable:
            params = [self.Wr, self.Ur, self.br, self.Wz, self.Uz, self.bz, self.Wg, self.Ug, self.bg]
            
        return params
        
    
    def setTrainable(self, trainable):
        self.trainable = trainable