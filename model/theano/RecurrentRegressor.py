# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:39:20 2016

@author: MindLab
"""

import numpy as NP
import theano as TH
import theano.tensor as THT
import theano.tensor.nnet as NNET
import tracking.util.data.Random as Random
from tracking.model.core.Regressor import Regressor

class RecurrentRegressor(Regressor):

    def __init__(self, inputDim, targetDim, zeroTailFc):
        self.WFc, self.bFc, self.Wz, self.Uz, self.bz = self.initRegressor(inputDim, targetDim, zeroTailFc)
        self.trainable = True
    
    
    def forward(self, prevPos, data):
        c = THT.tanh(THT.dot(data, self.WFc) + self.bFc)
        z = NNET.sigmoid(THT.dot(data, self.Wz) + THT.dot(prevPos, self.Uz) + self.bz)
        position = (1 - z) * prevPos + z * c
        
        return position
        
        
    def initRegressor(self, stateDim, targetDim, zeroTailFc):
        if not zeroTailFc:
            WFcinit = Random.glorotUniform((stateDim, targetDim))
            WzInit = Random.glorotUniform((stateDim, targetDim))
            UzInit = Random.glorotUniform((targetDim, targetDim))
        else:
            WFcinit = NP.zeros((stateDim, targetDim), dtype=TH.config.floatX)
            WzInit = NP.zeros((stateDim, targetDim), dtype=TH.config.floatX)
            UzInit = NP.zeros((targetDim, targetDim), dtype=TH.config.floatX)
            
        WFc = TH.shared(WFcinit, name='WFc')
        Wz = TH.shared(WzInit, name='Wz')
        Uz = TH.shared(UzInit, name='Uz')
        bFc = TH.shared(NP.zeros((targetDim,), dtype=TH.config.floatX), name='bFc')
        bz = TH.shared(NP.zeros((targetDim,), dtype=TH.config.floatX), name='bz')
        
        return WFc, bFc, Wz, Uz, bz
        
    
    def getParams(self):
        params = []
        
        if self.trainable:
            params = [self.WFc, self.bFc, self.Wz, self.Uz, self.bz]
            
        return params
        
    
    def setTrainable(self, trainable):
        self.trainable = trainable