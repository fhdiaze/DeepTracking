# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:06:45 2016

@author: MindLab
"""

from keras.models import Model
from tracking.model.keras.Module import Module

class Rnn(Module):
    
    def __init__(self, input, layers):
        self.build(input, layers)
        
    
    def build(self, input, layers):
        output = input
        
        for layer in layers:
            output = layer(output)
        
        self.model = Model(inputs=input, outputs=output)
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable
           
           
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        cfg = self.getModel().get_config()
        
        for lay in cfg["layers"]:
            conf = lay["config"]
            
            if "stateful" in conf:
                conf["stateful"] = stateful
            
            if "batch_input_shape" in conf:
                inShape = conf["batch_input_shape"]
                inShape = (batchSize, None, inShape[2])
                conf["batch_input_shape"] = inShape
            
        model = Model.from_config(cfg)
        weights = self.model.get_weights()
        model.set_weights(weights)
        self.model = model