# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 22:53:44 2016

@author: MindLAB
"""

class Module(object):
    
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        pass
    
    
    def getModel(self):
        
        return self.model
        
    
    def getOutputShape(self):
        
        return self.model.output_shape