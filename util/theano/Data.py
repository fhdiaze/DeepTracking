# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:02:40 2016

@author: MindLab
"""

import theano as TH
import theano.tensor as THT

class Data(object):
    
    @staticmethod
    def linspace(start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = THT.cast(start, TH.config.floatX)
        stop = THT.cast(stop, TH.config.floatX)
        num = THT.cast(num, TH.config.floatX)
        step = (stop-start)/(num-1)
        return THT.arange(num, dtype=TH.config.floatX) * step + start
        

    """
    Creates a theano tensor with specif dimensions
    
    @type    name:         String
    @param   name:         The name for the tensor
    @type    dtype:        dtype
    @param   dtype:        The type of the tensor
    @type    dim:          dtype
    @param   dim:          The type of the tensor
    @rtype:  theano.tensor
    @return: The theano tensor
    """ 
    @staticmethod
    def getTensor(name, dtype, dim):
        
        return THT.TensorType(dtype, [False] * dim, name=name)()