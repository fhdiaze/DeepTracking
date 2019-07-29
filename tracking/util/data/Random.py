# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:06:47 2016

@author: MindLab
"""

import numpy as NP
import theano as Theano
import numpy.random as Random
    
    
def getFans(shape):
    '''
    Borrowed from keras
    '''
    fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def orthogonal(shape, scale=1.1):
    '''
    Borrowed from keras
    '''
    flat_shape = (shape[0], NP.prod(shape[1:]))
    a = Random.normal(0, 1, flat_shape)
    u, _, v = NP.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    
    return NP.cast[Theano.config.floatX](q)
    
    
def glorotUniform(shape):
    '''
    Borrowed from keras
    '''
    fan_in, fan_out = getFans(shape)
    s = NP.sqrt(6. / (fan_in + fan_out))
    return NP.cast[Theano.config.floatX](Random.uniform(low=-s, high=s, size=shape))