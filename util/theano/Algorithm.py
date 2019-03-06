# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:58:10 2016

@author: MindLab
"""

import theano as TH
import theano.tensor as THT
import numpy as NP

from collections import OrderedDict
        

"""
Calculates the new parameters values for the tracker.

@type    loss: tehano.tensor
@param   loss: The loss of the predictions for a batch
@type    params: list(theano.shared)
@param   params: The parameters of the tracker
@type    lr: float
@param   lr: The learning rate
@type    rho: float
@param   rho: the momentum
@type    epsilon: float
@param   epsilon: The ???????
@rtype:  updates: OrderedDict
@return: updates: The new values for the parameters
""" 
def rmsprop(loss, params, lr=0.0005, rho=0.9, epsilon=1e-6):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = OrderedDict()
    grads = TH.grad(loss, params)
    acc = [TH.shared(NP.zeros(p.get_value().shape, dtype=TH.config.floatX)) for p in params]
    for p, g, a in zip(params, grads, acc):
        new_a = rho * a + (1 - rho) * g ** 2
        updates[a] = new_a
        new_p = p - lr * g / THT.sqrt(new_a + epsilon)
        updates[p] = new_p

    return updates