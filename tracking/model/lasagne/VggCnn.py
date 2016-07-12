# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:29:56 2016

@author: MindLab
"""

# Model definition for VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# Download pretrained model from: https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from tracking.model.core.Cnn import Cnn
import lasagne
import numpy as NP
import cPickle as pickle

class VggCnn(Cnn):
    
    net = None
    meanImage = None
    
    def __init__(self, modelPath, layerKey):
        d = pickle.load(open(modelPath))
        self.net = self.buildModel()
        self.layerKey = layerKey
        lasagne.layers.set_all_param_values(self.net['prob'], d['param values'])
        self.meanImage = d['mean value'][NP.newaxis, NP.newaxis, :, NP.newaxis, NP.newaxis]

    def forward(self, inData):
        #data = self.preprocess(inData)
        return lasagne.layers.get_output(self.net[self.layerKey], inData)
    
    def preprocess(self, data):
        # In: (batch, time, height, width, channels). Out: (batch, time, channels, height, width)
        data = NP.swapaxes(NP.swapaxes(data, 3, 4), 2, 3)
        # Make BGR
        data = data[:,:,::-1,:,:]
        # Subtract mean
        data -= self.meanImage
        
        return data

    def buildModel(self):
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
        net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)

        return net


    def getOutputDim(self, inDims):
        outDims = lasagne.layers.get_output_shape(self.net[self.layerKey], (1, ) + inDims)
        
        return outDims
        
        
    def getParams(self):
        
        return []