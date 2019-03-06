# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:28:12 2016

@author: MindLAB

"""

from keras import backend as K
from keras.layers import Activation, Dense, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from tracking.model.keras.Cnn import Cnn

class ResNet(Cnn):
    
    def __init__(self, modelPath, layerKey):
        self.layerKey = layerKey
        self.build(modelPath)
        
        
    def build(self, modelPath):
        input_shape = (3, 224, 224)
        img_input = Input(shape=input_shape)
        bn_axis = 1
    
        x = ZeroPadding2D((3, 3))(img_input)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', mode=2)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten(name='flat')(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)
        
        baseModel = Model(img_input, x)
        baseModel.load_weights(modelPath)
        model = Model(input=baseModel.input, output=baseModel.get_layer(self.layerKey).output)
        self.model = TimeDistributed(model)
        

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        '''The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        '''
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', mode=2)(x)
        x = Activation('relu')(x)
    
        x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                          border_mode='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', mode=2)(x)
        x = Activation('relu')(x)
    
        x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', mode=2)(x)
    
        x = merge([x, input_tensor], mode='sum')
        x = Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        '''conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        '''
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                          name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', mode=2)(x)
        x = Activation('relu')(x)
    
        x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', mode=2)(x)
        x = Activation('relu')(x)
    
        x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', mode=2)(x)
    
        shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', mode=2)(shortcut)
    
        x = merge([x, shortcut], mode='sum')
        x = Activation('relu')(x)
        
        return x       
    
        
    def getOutputShape(self):
        outDims = self.model.layer.output_shape
        
        return outDims
        
        
    def getModel(self):
        return self.model
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layer.layers:
            layer.trainable = trainable