# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:24:21 2016

@author: MindLAB
"""

import theano.tensor as THT
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from tracking.model.keras.Module import Module
from tracking.util.theano.Data import Data


class SpatialTransformer(Module):
    
    def __init__(self, input, downsampleFactor=1):
        if type(input) is not list or len(input) != 2:
            raise Exception("SpatialTransformer must be called on a list of two tensors. Got: " + str(input))
        
        output = TranformerLayer(downsampleFactor)(input)
        self.model = Model(inputs=input, outputs=output)
    
    
    def getModel(self):
        
        return self.model
        
        
class TranformerLayer(Layer):
    
    def __init__(self, downsampleFactor, **kwargs):
        self.downsampleFactor = downsampleFactor
        super(TranformerLayer, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        super(TranformerLayer, self).build(input_shape)

    
    def call(self, X):
        if type(X) is not list or len(X) != 2:
            raise Exception("TranformerLayer must be called on a list of two tensors. Got: " + str(X))
        
        frame, theta = X[0], X[1]
        
        # Reshaping the input to exclude the time dimension
        frameDim = K.ndim(frame)
        frameShape = K.shape(frame)
        (chans, height, width) = frameShape[-3:]
        frame = K.reshape(frame, (-1, chans, height, width))
        theta = K.reshape(theta, (-1, 3, 3))[:, :2, :]
        
        # Applying the spatial transformation
        output = TranformerLayer.transform(theta, frame, self.downsampleFactor)

        # Reshaping the frame to include time dimension
        outputShape = K.shape(output)
        outputShape = K.concatenate([frameShape[:-2], outputShape[-2:]])
        output = THT.reshape(output, outputShape, ndim=frameDim)
        
        return output

    
    def compute_output_shape(self, inputShapes):
        frameShape = inputShapes[0]
        height = int(frameShape[-2] / self.downsampleFactor)
        width = int(frameShape[-1] / self.downsampleFactor)
        outputShape = frameShape[:-2] + (height, width)
        
        return outputShape
        
    
    @staticmethod
    def transform(theta, input, downsample_factor):
        num_batch, num_channels, height, width = input.shape
        theta = THT.reshape(theta, (-1, 2, 3))
    
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = THT.cast(height / downsample_factor, 'int64')
        out_width = THT.cast(width / downsample_factor, 'int64')
        grid = TranformerLayer.meshgrid(out_height, out_width)
    
        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = THT.dot(theta, grid)
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()
    
        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = TranformerLayer.interpolate(
            input_dim, x_s_flat, y_s_flat,
            out_height, out_width)
    
        output = THT.reshape(
            input_transformed, (num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
        
        return output


    @staticmethod
    def meshgrid(height, width):
        # This function is the grid generator from eq. (1) in reference [1].
        # It is equivalent to the following numpy code:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        # It is implemented in Theano instead to support symbolic grid sizes.
        # Note: If the image size is known at layer construction time, we could
        # compute the meshgrid offline in numpy instead of doing it dynamically
        # in Theano. However, it hardly affected performance when we tried.
        x_t = THT.dot(THT.ones((height, 1)),
                    Data.linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = THT.dot(Data.linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    THT.ones((1, width)))
    
        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = THT.ones_like(x_t_flat)
        grid = THT.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        
        return grid
        
    
    @staticmethod
    def interpolate(im, x, y, out_height, out_width):
        # *_f are floats
        num_batch, height, width, channels = im.shape
        height_f = THT.cast(height, K.floatx())
        width_f = THT.cast(width, K.floatx())
    
        # clip coordinates to [-1, 1]
        x = THT.clip(x, -1, 1)
        y = THT.clip(y, -1, 1)
    
        # scale coordinates from [-1, 1] to [0, width/height - 1]
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)
    
        # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing. for
        # indexing, we need to take care they do not extend past the image.
        x0_f = THT.floor(x)
        y0_f = THT.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1
        x0 = THT.cast(x0_f, 'int64')
        y0 = THT.cast(y0_f, 'int64')
        x1 = THT.cast(THT.minimum(x1_f, width_f - 1), 'int64')
        y1 = THT.cast(THT.minimum(y1_f, height_f - 1), 'int64')
    
        # The input is [num_batch, height, width, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width, channels]. We need
        # to offset all indices to match the flat version
        dim2 = width
        dim1 = width*height
        base = THT.repeat(
            THT.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
    
        # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]
    
        # calculate interpolated values
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = THT.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        
        return output