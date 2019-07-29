# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:38:35 2015

@author: MindLab
"""

import subprocess
import tempfile
import shutil
import os
from PIL import Image


class Sequence(object):
    PROCESS_TEMPLATE = 'avconv -y -f image2pipe -vcodec mjpeg -r {} -i - -vcodec libx264 -qscale 5 -r {} {}'
    PROCESS_TEMPLATE_OFFLINE = 'avconv -y -f image2 -vcodec mjpeg -r {} -i {} -vcodec libx264 -qscale 5 -r {} {}'
    
    
    """
    Create a Sequence base on a list of frames.

    @type  frames: iterator(PIL.Image)
    @param frames: The frames iterator
    @type  positionModel: model.core.PositionModel
    @param positionModel: The model of object positions
    """
    def __init__(self, frames, positionModel):
        self.frames = frames
        self.positionModel = positionModel
        self.boxes = []

    
    """
    Add bounding boxes to the frames.

    @type  position: [[number]]
    @param position: Each element must be of the form [frame, id, position representation].
    """
    def addBoxes(self, position):
        # add the new positions
        self.boxes = position
        
    
    """
    Return the frames with bounding boxes drawn.

    @rtype:  iterator(PIL.Image)
    @return: An iterator over the frames.
    """
    def getFramesWithBoxes(self):
        for index, frame in enumerate(self.frames, start=1):
            frameBoxes = [box[1:] for box in self.boxes if box[0] == index]
            self.plotBoxes(frame, frameBoxes)
            frame = self.resizeImage(frame)
            yield frame
    
    
    """
    Plot many bounding boxes in a frame.

    @type  frame:           PIL.Image
    @param frame:           The frame
    @type  position: [[number]]
    @param position: Each element must be of the form [id, position representation].
    """ 
    def plotBoxes(self, frame, position):
        for p in position:
            outline = "blue"
            self.positionModel.plot(frame, p[1:], outline)
    
    
    """
    Correct image size to be even as needed by video codec.

    @type    image:   PIL.Image
    @param   image:   The frame
    @rtype:  PIL.image
    @return: The resized image
    """ 
    def resizeImage(self, image):
        evenSize = list(image.size)
        resize = False
        
        for index in range(len(evenSize)):
            if evenSize[index] % 2 == 1:
                evenSize[index] += 1
                resize = True
        
        if(resize):
            evenSize = tuple(evenSize)
            image = image.resize(evenSize, Image.ANTIALIAS)
        
        return image
    
    
    """
    Export the sequence to a video.

    @type    fps:    number
    @param   fps:    Frames per second
    @type    output: string
    @param   output: The name of the output video.
    """ 
    def exportToVideoPiped(self, fps, output):
        conversionProcess = subprocess.Popen(self.PROCESS_TEMPLATE.format(fps, fps, output).split(' '), stdin=subprocess.PIPE)
        
        for frame in self.getFramesWithBoxes():
            frame.save(conversionProcess.stdin, 'JPEG')

        conversionProcess.stdin.close()
        conversionProcess.wait()

        
    """
    Export the sequence to a video.

    @type    fps:    number
    @param   fps:    Frames per second
    @type    output: string
    @param   output: The name of the output video.
    """ 
    def exportToVideo(self, fps, output):
        tempPath = tempfile.mkdtemp()
        processString = self.PROCESS_TEMPLATE_OFFLINE.format(fps, os.path.join(tempPath, '%08d.jpg'), fps, output)
        
        for index, frame in enumerate(self.getFramesWithBoxes(), start=0):
            frame.save(os.path.join(tempPath, '{:08d}.jpg'.format(index)), format='JPEG')
        
        conversionProcess = subprocess.Popen(processString.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        conversionProcess.wait()
        shutil.rmtree(tempPath)