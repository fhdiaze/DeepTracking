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
    """
    def __init__(self, frames, positionModel):
        self.frames = frames
        self.boxes = {}
        self.positionModel = positionModel

    
    """
    Add bounding boxes to the frames.

    @type  position:  [[number]]
    @param position:  A list of list. Each element must be a list containing the points of a polygon.
    @type  outline: string
    @param outline: The color for the boxes.
    """
    def addBoxes(self, position, outline):
        # add the new bounding boxes to the dictionary
        self.boxes[outline] = position

    
    """
    Return the frames with bounding boxes drawn.

    @rtype:  iterator(PIL.Image)
    @return: An iterator over the frames.
    """
    def getFramesWithBoxes(self):
        for index, frame in enumerate(self.frames, start=0):
            frameBoxes = [(outline, boxes[index]) for outline, boxes in self.boxes.items()]
            self.plotBoxes(frame, frameBoxes)
            frame = self.resizeImage(frame)
            yield frame
    
    
    """
    Plot many bounding boxes in a frame.

    @type    frame:           PIL.Image
    @param   frame:           The frame
    @type    outlineBoxPairs: [(string, [])]
    @param   outlineBoxPairs: The list of color, points of a polygon
    """ 
    def plotBoxes(self, frame, outlineBoxPairs):
        for outline, box in outlineBoxPairs:
            self.positionModel.plot(frame, box, outline)
    
    
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