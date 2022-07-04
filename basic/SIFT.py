import os
import sys
import cv2
import numpy as np
from numba import jit
from time import process_time
from matplotlib import pyplot as plt

from filters import *

class feature :

    def __init__(self, img):

        scaleSpace = self.scaleSpace(img=img)
        self.octaveDepth = len(scaleSpace.keys())
        DOG = self.DOG(scaleSpace)

    def scaleSpace(self,
                   img,
                   k=1.6,
                   widthThr=16):

        # get image shape
        shape = img.shape
        width = shape[0]*shape[1]

        # define scale space
        scaleSpace = {}

        # build scale space
        octaveOrder = 0
        octaveLayer = img

        while width > widthThr :
            initSigma = 1.6
            beforeSigma = 0
            # make octave
            for idx in range(6):
                # from the second octave, the base image is not necessary to gaussian filter
                if idx==0 and octaveOrder!=0 :
                    octave = octaveLayer[:,:,np.newaxis]
                    continue

                sigma = initSigma*(k**idx)
                sigmaDiff = np.sqrt(sigma**2-beforeSigma**2)
                if idx==0 and octaveOrder==0 : sigmaDiff = np.sqrt(initSigma ** 2 - 0.25)
                octaveLayer = gaussian(octaveLayer,
                                       sigmaX=sigmaDiff,
                                       sigmaY=sigmaDiff)[:,:,np.newaxis]
                # stack
                if idx==0 : octave = octaveLayer
                else : octave = np.concatenate([octave, octaveLayer], axis=-1)
                beforeSigma = sigma

            scaleSpace[octaveOrder] = octave
            octaveOrder += 1
            octaveLayer = cv2.resize(src=scaleSpace[octaveOrder - 1][:, :, 3],
                                     dsize=None,
                                     fy=0.5,
                                     fx=0.5,
                                     interpolation=cv2.INTER_AREA)
            width = octaveLayer.shape[0]*octaveLayer.shape[1]
        return scaleSpace

    def DOG(self,
            scaleSpace):

        # DOG space
        DOG = {}
        # build DOG SPACE
        for idx in range(self.octaveDepth) :
            DOGLayer1 = scaleSpace[idx][:,:,1]-scaleSpace[idx][:,:,0]
            DOGLayer2 = scaleSpace[idx][:,:,2]-scaleSpace[idx][:,:,1]
            DOGLayer3 = scaleSpace[idx][:,:,3]-scaleSpace[idx][:,:,2]
            DOGLayer4 = scaleSpace[idx][:,:,4]-scaleSpace[idx][:,:,3]
            DOGLayer5 = scaleSpace[idx][:,:,5]-scaleSpace[idx][:,:,4]
            DOG[idx] = np.concatenate([DOGLayer1, DOGLayer2, DOGLayer3, DOGLayer4, DOGLayer5], axis=-1)
        return DOG

    def featureDetection(self):
        return

if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)

    t1 = process_time()
    feature(img=img)
    t2 = process_time()
