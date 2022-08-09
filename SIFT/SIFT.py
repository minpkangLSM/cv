import os
import sys
import cv2
import numpy as np
from numba import jit, njit, uint8, int64
from time import process_time
from matplotlib import pyplot as plt

from basic.filters import *

class feature :

    @staticmethod
    def processor(img):
        scaleSpace = feature.__scaleSpace(img=img)
        DOG = feature.__DOG(scaleSpace=scaleSpace)
        feature.featureDetection(dog=DOG)

    @staticmethod
    def __scaleSpace(img,
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

    @staticmethod
    def __DOG(scaleSpace):

        octaveDepth = len(scaleSpace.keys())

        # DOG space
        DOG = {}
        # build DOG SPACE
        for idx in range(octaveDepth) :
            DOGLayer1 = (scaleSpace[idx][:,:,1]-scaleSpace[idx][:,:,0])[:,:,np.newaxis]
            DOGLayer2 = (scaleSpace[idx][:,:,2]-scaleSpace[idx][:,:,1])[:,:,np.newaxis]
            DOGLayer3 = (scaleSpace[idx][:,:,3]-scaleSpace[idx][:,:,2])[:,:,np.newaxis]
            DOGLayer4 = (scaleSpace[idx][:,:,4]-scaleSpace[idx][:,:,3])[:,:,np.newaxis]
            DOGLayer5 = (scaleSpace[idx][:,:,5]-scaleSpace[idx][:,:,4])[:,:,np.newaxis]

            DOG[idx] = np.concatenate([DOGLayer1, DOGLayer2, DOGLayer3, DOGLayer4, DOGLayer5], axis=-1)
        return DOG

    @staticmethod
    @njit (int64(int64[:,:,:], int64[:], int64))
    def _featureDetection(dogOctave, featureList, octaveDepth):

        shape = dogOctave.shape
        for height in range(1,shape[0]-1):
            for width in range(1,shape[1]-1):
                for depth in range(1,shape[2]-1):
                    candidate = dogOctave[height, width, depth]
                    # surround
                    surround = dogOctave[height-1:height+2, width-1:width+2, depth-1:depth+2]
                    if (candidate >= surround).all() or (candidate <= surround).all() :
                        f = np.array([[height], [width], [octaveDepth], [depth]])
                        featureList = np.stack([featureList, f])

        return 1

    @staticmethod
    def featureDetection(dog):

        dogDepth = len(dog.keys())
        featureList = np.array([[0],[0],[0],[0]],dtype=np.int64)
        for depth in range(dogDepth):
            feature._featureDetection(dog[depth].astype(np.int64), featureList, depth)

        return

if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\parts\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)

    t1 = process_time()
    feature.processor(img)
    t2 = process_time()
    print(t2-t1)
