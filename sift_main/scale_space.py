"""
SIFT IMPLEMENTATION
from "Distinctive Image Features from Scale-Invariant Keypoints", Lowe, 2004.
Kangmin Park, 2022.10.20.
"""
import cv2
import numpy as np
from basic.filters import *
"""
Chapter 3.
Detection of Scale-Space Extrema
"""
def scaleSpace(img,
               s,
               octaveNum,
               sigma=1.6):
    """
    build Scale Space of the input image
    :param img: type = Gray Scale, image instance (not directory)
    :param s: the number of layer for each octavStack is (s+3)
    :param octaveNum: the number of octavStack
    :param sigma: gaussian scale parameter (default = 1.6)
    :return:
    """
    # set parameters
    k = 2**(1/s)
    layer = img
    space = {}
    sigmas = {}

    for octaveIdx in range(octaveNum) :

        sigmaList = []

        for layerIdx in range(s+3) :

            initSigma = sigma * (k**layerIdx)
            sigmaList.append(initSigma)

            if layerIdx==0 and octaveIdx==0 : sigmaDiff = np.sqrt(initSigma**2-0.5**2) # 토대영상(=완전 처음 영상)
            elif layerIdx==0 : sigmaDiff = initSigma # 토대영상은 아니지만 각 옥타브의 처음 영상인 경우
            else : sigmaDiff = np.sqrt(initSigma**2 - prevSigma**2) # 토대영상도, 처음 양상도 야닌 경우

            layer = gaussian(img=layer,
                             sigmaX=sigmaDiff,
                             sigmaY=sigmaDiff)[:, :, np.newaxis]
            if layerIdx==0 : octaveStack = layer
            else : octaveStack = np.concatenate([octaveStack, layer], axis=-1)

            prevSigma = initSigma

        sigmas[octaveIdx] = sigmaList
        space[octaveIdx] = octaveStack
        layer = cv2.resize(src=octaveStack[:,:,-3],
                           dsize=None,
                           fy=0.5,
                           fx=0.5,
                           interpolation=cv2.INTER_AREA)

        return space, sigmas

if __name__ == "__main__" :

    testDir = "D:\\cv\\data\\prac\\bcard.jpg"
    img = cv2.imread(testDir, cv2.IMREAD_GRAYSCALE)

    space, sigmas = scaleSpace(img=img,
                               s=3,
                               octaveNum=5)

    print(space)

