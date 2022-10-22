"""
SIFT IMPLEMENTATION
from "Distinctive Image Features from Scale-Invariant Keypoints", Lowe, 2004.
Kangmin Park, 2022.10.20.
"""
import cv2
import numpy as np
from time import process_time
import numba as nb
from numba import jit, njit, int16, int64, float64
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
    :return: space, sigmas
    """
    print("CHAPTER 3 : GENERATE SCALE SPACE FOR AN IMAGE.")
    print("\tGENERATING SCALE SPACE FOR THE IMAGE..")
    startTime = process_time()
    # set parameters
    k = 2**(1/s)
    layer = img
    space = {}
    sigmas = [sigma*(k**layerIdx) for layerIdx in range(s+3)]
    for octaveIdx in range(octaveNum) :

        for layerIdx in range(s+3) :

            initSigma = sigmas[layerIdx]

            if layerIdx==0 and octaveIdx==0 : sigmaDiff = np.sqrt(initSigma**2-0.5**2) # 토대영상(=완전 처음 영상)
            elif layerIdx==0 : sigmaDiff = initSigma # 토대영상은 아니지만 각 옥타브의 처음 영상인 경우
            else : sigmaDiff = np.sqrt(initSigma**2 - prevSigma**2) # 토대영상도, 처음 양상도 아닌 경우

            layer = gaussian(img=layer,
                             sigmaX=sigmaDiff,
                             sigmaY=sigmaDiff)[:, :, np.newaxis]
            if layerIdx==0 : octaveStack = layer
            else : octaveStack = np.concatenate([octaveStack, layer], axis=-1)

            prevSigma = initSigma

        space[octaveIdx] = octaveStack
        layer = cv2.resize(src=octaveStack[:,:,-3],
                           dsize=None,
                           fy=0.5,
                           fx=0.5,
                           interpolation=cv2.INTER_AREA)
    endTime = process_time()
    print("\t- FINISHED GENERATING SCALE SPACE. TIME : {0}".format(endTime-startTime))
    return space, sigmas

def dog(scaleSpace):
    """
    build difference of the gaussian (dog) scale space
    :param scaleSpace: scale space from def scale space
    :return: dog space (dog)
    """
    print("\tGENERATING DOG SCALE SPACE FOR THE SCALE SPACE..")
    startTime = process_time()
    dogSpace = {}
    for octaveIdx in scaleSpace.keys():
        octave = scaleSpace[octaveIdx]
        for layerIdx in range(octave.shape[2]-1):
            layerDiff = (octave[:,:,layerIdx+1] - octave[:,:,layerIdx])[:, :, np.newaxis]
            if layerIdx==0 : dogStack = layerDiff
            else : dogStack = np.concatenate([dogStack, layerDiff], axis=-1)
        dogSpace[octaveIdx] = dogStack
    endTime = process_time()
    print("\t- FINISHED GENERATING DOG SCALE SPACE. TIME : {0}".format(endTime-startTime))
    return dogSpace

def extremaDetection(dogSpace,
                     s,
                     contrastThr = 0.04):
    """
    detect extram from dog space
    :param dogSpace: dog space from def dog
    :param s: num of intervals
    :param contrastThr: 샘플의 가운데 픽셀에 적용되는 임계값 (논문에는 설명하지 않는 값)
    :return:
    """
    print("\tSEARCHING EXTREMA LOCATIONS IN DOG SCALE SPACE..")
    startTime = process_time()
    thr = np.floor(0.5 * contrastThr / s * 255) # https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5
    extremaLocation = {}
    for octaveIdx in dogSpace.keys():
        dogOctave = dogSpace[octaveIdx]
        extreBool = extremaDetectionSub(dogOctave=dogOctave,
                                        thr=thr)
        subExtremaLocation = np.where(extreBool==1)
        extremaLocation[octaveIdx] = subExtremaLocation
    endTime = process_time()
    print("\t- FINISHED SEARCHING EXTREMA LOCATIONS. TIME : {0}".format(endTime-startTime))
    return extremaLocation

@jit(float64[:,:,:](float64[:,:,:], float64))
def extremaDetectionSub(dogOctave,
                        thr):
    """
    Sub function for def extramDetection
    """
    boolBox = np.ones_like(dogOctave)
    extrBox = np.zeros_like(dogOctave).astype(float64)
    for x in range(1, boolBox.shape[1] - 1):
        for y in range(1, boolBox.shape[0] - 1):
            for z in range(1, boolBox.shape[2] - 1):
                if boolBox[y, x, z] == 0: continue
                dVal = dogOctave[y, x, z]
                if abs(dVal) > thr :
                    sample = dogOctave[y - 1:y + 2, x - 1:x + 2, z - 1:z + 2]
                    results = np.sum((dVal > sample).flatten()) - 1
                    if results == 0 or results == 26:
                        boolBox[y - 1:y + 2, x - 1:x + 2, z - 1:z + 2] = 0
                        extrBox[y, x, z] = 1
    return extrBox


if __name__ == "__main__" :

    testDir = "D:\\cv\\data\\prac\\bcard.jpg"
    img = cv2.imread(testDir, cv2.IMREAD_GRAYSCALE).astype(float) # not normalized
    print(img.shape)
    s = 3

    space, sigmas = scaleSpace(img=img,
                               s=s,
                               octaveNum=5)
    dogSpace = dog(scaleSpace=space)
    extremaLocation = extremaDetection(dogSpace=dogSpace,
                                       s=s)