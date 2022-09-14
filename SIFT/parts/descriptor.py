import cv2
import math
import numpy as np
from numba import jit
from basic.filters import *
from time import process_time
from featureExtractor import *

class orientation :

    @staticmethod
    def assign(dogSpace,
               sigmas,
               features):

        # setting to calculate gradient, orientation of key points
        octaveIdx = dogSpace.keys()
        oriFeatures = {}

        # calculate orientation histogram
        for idx in octaveIdx :

            newY = np.array([])
            newX = np.array([])
            newZ = np.array([])
            newOri = np.array([])

            # step 1 : 키포인트의 scale에 해당하는 이미지 L에 대한 기울기와 기울기 크기 계산을 위해 해당 옥타브 전체 계산
            subDog = dogSpace[idx]
            dy = sobelHeightAxis(subDog,
                                 ddepth=cv2.CV_64F)
            dx = sobelWidthAxis(subDog,
                                ddepth=cv2.CV_64F)
            magnitude = np.sqrt(dy*dy+dx*dx)
            theta = np.arctan2(dy,dx)*180/np.pi # rad to deg
            theta = orientation.__quantize(theta)

            locYs = features[idx][0]
            locXs = features[idx][1]
            locZs = features[idx][2]
            for locY, locX, locZ in zip(locYs, locXs, locZs):
                # orientation histogram
                oriHisto = {}

                # get L for each key point
                sigma = sigmas[idx][locZ] * 1.5
                LMagn = magnitude[:,:,locZ]
                LOri = theta[:,:,locZ]

                # set the range of histogram
                rangeYHead = int(max(0, locY-sigma/2))
                rangeYRear = int(min(LMagn.shape[1], locY+sigma/2))
                rangeXHead = int(max(0, locX-sigma/2))
                rangeXRear = int(min(LMagn.shape[1], locX+sigma/2))
                magSur = LMagn[rangeYHead:rangeYRear, rangeXHead:rangeXRear]
                oriSur = LOri[rangeYHead:rangeYRear, rangeXHead:rangeXRear]

                magShape = magSur.shape
                maxShape = max(magShape[0], magShape[1])
                gWeight = gaussianFilter(shape=magShape,
                                         sigma=maxShape/6)
                weightedMagSur = magSur*gWeight
                oriList = list(set(oriSur.flatten()))
                for ori in oriList :
                    count = np.sum(weightedMagSur[oriSur==ori])
                    oriHisto[ori] = count
                maxKV = max(zip(oriHisto.values(), oriHisto.keys()))
                maxKey = maxKV[1]
                maxVal = maxKV[0]
                for val, key in zip(oriHisto.values(), oriHisto.keys()) :
                    if val >= maxVal*0.8 and key != maxKey :
                        newY = np.append(newY, locY)
                        newX = np.append(newX, locX)
                        newZ = np.append(newZ, locZ)
                        newOri = np.append(newOri, key)

            oriFeatures[idx] = (newY,
                                newX,
                                newZ,
                                newOri)

        return oriFeatures

    @staticmethod
    def __quantize(theta):
        """
        sub function of def. assign
        quantize theta(Deg) into 36 bins(0~35)
        :param theta: unit -> deg
        :return:
        """
        quantizedDir = np.zeros_like(theta)
        quantizedDir[np.where((theta >= 0) & (theta < 10))] = 0
        quantizedDir[np.where((theta >= 10) & (theta < 20))] = 1
        quantizedDir[np.where((theta >= 20) & (theta < 30))] = 2
        quantizedDir[np.where((theta >= 33) & (theta < 40))] = 3
        quantizedDir[np.where((theta >= 40) & (theta < 50))] = 4
        quantizedDir[np.where((theta >= 50) & (theta < 60))] = 5
        quantizedDir[np.where((theta >= 60) & (theta < 70))] = 6
        quantizedDir[np.where((theta >= 70) & (theta < 80))] = 7
        quantizedDir[np.where((theta >= 80) & (theta < 90))] = 8
        quantizedDir[np.where((theta >= 90) & (theta < 100))] = 9
        quantizedDir[np.where((theta >= 100) & (theta < 110))] = 10
        quantizedDir[np.where((theta >= 110) & (theta < 120))] = 11
        quantizedDir[np.where((theta >= 120) & (theta < 130))] = 12
        quantizedDir[np.where((theta >= 130) & (theta < 140))] = 13
        quantizedDir[np.where((theta >= 140) & (theta < 150))] = 14
        quantizedDir[np.where((theta >= 150) & (theta < 160))] = 15
        quantizedDir[np.where((theta >= 160) & (theta < 170))] = 16
        quantizedDir[np.where((theta >= 170) & (theta <= 180))] = 17
        quantizedDir[np.where((theta < 0) & (theta >= -10))] = 18
        quantizedDir[np.where((theta < -10) & (theta >= -20))] = 19
        quantizedDir[np.where((theta < -20) & (theta >= -30))] = 20
        quantizedDir[np.where((theta < -30) & (theta >= -40))] = 21
        quantizedDir[np.where((theta < -40) & (theta >= -50))] = 22
        quantizedDir[np.where((theta < -50) & (theta >= -60))] = 23
        quantizedDir[np.where((theta < -60) & (theta >= -70))] = 24
        quantizedDir[np.where((theta < -70) & (theta >= -80))] = 25
        quantizedDir[np.where((theta < -80) & (theta >= -90))] = 26
        quantizedDir[np.where((theta < -90) & (theta >= -100))] = 27
        quantizedDir[np.where((theta < -100) & (theta >= -110))] = 28
        quantizedDir[np.where((theta < -110) & (theta >= -120))] = 29
        quantizedDir[np.where((theta < -120) & (theta >= -130))] = 30
        quantizedDir[np.where((theta < -130) & (theta >= -140))] = 31
        quantizedDir[np.where((theta < -140) & (theta >= -150))] = 32
        quantizedDir[np.where((theta < -150) & (theta >= -160))] = 33
        quantizedDir[np.where((theta < -160) & (theta >= -170))] = 34
        quantizedDir[np.where((theta < -170) & (theta >= -180))] = 35

        return quantizedDir

if __name__ == "__main__" :

    t1 = process_time()
    """STEP 1 : Loading an image"""
    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE) # shape order Y(height), X(width)
    img = cv2.resize(img, (250, 200)) # cv2.resize input shape order X(width), Y(height)

    """STEP 2 : Extracting key point from the image"""
    # STEP 2-1 : build scale space from the image
    scaleSpace, sigmas = extract_feature.scaleSpace(img=img,
                                                    s=3,
                                                    octaveNum=5)
    # STEP 2-2 : create Difference of Gaussian (DOG) space from the scale space
    DoG = extract_feature.dog(scaleSpace=scaleSpace)

    # STEP 2-3 : get naive extremums from the DoG space
    naiveExtremum = extract_feature.extractExtremum(dogSpace=DoG)

    # STEP 2-4 : interpolate the extremum and remove what has low constrast
    localizedExtremum = extract_feature.localization(dogSpace=DoG,
                                                     extremum=naiveExtremum,
                                                     offsetThr=0.5,
                                                     contrastThr=0.03)
    # STEP 2-5 : remove features on the edge
    features = extract_feature.edgeRemover(dogSpace=DoG,
                                           extremum=localizedExtremum,
                                           sigmaY=1.5,
                                           sigmaX=1.5,
                                           r=10)

    t2 = process_time()
    print("Process time of Chapter 4 : ", t2 - t1)

    """STEP 3 : making descriptor"""
    oriFeatures = orientation.assign(dogSpace=DoG, sigmas=sigmas, features=features)

    t2 = process_time()
    print("Process time of Chapter 5 : ", t2 - t1)