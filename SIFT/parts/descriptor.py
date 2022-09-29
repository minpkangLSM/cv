import cv2
import math
import numpy as np
from numba import jit, int64, float32, float64
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

            # step 1 : 키포인트의 scale에 해당하는 이미지 L에 대한 기울기와
            # 기울기 크기 계산을 위해 해당 옥타브 전체 계산
            subDog = dogSpace[idx]
            dy = sobelHeightAxis(subDog,
                                 ddepth=cv2.CV_64F)
            dx = sobelWidthAxis(subDog,
                                ddepth=cv2.CV_64F)

            # calculate magnitude and renormalize
            magnitude = np.sqrt(dy*dy+dx*dx)
            magnitude[magnitude>=0.2] = 0.2 # Limit the maxium of value - Lowe, 2004, chapter 6.1 (p.102)
            magnitude = magnitude/0.2 # Renormalize to unit length - Lowe, 2004, chapter 6.1 (p.102)

            theta = np.arctan2(dy,dx)*180/np.pi # rad to deg
            theta = orientation.__quantize36(theta)

            # step 2 : assign orientation to features
            locYs = features[idx][0].astype(np.int16)
            locXs = features[idx][1].astype(np.int16)
            locZs = features[idx][2].astype(np.int16)
            sigma = np.array(sigmas[idx])

            newFeatures = orientation.__orientationHist(locYs=locYs,
                                                        locXs=locXs,
                                                        locZs=locZs,
                                                        sigmaList=sigma,
                                                        theta=theta,
                                                        mag=magnitude)

            if newFeatures.size != 0 :
                oriFeatures[idx] = (newFeatures[:,0],
                                    newFeatures[:,1],
                                    newFeatures[:,2],
                                    newFeatures[:,3])
            else : # extremum이 없어서 비어있는 경우에는 히스토그램 생성이 불가하므로 empty 값으로 넣어준다.
                oriFeatures[idx] = (np.array([]),
                                    np.array([]),
                                    np.array([]),
                                    np.array([]))
        return oriFeatures

    @staticmethod
    @jit (int16[:,:](int16[:], int16[:], int16[:], float64[:], float64[:,:,:], float64[:,:,:]))
    def __orientationHist(locYs,
                          locXs,
                          locZs,
                          sigmaList,
                          theta,
                          mag):
        count = 0
        for locY, locX, locZ in zip(locYs, locXs, locZs) :

            # set feature arrays
            histogram = np.zeros(36).astype(np.float64)  # index = quantized orientation

            # get L(y, x) for each key point
            sigma = sigmaList[locZ]
            LMag = mag[:, :, locZ]
            LOri = theta[:, :, locZ]

            # set the range of histogram
            rangeYHead = int(max(0, locY - sigma / 2))
            rangeYRear = int(min(LMag.shape[1], locY + sigma / 2))
            rangeXHead = int(max(0, locX-sigma/2))
            rangeXRear = int(min(LMag.shape[1], locX + sigma / 2))
            magSur = LMag[rangeYHead:rangeYRear, rangeXHead:rangeXRear]
            oriSur = LOri[rangeYHead:rangeYRear, rangeXHead:rangeXRear]
            magShape = magSur.shape
            maxShape = np.array(max(magShape[0], magShape[1])).astype(np.float64)

            # get gaussian filter (=gWeight)
            m, n = [(ss - 1.) / 2. for ss in magShape]
            y = np.arange(-m, m+1).reshape(-1, 1)
            x = np.arange(-n, n+1).reshape(1, -1)
            gWeight = np.exp(-(x * x + y * y) / (2. * maxShape/6. * maxShape/6.))
            # gWeight[gWeight < np.finfo(gWeight.dtype).eps * gWeight.max()] = 0
            sumh = gWeight.sum()
            if sumh != 0 : gWeight /= sumh

            # make histogram
            weightedMagSur = magSur*gWeight
            for j in range(oriSur.shape[0]):
                for i in range(oriSur.shape[1]):
                    q = oriSur[j,i]
                    v = weightedMagSur[j,i]
                    histogram[int(q)] += v

            # histogram
            fIdx = np.where(histogram >= histogram.max()*0.8)
            loc = np.repeat(np.array([locY, locX, locZ]), len(fIdx[0]))
            if len(fIdx[0])>=2 : arr = loc.reshape(-1, len(fIdx[0]))
            else : arr = loc.reshape(-1,1)
            fIdxTmp = fIdx[0].reshape(1,-1)

            # assign orientation to feature array
            if count == 0 :
                arr2 = np.concatenate((arr, fIdxTmp), axis=0).transpose(1, 0)
            else :
                arr2 = np.concatenate((arr2, np.concatenate((arr, fIdxTmp), axis=0).transpose(1, 0)), axis=0)
            count+=1

        return arr2.astype(np.int16)

    @staticmethod
    def __quantize36(theta):
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
        quantizedDir[np.where((theta >= 30) & (theta < 40))] = 3
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

    @staticmethod
    def featureVector(oriFeatures,
                      dogSpace):

        octaveIdx = oriFeatures.keys()

        for idx in octaveIdx :

            # 기울기(orientation) / 크기(magnitude) 계산을 위해 해당 1개의 옥타브 전체 계산
            subDog = dogSpace[idx]
            dy = sobelHeightAxis(subDog,
                                 ddepth=cv2.CV_64F)
            dx = sobelWidthAxis(subDog,
                                ddepth=cv2.CV_64F)

            #calculate magnitude and renormalize
            magnitude = np.sqrt(dy * dy + dx * dx)
            magnitude[magnitude >= 0.2] = 0.2  # Limit the maxium of value - Lowe, 2004, chapter 6.1 (p.102)
            magnitude = magnitude / 0.2  # Renormalize to unit length - Lowe, 2004, chapter 6.1 (p.102)

            theta = np.arctan2(dy,dx)*180/np.pi # rad to deg
            theta = orientation.__quantize8(theta)

            # 1개의 옥타브 내 존재하는 배열 순서
            """
            arr[0] : 피쳐의 Y 좌표
            arr[1] : 피쳐의 X 좌표
            arr[2] : 피쳐의 Z 좌표
            arr[3] : 피쳐의 양자화 된 방향(Orientation) 
            """
            locYs = oriFeatures[idx][0].astype(np.int16)
            locXs = oriFeatures[idx][1].astype(np.int16)
            locZs = oriFeatures[idx][2].astype(np.int16)
            locOs = oriFeatures[idx][3].astype(np.int16)

            newFeatures = orientation.__orientationHist2(locYs=locYs,
                                                         locXs=locXs,
                                                         locZs=locZs,
                                                         locOs=locOs,
                                                         ori=theta,
                                                         mag=magnitude)
            # normalize feature vector to unit length - 2004, Lowe, chapter 6.1 (p.101)
            max = newFeatures.max(axis=-1).reshape(-1,1)
            newFeatures[:, :128] = newFeatures[:, :128]/max
            if idx==0 : features = newFeatures
            else : features = np.concatenate((features, newFeatures), axis=0)

        return features

    @staticmethod
    @jit (float32[:,:](int16[:], int16[:], int16[:], int16[:], float64[:,:,:], float64[:,:,:]))
    def __orientationHist2(locYs,
                           locXs,
                           locZs,
                           locOs,
                           ori,
                           mag):
        # 128개의 방향 피쳐(finger print of feature)를 저장할 배열생성 (128개 + Y,X,Z,O)
        idx = 0
        fVec = np.zeros((len(locYs), 132)).astype(np.float32)

        for locY, locX, locZ, locO in zip(locYs, locXs, locZs, locOs):

            ff = np.zeros(132).astype(np.float32)

            # get L for each key point
            LMag = mag[:, :, int(locZ)]
            LOri = ori[:, :, int(locZ)]

            # set the range of histogram
            rangeYHead = int(max(0, locY - 8))
            rangeYRear = int(min(LMag.shape[0], locY + 8))
            rangeXHead = int(max(0, locX - 8))
            rangeXRear = int(min(LMag.shape[1], locX + 8))
            magSur = LMag[rangeYHead:rangeYRear, rangeXHead:rangeXRear]
            oriSur = LOri[rangeYHead:rangeYRear, rangeXHead:rangeXRear]

            magShape = magSur.shape
            maxShape = max(magShape[0], magShape[1])

            # get gaussian filter (=gWeight)
            m, n = [(ss - 1.) / 2. for ss in magShape]
            y = np.arange(-m, m + 1).reshape(-1, 1)
            x = np.arange(-n, n + 1).reshape(1, -1)
            gWeight = np.exp(-(x * x + y * y) / (2. * maxShape / 6. * maxShape / 6.))
            # gWeight[gWeight < np.finfo(gWeight.dtype).eps * gWeight.max()] = 0
            sumh = gWeight.sum()
            if sumh != 0: gWeight /= sumh
            weightedMagSur = magSur * gWeight

            cnt = 0
            for idxY in range(0, magShape[0], 4):
                 for idxX in range(0, magShape[1], 4):
                     idxYHead = idxY
                     idxYRear = min(idxY + 4, magShape[0])
                     idxXHead = idxX
                     idxXRear = min(idxX + 4, magShape[1])

                     magPart = weightedMagSur[idxYHead:idxYRear, idxXHead:idxXRear].flatten()
                     oriPart = oriSur[idxYHead:idxYRear, idxXHead:idxXRear].flatten()

                     baseIdx = cnt*8 + int(idxX/4)*8

                     for qOri in range(0,8):
                         target = np.where(oriPart==qOri)
                         if len(target)==0 : continue
                         arrIdx = baseIdx + qOri
                         ff[arrIdx] = magPart[target].sum()

                 cnt += 4

            ff[-1] = locO
            ff[-2] = locZ
            ff[-3] = locX
            ff[-4] = locY
            fVec[idx, :] = ff
            idx+=1

        return fVec

    @staticmethod
    def __quantize8(theta):
        """
        sub function of def. assign
        quantize theta(Deg) into 8 bins(0~7)
        :param theta: unit -> deg
        :return:
        """
        quantizedDir = np.zeros_like(theta)
        quantizedDir[np.where((theta >= 0) & (theta < 45))] = 0
        quantizedDir[np.where((theta >= 45) & (theta < 90))] = 1
        quantizedDir[np.where((theta >= 90) & (theta < 135))] = 2
        quantizedDir[np.where((theta >= 135) & (theta < 180))] = 3
        quantizedDir[np.where((theta < 0) & (theta >= -45))] = 4
        quantizedDir[np.where((theta < -45) & (theta >= -90))] = 5
        quantizedDir[np.where((theta < -90) & (theta >= -135))] = 6
        quantizedDir[np.where((theta < -135) & (theta >= -180))] = 7

        return quantizedDir

if __name__ == "__main__" :

    t1 = process_time()
    """STEP 1 : Loading an image"""
    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE) # shape order Y(height), X(width)
    img = cv2.resize(img, (250, 400)) # cv2.resize input shape order X(width), Y(height)

    # STEP 1-2 : normalize image into 0 ~ 1
    img = img / 255.

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

    """STEP 3 : making descriptor"""
    oriFeatures = orientation.assign(dogSpace=DoG, sigmas=sigmas, features=features)
    featureVect = orientation.featureVector(oriFeatures=oriFeatures,
                                            dogSpace=DoG)

    t2 = process_time()
    print(featureVect.shape)
    print("Process time from Chapter 3 to Chapter 5 : ", t2 - t1)