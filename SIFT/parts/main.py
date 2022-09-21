import cv2
from featureExtractor import *
from descriptor import *
from matching import *


t1 = process_time()
"""STEP 1 : Loading an image"""
imgDir = "D:\\cv\\data\\prac\\cannytest.png"
img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)  # shape order Y(height), X(width)
img = cv2.resize(img, (250, 200))  # cv2.resize input shape order X(width), Y(height)

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
print("Process time from Chapter 3 to Chapter 5 : ", t2 - t1)

"""STEP 4 : matching feature point"""