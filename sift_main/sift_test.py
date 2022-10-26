import cv2
import numpy as np
import numba as nb
from numba import jit, njit, int16, int64, float64
from basic.filters import *
from feature_extractor import *
from localization import *

if __name__ == "__main__" :

    testDir = "D:\\cv\\data\\prac\\bcard.jpg"
    img = cv2.imread(testDir, cv2.IMREAD_GRAYSCALE).astype(float) # not normalized
    s = 3

    space, sigmas = scaleSpace(img=img,
                               s=s,
                               octaveNum=5)
    dogSpace = dog(scaleSpace=space)
    extremaLocation = extremaDetection(dogSpace=dogSpace,
                                       s=s)
    extremasLocalized = localize(dogSpace=dogSpace,
                                 interval_num=s,
                                 extremaLocation=extremaLocation)
    print(extremasLocalized)