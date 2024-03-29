"""
SIFT IMPLEMENTATION
from "Distinctive Image Features from Scale-Invariant Keypoints", Lowe, 2004.
Kangmin Park, 2022.10.20.
"""
import cv2
import numpy as np
import numba as nb
from numba import jit, njit, int16, int64, float64
from basic.filters import *
from feature_extractor import *
from localization import *
"""
Chapter 5.
Orientation Assigment
"""

def oriAssign(extremasLocalized,
              scaleFactor=1.5):

    # scale = scaleFactor * localized 키포인트의 scale / (2**(옥타브+1)) -> 원래 옥타브에 적용됐던 스케일 값
    # -> localized 키포인트의 scale 추출값은 옥타브까지 고려한 scale 값
    # 키포인트에 대해서 한 번에 히스토그램을 그리는게 아니라 키포인트 주변 일점 범위 내 존재하는 모든 점들에 대해서 하나씩 히스토를 측정한다?
