import cv2
import numpy as np
import numba as nb
from numba import jit, int64, float32, float64

class kdTree :

    class node :
        def __init__(self,
                     dimension,
                     leftGroup,
                     rightGroup):
            dim = dimension
            left = leftGroup
            right = rightGroup

    def __init__(self):
        self.length = 0
        node = kdTree.node(dimension=None,
                           leftGroup=None,
                           rightGroup=None)

    def findMaxDim(self,
               vectors):
        """
        :param vectors: N x D 차원의 행렬로 가정
        N : 벡터의 개수
        D : 벡터의 차원
        :return:
        """
        distribution = vectors.var(axis=-1)
        maxDim = distribution.argmax()
        return maxDim
