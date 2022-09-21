import cv2
import numpy as np
import numba as nb
from numba import jit, int64, float32, float64

class node:

    def __init__(self,
                 value,
                 dimension,
                 leftGroup=None,
                 rightGroup=None):

        val = value
        dim = dimension
        left = leftGroup
        right = rightGroup

class kdTree :

    def __init__(self,
                 vectors):

        self.length = 0
        self.vectors = vectors

    def main(self):

        # 분산이 가장 큰 차원 찾기
        maxDim = kdTree.findMaxDim()

        # 해당 분산을 갖는 축을 기준으로 vectors 정렬
        kdTree.mergeSort(dim=maxDim)

        # 가장 분산이 큰 차원의 축을 기준으로 정렬된 vectors에 대하여 해당 축에서 median에 위치하는 벡터 찾기
        medianVector = self.vectors[self.vectors.shape[0]//2, :]
        left = self.vectors[:self.vectors.shape[0]//2] # 해당 벡터에 대해서도 재귀
        right = self.vectors[self.vectors.shape[0]//2:] # 해당 벡터에 대해서도 재귀

    def findMaxDim(self):
        """
        :param vectors: N x D 차원의 행렬로 가정
        N : 벡터의 개수
        D : 벡터의 차원
        :return:
        """
        distribution = self.vectors.var(axis=0)
        maxDim = distribution.argmax()
        return maxDim

    def mergeSort(self,
                  dim):
        """
        :param dim: def findMaxDim에서 가장 분산이 큰 축의 번호 (0~)
        :return:
        """
        sortTarget = self.vectors[:, dim]





if __name__ == "__main__":

    arr = np.array([[3,1],
                    [2,3],
                    [6,2],
                    [4,4],
                    [3,6],
                    [8,5],
                    [7,6.5],
                    [5,8],
                    [6,10],
                    [6,11]])

    tree = kdTree(vectors=arr)
    maxDim = tree.findMaxDim()
    tree.mergeSort(maxDim)
