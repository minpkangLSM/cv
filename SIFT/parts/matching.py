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

    @staticmethod
    def main(vectors):

        # 분산이 가장 큰 차원 찾기
        maxDim = kdTree.findMaxDim(vectors=vectors)

        # 해당 분산을 갖는 축을 기준으로 vectors 정렬
        dimVec = vectors[:, maxDim]
        print(dimVec.shape)
        vectors, dimVec = kdTree.mergeSort(vectors=vectors, dimVec=dimVec)
        print(dimVec)
        # 가장 분산이 큰 차원의 축을 기준으로 정렬된 vectors에 대하여 해당 축에서 median에 위치하는 벡터 찾기
        medianVector = vectors[vectors.shape[0]//2, :]
        print(medianVector)
        left = vectors[:vectors.shape[0]//2] # 해당 벡터에 대해서도 재귀
        right = vectors[vectors.shape[0]//2:] # 해당 벡터에 대해서도 재귀

    @staticmethod
    def findMaxDim(vectors):
        """
        :param vectors: N x D 차원의 행렬로 가정
        N : 벡터의 개수
        D : 벡터의 차원
        :return:
        """
        distribution = vectors.var(axis=0)
        maxDim = distribution.argmax()
        return maxDim

    @staticmethod
    def mergeSort(vectors,
                  dimVec):
        """
        :param dimVec: def findMaxDim을 통해 추출된 가장 큰 분산을 갖는 벡터의 차원 값들
        :return:
        """
        if dimVec.shape[0]==1 : return vectors, dimVec
        mid = dimVec.shape[0] // 2
        leftvectors, leftdimVec = kdTree.mergeSort(vectors=vectors[:mid, :], dimVec=dimVec[:mid])
        rightvectors, rightdimVec = kdTree.mergeSort(vectors=vectors[mid:, :], dimVec=dimVec[mid:])
        i = 0
        j = 0
        k = 0
        while i < leftdimVec.shape[0] and j < rightdimVec.shape[0] :
            if leftdimVec[i] < rightdimVec[j] :
                vectors[k, :] = leftvectors[i, :]
                dimVec[k] = leftdimVec[i]
                k += 1
                i += 1
            else :
                vectors[k, :] = rightvectors[j, :]
                dimVec[k] = rightdimVec[j]
                k += 1
                j += 1

        if i == leftdimVec.shape[0] :
            while j < rightdimVec.shape[0]:
                vectors[k, :] = rightvectors[j, :]
                dimVec[k] = rightdimVec[j]
                k += 1
                j += 1
        elif j == rightdimVec.shape[0] :
            while i < leftdimVec.shape[0]:
                vectors[k, :] = leftvectors[i, :]
                dimVec[k] = leftdimVec[i]
                k += 1
                i += 1

        return vectors, dimVec





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

    tree = kdTree.main(vectors=arr)
