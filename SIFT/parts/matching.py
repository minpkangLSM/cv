import cv2
from time import process_time
import numpy as np
import numba as nb
from numba import jit, int64, float32, float64

class Node:
    def __init__(self,
                 value=None,
                 dimension=None,
                 leftNode=None,
                 rightNode=None):

        self.val = value
        self.dim = dimension
        self.left = leftNode
        self.right = rightNode

class KdTree :

    @staticmethod
    def makeTree(vectors):

        if vectors.size == 0 :
            node = Node(value=None)
            return node

        elif vectors.shape[0]==1 :
            node = Node(value=vectors[0,:])
            return node

        else :
            # 분산이 가장 큰 차원 찾기
            maxDim = KdTree.findMaxDim(vectors=vectors)

            # 해당 분산을 갖는 축을 기준으로 vectors 정렬
            vectors, maxDim = KdTree.mergeSort(vectors=vectors,
                                               maxDim=maxDim)
            # median에 위치하는 벡터 찾기 / median 기준 left, right 분할
            medianVector = vectors[vectors.shape[0]//2, :]
            left = vectors[:vectors.shape[0]//2]
            right = vectors[vectors.shape[0]//2+1:]

            node = Node(value=medianVector,
                        dimension=maxDim)
            node.left = KdTree.makeTree(left)
            node.right = KdTree.makeTree(right)
            return node

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
                  maxDim):
        """
        :param dimVec: def findMaxDim을 통해 추출된 가장 큰 분산을 갖는 벡터의 차원 값들
        :return:
        """

        if vectors.shape[0]==1 : return vectors, maxDim
        mid = vectors.shape[0] // 2
        leftVectors, maxDim = KdTree.mergeSort(vectors=np.copy(vectors[:mid, :]),
                                               maxDim=maxDim)
        rightVectors, maxDim = KdTree.mergeSort(vectors=np.copy(vectors[mid:, :]),
                                                maxDim=maxDim)
        i = 0
        j = 0
        k = 0
        while i < leftVectors.shape[0] and j < rightVectors.shape[0]:
            if leftVectors[i, maxDim] < rightVectors[j, maxDim]:
                vectors[k, :] = leftVectors[i, :]
                k += 1
                i += 1
            else :
                vectors[k, :] = rightVectors[j, :]
                k += 1
                j += 1

        if i == leftVectors.shape[0] :
            while j < rightVectors.shape[0]:
                vectors[k, :] = rightVectors[j, :]
                k += 1
                j += 1
        elif j == rightVectors.shape[0] :
            while i < leftVectors.shape[0]:
                vectors[k, :] = leftVectors[i, :]
                k += 1
                i += 1

        return vectors, maxDim

class matching :

    nearestNode = None
    nearestDistance = np.inf

    @staticmethod
    def isLeaf(node):
        return (node.left == None and node.right == None)

    @staticmethod
    def findNearest(kdTree,
                     target):
        s = [] # list for stack
        root = kdTree
        while not matching.isLeaf(root) :
            if target[root.dim] < root.val[root.dim] :
                if root.left.val is None :
                    s.append((root, "left"))
                    root = root.right
                else :
                    s.append((root, "right"))
                    root = root.left
            else :
                if root.right.val is None :
                    s.append((root, "left"))
                    root = root.left
                else :
                    s.append((root, "right"))
                    root = root.right

        distance = np.linalg.norm(root.val-target)
        if distance < matching.nearestDistance :
            matching.nearestNode = root
            matching.nearestDistance = distance

        while len(s) != 0 :
            (node, direction) = s.pop()
            distance = np.sqrt(((node.val-target)*(node.val-target)).sum())
            if distance < matching.nearestDistance :
                matching.nearestNode = node
                matching.nearestDistance = distance
            boundaryDistance = np.abs(node.val[node.dim]-target[node.dim])
            if boundaryDistance < matching.nearestDistance :
                if direction == "left":
                    matching.findNearest(kdTree=node.left,
                                         target=target)
                elif direction == "right":
                    matching.findNearest(kdTree=node.right,
                                         target=target)

if __name__ == "__main__":

    t1 = process_time()
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
    target = np.array([7,5.5])
    kdTree = KdTree.makeTree(vectors=arr)

    # 공통변수때문에 인스턴스를 선언하고, findNearest를 수행해야함 (따로
    a = matching()
    a.findNearest(kdTree=kdTree,
                   target=target)

    print("NEAREST NODE : ", a.nearestNode.val)
    print("DISTANCE : ", a.nearestDistance)

    t2 = process_time()
    print(t2-t1)


