"""
3. Detection of Scale-Space Extrema, Lowe(2004)
- SCALE SPACE, Kang min Park, 2022.08.09.
"""
import cv2
import numpy as np
from basic.filters import *
from time import process_time
from matplotlib import pyplot as plt

class feature :

    @staticmethod
    def scaleSpace(img,
                     s,
                     octaveNum,
                     sigma=1.6):
       """
       the number of images in the stack of blurred images for each octave : s+3
       :param img: image type = gray scale
       :param s: a parameter for constant factor k = 2^(1/s)
                 s는 이전 옥타브 내 s번째 이미지를 다음 옥타브의 첫 번째 이미지로 삼겠다는 의미가 된다.
       :param octaveNum: the number of octaves
       :param sigma : gaussian blurr parameter
       :return:
       """
       # set parameters of scale space
       k = 2**(1/s) # gFactor = k, sigma factor
       octaveLayer = img
       scaleSpace = {}

       # create scale space
       for oIdx in range(octaveNum):
           initSigma = sigma

           # create each octave
           for iIdx in range(s+3):

               if iIdx==0 and oIdx!=0:
                   octave = octaveLayer[:,:,np.newaxis]
                   continue

               # set sigma factor
               initSigma = initSigma * (k ** iIdx)
               if iIdx==0 and oIdx==0 : sigmaDiff = np.sqrt(initSigma**2 - 0.5**2)
               else : sigmaDiff = np.sqrt(initSigma**2 - prevSigma**2)

               # make a layer
               octaveLayer = gaussian(octaveLayer,
                                      sigmaX=sigmaDiff,
                                      sigmaY=sigmaDiff)[:,:,np.newaxis]
               # make octave
               if iIdx==0 : octave = octaveLayer
               else : octave = np.concatenate([octave, octaveLayer], axis=-1)

               prevSigma = initSigma

           scaleSpace[oIdx] = octave
           prevSigma = sigma
           octaveLayer = cv2.resize(src=octave[:,:,s],
                                    dsize=None,
                                    fy=0.5,
                                    fx=0.5,
                                    interpolation=cv2.INTER_AREA)
       return scaleSpace

    @staticmethod
    def dog(scaleSpace):
        """
        order) feature class scale space -> dog
        create difference of gaussian (DoG) space from scale space of feature.scalespace function.
        :param scaleSpace:
        :return: dog space, dtype = dictionary
        """
        # setting
        dogSpace = {}
        octaves = scaleSpace.keys()

        for idx in octaves :
            octave = scaleSpace[idx]
            floor = octave.shape[2]
            for sub_idx in range(0, floor-1):
                l1 = octave[:,:,sub_idx]
                l2 = octave[:,:,sub_idx+1]
                diff = l1-l2
                diff = diff[:,:,np.newaxis]
                if sub_idx == 0 : dogOctave = diff
                else : dogOctave = np.concatenate([dogOctave, diff], axis=-1)

            dogSpace[idx] = dogOctave

        return dogSpace

    @staticmethod
    def extremum(dogSpace):
        """
        order) scale space -> dog -> extremum
        select the candidate points from scale space from feature.scaleSpace function.
        :param scaleSpace:
        :return:
        """
        dogIdx = dogSpace.keys()
        mask = []
        for key in dogIdx :
            boolBox = np.ones_like(dogSpace[key])
            for x in range(1, dogSpace[key].shape[0]-1):
                for y in range(1, dogSpace[key].shape[1]-1):
                    for z in range(1, dogSpace[key].shape[2]-1):
                        if boolBox[x,y,z] == 0 : continue
                        dVal = dogSpace[key][x,y,z]
                        upper = dogSpace[key][x-1:x+2, y-1:y+2, z-1]
                        mid = dogSpace[key][x-1:x+2, y-1:y+2, z]
                        bottom = dogSpace[key][x-1:x+2, y-1:y+2, z+1]

                        results = np.sum((dVal>=upper).flatten()) + np.sum((dVal>=mid).flatten()) + np.sum((dVal>=bottom).flatten())
                        print(results)
                        if results==0 or results==9 :
                            print(dVal)
                            print(upper)
                            print(mid)
                            print(bottom)
                            print("extruemum")
            break



if __name__ == "__main__":

    imgDir = "D:\\cv\\data\\prac\\cannytest.png"
    img = cv2.imread(imgDir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (350, 350))

    t1 = process_time()

    ss = feature.scaleSpace(img=img,
                            s=3,
                            octaveNum=5)
    ds = feature.dog(ss)
    ex = feature.extremum(ds)
    t2 = process_time()
    print("Process time : ", t2 - t1)

    # for key in ds.keys() :
    #     plt.subplot(231)
    #     plt.imshow(ds[key][:, :, 0], cmap='gray')
    #     plt.subplot(232)
    #     plt.imshow(ds[key][:, :, 1], cmap='gray')
    #     plt.subplot(233)
    #     plt.imshow(ds[key][:, :, 2], cmap='gray')
    #     plt.subplot(234)
    #     plt.imshow(ds[key][:, :, 3], cmap='gray')
    #     plt.subplot(235)
    #     plt.imshow(ds[key][:, :, 4], cmap='gray')
    #     plt.show()
        