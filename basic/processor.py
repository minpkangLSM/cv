import os
import cv2
import PIL
import numpy as np
from matplotlib import pyplot as plt

os.path.join("..data\prac")

def histogram(file_dir,
              read_mode=1,
              resize_scale=0.5,
              bins=10,
              mode=1):

    # BGR to RGB / resizing image
    img = cv2.imread(file_dir, flags=read_mode)
    if read_mode!=0 : img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_scale : img = cv2.resize(img, dsize=(0,0), fx=resize_scale, fy=resize_scale)

    # visualizing image
    if read_mode==0 : img = np.expand_dims(img, axis=-1)
    channel = img.shape[2]
    plt.subplot(int('1' + str(channel + 1) + '1'))
    if read_mode==0 : plt.imshow(img, cmap='gray')
    else : plt.imshow(img)

    # histogram
    step = 255. / bins
    for idx in range(channel):
        grayscale = img[:,:,idx].flatten()
        hist_dict = {}
        min = 0
        max = 0
        for i in range(bins):
            if i > 0 : min = max
            max = min + step
            mid = round((min+max)/2.)
            count_mask = np.where((grayscale >= min) & (grayscale < max))
            hist_dict[mid] = len(grayscale[count_mask])
            plt.subplot(int('1' + str(channel + 1) + str(idx+2)))
            plt.bar(list(hist_dict.keys()), list(hist_dict.values()), width=step*0.8)

    if mode==1 : plt.show()
    else : return hist_dict

def hist_equalization(file_dir,
                      resize_scale,
                      bins):
    # BGR to RGB / resizing image
    img = cv2.imread(file_dir, flags=0)
    if resize_scale: img = cv2.resize(img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

    # visualizing image
    img = np.expand_dims(img, axis=-1)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')

    # histogram equaliziation
    step = 255. / bins
    min = 0
    max = 0
    sum = 0
    img_equalized = np.zeros_like(img)
    for i in range(bins):
        if i > 0: min = max
        max = min + step
        count_mask = np.where((img >= min) & (img < max))
        sum += len(img[count_mask])
        ratio = sum / (img.shape[0]*img.shape[1])
        intensity = 255 * ratio
        img_equalized[count_mask] = intensity
    plt.subplot(122)
    plt.imshow(img_equalized, cmap='gray')
    plt.show()



if __name__ == "__main__" :

    file_dir = "D:\\cv\\data\\prac\\KakaoTalk_20220518_215457616_01.jpg"
    hist_equalization(file_dir=file_dir,
                      resize_scale=0.5,
                      bins=50)