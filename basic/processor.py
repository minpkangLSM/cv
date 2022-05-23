import os
import cv2
import PIL
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import *
os.path.join("..data\prac")

def binary_image(file_dir,
                 thr):
    """
    convert the input into binary (0 or 1) image format using thr(threshold).
    :param file_dir:
    :param thr:
    :return:
    """
    img = cv2.imread(file_dir, flags=cv2.IMREAD_GRAYSCALE)
    mask_0 = img>=thr
    mask_1 = img<thr
    img[mask_0] = 0
    img[mask_1] = 1
    img = img.astype(np.float16)
    return img

def histogram(file_dir,
              read_mode=1,
              resize_scale=0.5,
              bins=10,
              mode=1,
              norm=True):
    """
    :param file_dir: image directory
    :param read_mode: bgr(1), binary(0), bgr-alpha(-1) :: cv2 flags
    :param resize_scale: 0~1
    :param bins: histogram intervals
    :param mode: 0 - imshow image, 1 - return histogram dictionary data
    :param norm: normalize histogram or not
    :return:
    """

    # BGR to RGB / resizing images
    img = cv2.imread(file_dir, flags=read_mode)
    if read_mode!=0 : img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_scale : img = cv2.resize(img, dsize=(0,0), fx=resize_scale, fy=resize_scale)

    if read_mode==0 : img = np.expand_dims(img, axis=-1) # gray mode(0) dim = 2
    channel = img.shape[2]
    if mode==0 :
        plt.subplot(int('1' + str(channel + 1) + '1'))
        plt.imshow(img, cmap='gray')
    else :
        plt.imshow(img)

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
            if norm==True :
                if max==255 : hist_dict[mid] = (len(grayscale[count_mask]) + sum(grayscale==255))/img.size
                else : hist_dict[mid] = (len(grayscale[count_mask]))/img.size
            else :
                if max==255 : hist_dict[mid] = len(grayscale[count_mask]) + sum(grayscale==255)
                else : hist_dict[mid] = len(grayscale[count_mask])
            if mode==1:
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

def otsu_binary(file_dir,
                resize_scale=0.8):
    """
    Otsu binary
    calculate the threshold making the sum of the both of variations minimum.
    :param file_dir:
    :param resize_scale:
    :return: threshold making the variation minimum
    """
    hist_dict = histogram(file_dir=file_dir,
                          read_mode=0,
                          resize_scale=resize_scale,
                          mode=0,
                          norm=True)
    # initialize
    m = 0
    for key, value in hist_dict.items():
        m += key*value
    w0_init = 0
    m0_init = 0
    m1_init = m
    if hist_dict.get(0) != None : w0_init = sum(hist_dict.get(0))

    v = {}
    v[0] = w0_init*(1-w0_init)*(m0_init-m1_init)**2

    for i in range(1, 255+1):
        if hist_dict.get(i) == None : h_i = 0
        else : h_i = hist_dict.get(i)
        w0 = w0_init+h_i
        m0 = (w0_init*m0_init+i*h_i) / (w0+1e-6)
        if w0==1 : break
        m1 = (1*m-w0*m0) / (1-w0)
        v[i] = w0*(1-w0)*(m0-m1)**2
        w0_init = w0
        m0_init = m0

    return max(v, key=v.get)

def flood_fill(img,
               mode=4):
    """
    flood_fill : bfs version
    img : binary map (0 or -1) -> 0 : no target / -1 : target (not clustered)
    mode : 4 - 4 connection / 8 - 8 connection
    ==========================================
    [4 connection]          [8 connection]
        |                      \ | /
      - 4 -                    - 8 -
        |                      / | \
    ==========================================
    :return: clustered image
    """
    # Test image
    # img = np.array([[0, 0, 0,  0,  0, -1],
    #                 [0, 0, 0, -1, -1, -1],
    #                 [0, 0, -1, 0,  0,  0],
    #                 [-1, -1, -1, 0, 0, 0],
    #                 [-1,  0,  0, 0, 0, 0],
    #                 [0, 0, 0, -1, -1, -1]])

    if mode==4:
        i_step = [1, -1, 0, 0]
        j_step = [0, 0, -1, 1]
    elif mode==8:
        i_step = [1, -1, 0, 0, 1, 1, -1, -1]
        j_step = [0, 0, -1, 1, -1, 1, -1, 1]
    else :
        raise Exception("Only 2 modes : 4 or 8")

    def bfs(i, j):
        q = queue()
        q.enqueue([i,j])
        img[i][j] = grouping
        while len(q.q_list)!=0:
            for _ in range(len(q.q_list)):
                coord = q.dequeue()
                for sub_i in range(mode):
                    new_i = coord[0]+i_step[sub_i]
                    new_j = coord[1]+j_step[sub_i]
                    if new_i>=0 and new_i < img.shape[0] and new_j>=0 and new_j < img.shape[1]:
                        if img[new_i][new_j]==-1:
                            q.enqueue([new_i, new_j])
                            img[new_i][new_j] = grouping

    img_shape = img.shape
    grouping = 1
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i][j] == -1:
                bfs(i,j)
                grouping+=1
    return img

def




























































if __name__ == "__main__" :

    file_dir = "D:\\cv\\data\\prac\\KakaoTalk_20220518_215457616_01.jpg"
    thr = otsu_binary(file_dir,resize_scale=0.9)
    img = binary_image(file_dir, thr)
    img[img==1] = -1
    binary = flood_fill(img=img,
                        mode=4)
    binary = binary.astype(np.int8)
    plt.imshow(binary, cmap="gray")
    plt.show()