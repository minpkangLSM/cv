import os
import cv2
from time import process_time
from PIL import Image
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

def sobel(img):
    f_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    f_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
    imgDx = cv2.filter2D(img, ddepth=cv2.CV_16S, kernel=f_x)
    imgDy = cv2.filter2D(img, ddepth=cv2.CV_16S, kernel=f_y)

    return imgDy, imgDx

def edgeDirection(Dy,
                  Dx,
                  Dyx=None,
                  mode=0):
    """
    -27.5 <= deg < 27.5 : 0
    27.5 <= deg < 72.5 : 1
    72.5 <= deg < 117.5 : 2
    117.5 <= deg < 162.5 : 3
    162.5 <= deg < 180 : 4
    -180 <= deg < -162.5 : 4
    -162.5 <= deg < -117.5 : 5
    -117.5 <= deg < -72.5 : 6
    -72.5 <= deg < -27.5 : 7
    90 <= deg < 135
    :param Dy: Dx calculated by sobel y
    :param Dx: Dy calculated by sobel x
    :param mode : 0 - default / 1 - de genzo
    :return:
    """
    if mode not in [0, 1] : raise ValueError("MODE has 2 values : 0 - default, 1 - di genzo mode.")
    # if (mode==1) & (Dyx==None) : raise ValueError("It is necessary for MODE=1(De genzo) to have Dyx value.")
    if mode==0 : edgeDirMapRaw = np.rad2deg(np.arctan2(Dy, Dx))
    elif mode==1 : edgeDirMapRaw = np.rad2deg(np.arctan2(2*Dyx, (Dx-Dy)))
    edgeDirMapQuant = np.zeros_like(edgeDirMapRaw)
    edgeDirMapQuant[np.where((edgeDirMapRaw >= -27.5) & (edgeDirMapRaw < 27.5))] = 0
    edgeDirMapQuant[np.where((edgeDirMapRaw >= 27.5) & (edgeDirMapRaw < 72.5))] = 1
    edgeDirMapQuant[np.where((edgeDirMapRaw >= 72.5) & (edgeDirMapRaw < 117.5))] = 2
    edgeDirMapQuant[np.where((edgeDirMapRaw >= 117.5) & (edgeDirMapRaw < 162.5))] = 3
    edgeDirMapQuant[np.where((edgeDirMapRaw >= 162.5) & (edgeDirMapRaw <= 180))] = 4
    edgeDirMapQuant[np.where((edgeDirMapRaw >= -180) & (edgeDirMapRaw < -162.5 ))] = 4
    edgeDirMapQuant[np.where((edgeDirMapRaw >= -162.5) & (edgeDirMapRaw < -117.5))] = 5
    edgeDirMapQuant[np.where((edgeDirMapRaw >= -117.5) & (edgeDirMapRaw < -72.5))] = 6
    edgeDirMapQuant[np.where((edgeDirMapRaw >= -72.5) & (edgeDirMapRaw < -27.5))] = 7

    return edgeDirMapRaw, edgeDirMapQuant

def NMS(edgeMap,
        edgeDirMapQuant,
        edge_dir = 8):
    """
    NMS : None Maximum Suppression
    :return:
    """
    edgeMap = np.pad(edgeMap, (1,1))

    edge_dict = {
        0 : {'x' : [+0, -0], 'y' : [+1, -1]},
        1 : {'x' : [+1, -1], 'y' : [-1, +1]},
        2 : {'x' : [+1, -1], 'y' : [+0, -0]},
        3 : {'x' : [+1, -1], 'y' : [+1, -1]},
        4 : {'x' : [+0, -0], 'y' : [-1, +1]},
        5 : {'x' : [+1, -1], 'y' : [+1, -1]},
        6 : {'x' : [+0, -0], 'y' : [-1, +1]},
        7 : {'x' : [+1, -1], 'y' : [+1, -1]},
    }

    for dir in range(edge_dir):
        dir_dict = edge_dict[dir]
        y = dir_dict['y']
        x = dir_dict['x']

        coord = np.where(edgeDirMapQuant==dir)
        coord1 = np.copy(coord)
        coord2 = np.copy(coord)

        # 비교대상 추출
        coord1 = (coord1[0]+y[0], coord1[1]+x[0])
        coord2 = (coord2[0]+y[1], coord2[1]+x[1])

        # extract target suppressed
        sup_target = ((edgeMap[coord] <= edgeMap[coord1]) | (edgeMap[coord] <= edgeMap[coord2]))
        coord_list = np.array([coord[0], coord[1]])
        idx_target = coord_list[:, sup_target]
        edgeMap[np.array(idx_target[0,:]), np.array(idx_target[1,:])] = 0

    edgeMap = edgeMap[1:edgeMap.shape[0]-1, 1:edgeMap.shape[1]-1]
    return edgeMap

def hys_thr(edgeMap,
            tLow,
            tHigh):

    edge = np.zeros_like(edgeMap)
    visited = np.zeros_like(edgeMap)
    shape = edge.shape

    q = queue()
    y_coord = [-1, +1, -1, +1, -1, +1, 0, 0]
    x_coord = [0, 0, -1, +1, +1, -1, -1, +1]
    q.enqueue([0,0])

    for yj in range(0, shape[0]-1):
        for xi in range(0, shape[1]-1):

            if visited[yj, xi] == 0:
                visited[yj, xi] = 1
                if edgeMap[yj, xi] > tHigh :
                    if yj!=0 and xi!=0 : q.enqueue([yj, xi])
                    edge[yj, xi] = 1

            while len(q.q_list) != 0:
                for _ in q.q_list :
                    yj, xi = q.dequeue()
                    for ym, xm in zip(y_coord, x_coord):
                        yNew = yj+ym
                        xNew = xi+xm
                        if yNew >= 0 and yNew < edge.shape[0] and xNew >= 0 and xNew < edge.shape[1] : # 영상의 범위 안에 들어오고
                            if visited[yNew, xNew] == 0:
                                visited[yNew, xNew] = 1
                                if edgeMap[yNew, xNew] > tLow :
                                    q.enqueue([yNew, xNew])
                                    edge[yNew, xNew] = 1

    return edge

def canny_edge(img,
               imread_mode,
               tLow,
               tHigh,
               resize_scale=0,
               ksize=3,
               sigmaX=1,
               sigmaY=1):

    t1 = process_time()
    # load image
    img = img

    # resize image
    if resize_scale: img = cv2.resize(img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

    # distinguish GRAY or RGB
    if len(img.shape)==2 :
        channels = 1 # gray scale
    elif len(img.shape)>=3 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
        channels = img.shape[2] # RGB or over than RGB channels

    # Gaussian Blurring
    imgBlurred = cv2.GaussianBlur(img,
                                  ksize=(ksize, ksize),
                                  sigmaX=sigmaX,
                                  sigmaY=sigmaY)
    if len(imgBlurred.shape)==2 : imgBlurred = np.expand_dims(imgBlurred, -1)

    img_frame = np.zeros_like(imgBlurred)

    for channel in range(channels) :

        gray_scale = imgBlurred[:, :, channel]
        # calculate edge-map
        Dy, Dx = sobel(gray_scale)
        edgeMap = np.sqrt(Dx*Dx + Dy*Dy).astype(np.float)
        raw, quant = edgeDirection(Dy, Dx)

        # None Maximum Suppression
        edgeMagnitude = NMS(edgeMap=edgeMap,
                            edgeDirMapQuant=quant)

        # hysteresis thresholding
        edge = hys_thr(edgeMap=edgeMagnitude,
                       tLow=tLow,
                       tHigh=tHigh)
        img_frame[:, :, channel] = edge
    t2 = process_time()
    print("CANNY PROCESSING TIME : {0}s".format(t2-t1))
    return img_frame

def canny_edge_genzo(img,
                     tLow,
                     tHigh,
                     resize_scale=0,
                     ksize=3,
                     sigmaX=3,
                     sigmaY=3,):

    t1 = process_time()
    # load image
    img = img
    # reszie the image
    if resize_scale : img = cv2.resize(img, dsize=(0, 0), fx=resize_scale, fy=resize_scale)

    # Gaussian blurring
    imgBlurred = cv2.GaussianBlur(img,
                                  ksize=(ksize, ksize),
                                  sigmaX=sigmaX,
                                  sigmaY=sigmaY)

    # edge intensity, direction
    Dry, Drx = sobel(imgBlurred[:,:,0]) # Red derivation
    Dgy, Dgx = sobel(imgBlurred[:,:,1]) # Green derivation
    Dby, Dbx = sobel(imgBlurred[:,:,2]) # Blue derivation

    gyy = Dry*Dry + Dgy*Dgy + Dby*Dby
    gxx = Drx*Drx + Dgx*Dgx + Dbx+Dbx
    gyx = Dry*Drx + Dgy*Dgx + Dby*Dbx

    raw, quant = edgeDirection(Dy=gyy,
                               Dx=gxx,
                               Dyx=gyx,
                               mode=1)
    edgeMagnitude = np.sqrt(0.5 * ((gyy + gxx) - (gxx - gyy) * np.cos(2 * raw) + 2 * gyx * np.sin(2 * raw)))

    # None Maximum Suppression
    edgeMap = NMS(edgeMap=edgeMagnitude,
                  edgeDirMapQuant=quant)

    # hysteresis thresholding
    edge = hys_thr(edgeMap=edgeMap,
                   tLow=tLow,
                   tHigh=tHigh)
    t2 = process_time()
    print(t2-t1)
    return edge

def SPTA(img):
    # reshape img 3 dims into 2 dim
    img = np.reshape(img, (img.shape[0], img.shape[1]))
    # pad image
    img = np.pad(img, (1,1))
    initImg = img
    count = 0
    t1 = process_time()
    while True :
        count+=1
        edgeIndex = np.where(initImg == 1)
        edgeBoolean = np.where(initImg == 1, True, False)

        n0 = edgeBoolean[(edgeIndex[0]    , edgeIndex[1] + 1)]
        n1 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1] + 1)]
        n2 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1]    )]
        n3 = edgeBoolean[(edgeIndex[0] + 1, edgeIndex[1] - 1)]
        n4 = edgeBoolean[(edgeIndex[0]    , edgeIndex[1] - 1)]
        n5 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1] - 1)]
        n6 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1]    )]
        n7 = edgeBoolean[(edgeIndex[0] - 1, edgeIndex[1] + 1)]

        mask0 = np.invert(n0) & (n4 & (n5 | n6 | n2 | n3) & (n6 | np.invert(n7)) & (n2 | np.invert(n1)))
        mask2 = np.invert(n4) & (n0 & (n1 | n2 | n6 | n7) & (n2 | np.invert(n3)) & (n6 | np.invert(n5)))
        mask4 = np.invert(n2) & (n6 & (n7 | n0 | n4 | n5) & (n0 | np.invert(n1)) & (n4 | np.invert(n3)))
        mask6 = np.invert(n6) & (n2 & (n3 | n4 | n0 | n1) & (n4 | np.invert(n5)) & (n0 | np.invert(n7)))
        merge = (mask0 | mask2 | mask4 | mask6)
        if not merge.any() :
            break
        noneEdge = (edgeIndex[0][merge], edgeIndex[1][merge])
        initImg[noneEdge] = 0

    # remove pad
    img = initImg[1:img.shape[0] - 1, 1:img.shape[1] - 1]
    t2 = process_time()
    print("SPTA PROCESSING TIME : {0}s".format(t2-t1))
    return img

"""
# deprecated version, 이유 : 병렬문으로 처리하는 것(def SPTA)과 동일한 결과를 보이나, 소요시간이 10배 이상 더 걸림
def SPTA_loop(img):
    # reshape img 3 dims into 2 dim
    img = np.reshape(img, (img.shape[0], img.shape[1]))
    # pad image
    img = np.pad(img, (1,1))
    init_img = img
    count = 0
    t1 = process_time()
    while True :
        count+=1
        edge_index = np.where(init_img == 1)
        edge_boolean = np.where(init_img == 1, True, False)
        loop = False

        for j, i in zip(edge_index[0], edge_index[1]):

            n0 = edge_boolean[j  , i+1]
            n1 = edge_boolean[j+1, i+1]
            n2 = edge_boolean[j+1, i  ]
            n3 = edge_boolean[j+1, i-1]
            n4 = edge_boolean[j  , i-1]
            n5 = edge_boolean[j-1, i-1]
            n6 = edge_boolean[j-1, i  ]
            n7 = edge_boolean[j-1, i+1]

            mask0 = np.invert(n0) & (n4 & (n5 | n6 | n2 | n3) & (n6 | np.invert(n7)) & (n2 | np.invert(n1)))
            mask2 = np.invert(n4) & (n0 & (n1 | n2 | n6 | n7) & (n2 | np.invert(n3)) & (n6 | np.invert(n5)))
            mask4 = np.invert(n2) & (n6 & (n7 | n0 | n4 | n5) & (n0 | np.invert(n1)) & (n4 | np.invert(n3)))
            mask6 = np.invert(n6) & (n2 & (n3 | n4 | n0 | n1) & (n4 | np.invert(n5)) & (n0 | np.invert(n7)))
            merge = (mask0 | mask2 | mask4 | mask6)

            if merge :
                loop = True
                init_img[j,i]=0

        if not loop : break
    print(count)
    img = init_img[1:img.shape[0] - 1, 1:img.shape[1] - 1]
    t2 = process_time()
    print(t2-t1)
    return img
"""

def edgeSegment(img):
    """
    :param img: 에지 이미지
    :return:
    """
    t1 = process_time()
    # find end or bifurcation point
    endBifDict = {}
    img = np.pad(img, (1,1))
    edgeIndex = np.where(img==1)
    n0_group = img[(edgeIndex[0]    , edgeIndex[1] + 1)]
    n1_group = img[(edgeIndex[0] + 1, edgeIndex[1] + 1)]
    n2_group = img[(edgeIndex[0] + 1, edgeIndex[1]    )]
    n3_group = img[(edgeIndex[0] + 1, edgeIndex[1] - 1)]
    n4_group = img[(edgeIndex[0]    , edgeIndex[1] - 1)]
    n5_group = img[(edgeIndex[0] - 1, edgeIndex[1] - 1)]
    n6_group = img[(edgeIndex[0] - 1, edgeIndex[1]    )]
    n7_group = img[(edgeIndex[0] - 1, edgeIndex[1] + 1)]

    idx = 0
    for n0, n1, n2, n3, n4, n5, n6, n7 in zip(n0_group, n1_group, n2_group, n3_group, n4_group, n5_group, n6_group, n7_group):
        count = 0
        dirList = []
        if n0 == 1 and n1 == 0 :
            count+=1
            dirList.append(0)
        if n1 == 1 and n2 == 0 :
            count+=1
            dirList.append(1)
        if n2 == 1 and n3 == 0 :
            count+=1
            dirList.append(2)
        if n3 == 1 and n4 == 0 :
            count+=1
            dirList.append(3)
        if n4 == 1 and n5 == 0 :
            count+=1
            dirList.append(4)
        if n5 == 1 and n6 == 0 :
            count+=1
            dirList.append(5)
        if n6 == 1 and n7 == 0 :
            count+=1
            dirList.append(6)
        if n7 == 1 and n0 == 0 :
            count+=1
            dirList.append(7)
        if count==1 or count>=3:
            endBifDict[(edgeIndex[0][idx], edgeIndex[1][idx])] = dirList
        idx += 1

    # start edge segmentation
    segId = 0
    visited = np.zeros_like(img)

    # initialize seg_dict, cy, cx
    segDict = {}
    cy=0
    cx=0

    # define front pixel location
    frontLoc = {
        0 : [[-1, +1, 7], [ 0, +1, 0], [+1, +1, 1]],
        1 : [[-1, +1, 7], [ 0, +1, 0], [+1, +1, 1], [+1,  0, 2], [+1, -1, 3]],
        2 : [[+1, -1, 3], [+1,  0, 2], [+1, +1, 1]],
        3 : [[-1, -1, 5], [ 0, -1, 4], [+1, -1, 3], [+1,  0, 2], [+1, +1, 1]],
        4 : [[-1, -1, 5], [ 0, -1, 4], [+1, -1, 3]],
        5 : [[+1, -1, 3], [ 0, -1, 4], [-1, -1, 5], [-1,  0, 6], [-1, +1, 7]],
        6 : [[-1, -1, 5], [-1,  0, 6], [-1, +1, 7]],
        7 : [[-1, -1, 5], [-1,  0, 6], [-1, +1, 7], [ 0, +1, 0], [+1, +1, 1]]
    }

    endBifList = endBifDict.keys()
    for coord, dirList in endBifDict.items() :
        y = coord[0]
        x = coord[1]
        for part_dir in dirList :
            segTmpList = []
            if part_dir==0 :
                cy = y
                cx=x+1
            elif part_dir==1:
                cy=y+1
                cx=x+1
            elif part_dir==2:
                cy=y+1
                cx = x
            elif part_dir==3:
                cy=y+1
                cx=x-1
            elif part_dir==4:
                cy = y
                cx=x-1
            elif part_dir==5:
                cy=y-1
                cx=x-1
            elif part_dir==6:
                cy=y-1
                cx = x
            elif part_dir==7:
                cy=y-1
                cx=x+1
            if visited[cy,cx]==1 : continue

            # increase segment id(segID)
            segId+=1
            segTmpList.append((y,x))
            segTmpList.append((cy,cx))
            visited[y,x]=1
            visited[cy,cx]=1
            # 두 칸짜리 토막
            if (cy,cx) in endBifList : continue
            switch = True
            while switch :
                # investigate front pixels
                moveLocs = frontLoc[part_dir]
                for my, mx, sub_dir in moveLocs :
                    frontCoord = (cy+my, cx+mx)
                    if img[frontCoord[0], frontCoord[1]]==1 :
                        if frontCoord in endBifList : # 마디 끝
                            segTmpList.append(frontCoord)
                            visited[frontCoord[0], frontCoord[1]] = 1
                            switch = False # while loop 문 끝
                            break
                        # 만일 frontCoord가 이민 segTmpList 내에 존재하면, 원형에서 돌고 있으므로 벗어나야 한다.
                        if frontCoord in segTmpList :
                            switch = False
                            break
                        segTmpList.append(frontCoord)
                        visited[frontCoord[0], frontCoord[1]] = 1
                        cy = frontCoord[0]
                        cx = frontCoord[1]
                        part_dir = sub_dir
                        break

            segDict[segId] = segTmpList
    t2 = process_time()
    print("EDGE SEGMENTATION TIME : {0}s".format(t2-t1))
    return segDict

if __name__ == "__main__" :

    file_dir = "D:\\cv\\data\\prac\\KakaoTalk_20220518_215457616_02.jpg"
    # thr = otsu_binary(file_dir,resize_scale=0.9)
    # img = binary_image(file_dir, thr)
    # img[img==1] = -1
    # binary = flood_fill(img=img,
    #                     mode=4)
    # binary = binary.astype(np.int8)
    # plt.imshow(binary, cmap="gray")
    # plt.show()

    # CANNY EDGE
    img = cv2.imread(file_dir, 0)
    edge_results = canny_edge(img=img,
                              imread_mode=0,
                              tLow=50,
                              tHigh=100,
                              resize_scale=1,
                              sigmaX=5,
                              sigmaY=5)

    spta_fast = SPTA(edge_results)
    edgeSeg = edgeSegment(spta_fast)






    # Yellow : Red+Green, Magenta : Red+Blue, Cyan : Green + Blue

    ## CANNY EDGE DI GENZO
    # edge_results2 = canny_edge_genzo(file_dir = file_dir,
    #                                  tLow = 50,
    #                                  tHigh = 100,
    #                                  resize_scale=0.5,
    #                                  sigmaX=5,
    #                                  sigmaY=5)

    # # video version
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    #
    # while cv2.waitKey(50) < 0:
    #     ret, frame = capture.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     edge_results = canny_edge(img=frame,
    #                               imread_mode=0,
    #                               tLow=50,
    #                               tHigh=100,
    #                               resize_scale=0,
    #                               sigmaX=5,
    #                               sigmaY=5)
    #     cv2.imshow("VideoFrame", edge_results*255)
    #
    # capture.release()
    # cv2.destroyAllWindows()



