import cv2
import math
import numpy as np
import numpy.linalg as lin
from matplotlib import pyplot as plt

"""
input data format : [x, y, i]^T, homogeneous coordinate
                    (i = 1 or h, dummy)
"""
def arctan2(y, x) :
    if x == 0:
        if y > 0 : theta = 90
        elif y < 0 : theta = -90
    elif x > 0:
        if y==0 : theta = 0
        elif y > 0 : theta = math.degrees(math.atan(y/x))
        else : theta = math.degrees(math.atan(y/x))
    else:
        if y==0 : theta = 180
        elif y > 0 : theta = 180 + math.degrees(math.atan(y/x))
        else : theta = -180 + math.degrees(math.atan(y/x))
    return theta

def tran(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

def rot(theta):
    """
    :param theta: unit - degree
    :return:
    """
    rad = theta*np.pi/180
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad),  np.cos(rad), 0], [0,            0,           1]])

def scale(sx, sy):
    return np.array([[sx, 0, 1], [0, sy, 1], [0, 0, 1]])

def inrange(coord, shape):
    return coord[0]>=0 and coord[0]<shape[1] and coord[1]>=0 and coord[1]<shape[0]

def bi_inter(coord, shape):

    upper_left = np.array([math.floor(coord[0]), math.floor(coord[1])])
    upper_right = np.array([math.floor(coord[0]), math.ceil(coord[1])])
    bottom_left = np.array([math.ceil(coord[0]), math.floor(coord[1])])
    bottom_right = np.array([math.ceil(coord[0]), math.ceil(coord[1])])
    if not inrange(upper_left, shape) : upper_left = coord.astype(np.int16)
    if not inrange(upper_right, shape) : upper_right = coord.astype(np.int16)
    if not inrange(bottom_left, shape) : bottom_left = coord.astype(np.int16)
    if not inrange(bottom_right, shape) : bottom_right = coord.astype(np.int16)

    return upper_left, upper_right, bottom_left, bottom_right

def rotation(file_dir,
             theta,
             x_resize=0.5,
             y_resize=0.5,
             center=True,
             interp=True):
    """
    rotate image theta,
    :param file_dir: image directory
    :param theta: unit - degree
    :param center: rotate image in origin(=False) or center(=True)
    :param interp: binary interpolation
    :return:
    """
    img = cv2.imread(file_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(0,0), fx=x_resize, fy=y_resize)

    shift = tran(0, 0)
    shift_rev = tran(0, 0)
    if center :
        shift = tran(-img.shape[1] / 2., -img.shape[0] / 2.)
        shift_rev = tran(img.shape[1] / 2., img.shape[0] / 2.)
    rot_matrix = rot(theta)
    # h matrix order : shift center of the image to origin (shift)
    # -> rotate image (rotation) -> shift original frame (shift_rev)
    h_matrix = lin.multi_dot([shift_rev, rot_matrix, shift])

    # frame shape : (y, x, z)
    frame = np.zeros_like(img)

    x_coord = np.array([x for x in range(frame.shape[1])])
    y_coord = np.array([y for y in range(frame.shape[0])])
    x_coord, y_coord = np.meshgrid(x_coord, y_coord)
    """
    coord_homo shape : [[x1, x2, x3, ....],
                        [y1, y1, y1, ....],
                        [ 1,  1,  1, ....]]
    """
    coord_homo = np.stack([x_coord.flatten(),
                           y_coord.flatten(),
                           np.ones_like(x_coord.flatten())], axis=0).astype(np.int64)
    source_coord = np.dot(lin.inv(h_matrix), coord_homo)

    for i, j in zip(source_coord.transpose(1,0), coord_homo.transpose(1,0)):
        if i[0]>=0 and i[0]<frame.shape[1] and i[1]>=0 and i[1]<frame.shape[0]:
            if interp :
                ul, ur, bl, br = bi_inter(i, img.shape)
                alpha = i[0]-ul[0]
                beta = i[1]-ul[1]
                interp_1 = (1-alpha) * img[ul[1]][ul[0]][:] + alpha * img[ur[1]][ur[0]][:]
                interp_2 = (1-alpha) * img[bl[1]][bl[0]][:] + alpha * img[br[1]][br[0]][:]
                interp_3 = (1-beta) * interp_1 + beta * interp_2
                frame[j[1]][j[0]][:] = interp_3
            else :
                frame[j[1]][j[0]][:] = img[i[1].astype(int)][i[0].astype(int)][:]

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(frame)
    plt.show()

if __name__ == "__main__" :
    test = arctan2(-2, 3)
    # file_dir = "D:/cv/data/parts/KakaoTalk_20220518_215457616_01.jpg"
    # rotation(file_dir=file_dir,
    #          theta=10,
    #          interp=False)