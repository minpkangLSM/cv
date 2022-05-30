import cv2
import numpy as np

class queue :
    def __init__(self):
        self.q_list = []

    def enqueue(self, value):
        self.q_list.append(value)

    def dequeue(self):
        if len(self.q_list)!=0:
            return self.q_list.pop(0)

class stack :
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if len(self.stack)!=0:
            return self.stack.pop(-1)

    def peek(self):
        if len(self.stack)!=0:
            return self.stack[-1]

def vid2img(file_dir) :
    vidcap = cv2.VideoCapture(file_dir)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 15 == 0:
            cv2.imwrite("C:\\Users\\Kangmin\\Desktop\\gongam\\T_star\\data\\5\\capture\\{0}.jpg".format(count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    print("finish! convert video to frame")

if __name__ == "__main__" :
    file_dir = "C:\\Users\\Kangmin\\Desktop\\gongam\\T_star\\data\\5\\1.mp4"
    output_dir = "C:\\Users\\Kangmin\\Desktop\\gongam\\T_star\\data\\5\\capture\\{0}.jpg"
    vid2img(file_dir)