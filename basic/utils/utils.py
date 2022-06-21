import cv2
import matplotlib as plt
import numpy as np

class queue :
    def __init__(self):
        self.q_list = []

    def initQueue(self):
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

class imgTool :

    @staticmethod
    def plt2arr(img):

        fig = plt.figure()
"""
        t2 = process_time()
        lineIdx = np.where(accArray>=thr)
        x = np.array([i for i in range(shape[1])])
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        for p, t in zip(lineIdx[0], lineIdx[1]) :
            y = (-np.sin(np.deg2rad(t-90))*x + p-D)//np.cos(np.deg2rad(t-90))
            plt.plot(x, y, color='orange', alpha=0.1)
        plt.xlim(0, shape[1])
        plt.ylim(0, shape[0])
        plt.gca().invert_yaxis()
        fig.canvas.draw()
        f_arr = np.array(fig.canvas.renderer._renderer)

        if time!=None : print("Hough PLT to ARRAY PROCESSING TIME : ", t2 - t1)
        return f_arr
"""

if __name__ == "__main__" :
    pass