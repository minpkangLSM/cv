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

if __name__ == "__main__" :
    pass