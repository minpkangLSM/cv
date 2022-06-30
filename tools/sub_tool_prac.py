import numpy as np

class Node :
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right= right

class bst :

    def __init__(self):
        self.root = None

    def get(self, k):
        return self.get_item(n=self.root,
                             key=k)

    def get_item(self, n, key):
        if n == None :
            return None
        elif n.key > key:
            return self.get_item(n.left, key)
        elif n.key < key:
            return self.get_item(n.right, key)
        else :
            return n.key

    def put(self, k):
       self.root = self.put_item(self.root, k)

    def put_item(self, n, k):
        if n==None : return Node(k)
        if n.key > k : n.left = self.put_item(n.left, k)
        elif n.key < k : n.right = self.put_item(n.right, k)
        return n

    def min(self):
        if self.root == None : return None
        return self.minimum(self.root)

    def minimum(self, n):
        if n.left == None : return n
        else : self.minimum(n.left)

    def delete_min(self):
        if self.root == None : print("None")
        self.root = self.del_min(self.root)

    def del_min(self, n):
        if n.left == None : return n.right
        n.left = self.del_min(n.left)
        return n

    def delete(self, k):
        self.root = self.del_node(self.root, k)

    def del_node(self, n, k):
        if n == None :
            return None
        if n.key > k :
            n.left = self.del_node(n, k)
        elif n.key < k :
            n.right = self.del_node(n, k)
        else :
            if n.right == None : return n.left
            if n.left == None : return n.right
            target = n
            n = self.minimum(target)
            n.left = target.left
            n.right = self.del_min(target.right)
        return n


