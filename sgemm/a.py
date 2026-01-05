import numpy as np

def test_acc():
    a = np.arange(16).reshape(4, 4)
    b = np.arange(16).reshape(4, 4)
    c = a @ b
    print(a)
    print(c)

    '''
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
     
    [[ 56  62  68  74]
     [152 174 196 218]
     [248 286 324 362]
     [344 398 452 506]]
    '''

class Mat:
    def __init__(self, data, stride, offset=0):
        self.data = data
        self.stride = stride
        self.offset = offset

    def index(self, step, i, j):
        return self.data[i * self.stride * step + j * step + self.offset]

    def set(self, step, i, j, val):
        self.data[i * self.stride * step + j * step + self.offset] = val
    def slice(self, step, i , j):
        new_offset = i * self.stride * step + j * step + self.offset
        new_stride = self.stride
        return Mat(self.data, new_stride, new_offset)

data = np.arange(16)
a = Mat(data, 4)
b = a.slice(2, 1, 1)
b.set(1, 0, 0, -1)
b.set(1, 0, 1, -1)
b.set(1, 1, 0, -1)
b.set(1, 1, 1, -1)

print(data.reshape(4, 4))





