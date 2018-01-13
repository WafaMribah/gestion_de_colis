import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import pickle
import  random
from mnist import MNIST
from matplotlib import pyplot
import matplotlib as mat
import tkinter
import tensorflow as tf


with open('library/train-labels-idx1-ubyte', 'rb') as flbl:
        flbl.read(8)
        lbl = np.fromfile(flbl, dtype=np.int8)
        print(len(lbl))



with open('library/train-images-idx3-ubyte', 'rb') as f:

        magic_number=f.read(4)
        len_img=int.from_bytes(f.read(4), byteorder='big', signed=False)
        print(len_img)
        cols=int.from_bytes(f.read(4), byteorder='big', signed=False)
        rows=int.from_bytes(f.read(4), byteorder='big', signed=False)
        print(cols)
        print(rows)

        img = int.from_bytes(f.read(28*28), byteorder='big', signed=False)
        #A = np.fromfile(f,dtype=np.uint8,rows*cols);
        #input=int.from_bytes(f.readline(784),byteorder='big',signed=False)
        print(img)

        '''
      
        pixels = img.reshape((28, 28))
        pyplot.imshow(pixels, cmap='gray')
        pyplot.show()
       
        pyplot.savefig("fig.png")
        
        '''
        











