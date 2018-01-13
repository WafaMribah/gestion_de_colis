import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  random
import numpy as np
from matplotlib import pyplot
import matplotlib as mat

def load_data():
    from mnist import MNIST

    data_file = MNIST('library')

    train_images, train_labels = data_file.load_training()

    test_images, test_labels = data_file.load_testing()

    data ={
        'train_images':train_images,
        'train_labels':train_labels,
        'test_images' :test_images,
        'test_labels' :test_labels

    }
    return  data


def reshape(data) :
    for i in range(0,len(data['test_images'])):
        img= data['test_images'][i]
        array=np.asarray(img)
        pixel=array.reshape([28,28])
        data['test_images'][i]=pixel

    for i in range(0, len(data['train_images'])):
        img = data['train_images'][i]
        array = np.asarray(img)
        pixel = array.reshape([28, 28])
        data['train_images'][i] = pixel


    return data




def main():
  data_sets = load_data()
  print(data_sets['train_images'].__sizeof__())
  print(data_sets['train_labels'].__sizeof__())
  print(data_sets['test_images'].__sizeof__())
  print(data_sets['test_labels'].__sizeof__())
  index = random.randrange(0, len(data_sets['test_images']))
  #print(data_file.display(data_sets['test_images'][index]))
  reshape(data_sets)
  img = data_sets['test_images'][index]  # list
  print(img)
  pyplot.imshow(img, cmap='gray')
 # myarray = np.asarray(img)
  #pixels = myarray.reshape([28, 28])
  #print(pixels)
  #pyplot.imshow(pixels, cmap='gray')
  pyplot.show()

  pyplot.savefig("fig.png")





if __name__ == '__main__':
  main()

