
'''
def load_data():

    data_file = MNIST('library')

    train_images, train_labels = data_file.load_training()

    test_images, test_labels = data_file.load_testing()


    #batching





    data ={
        'train_images':train_images,
        'train_labels':train_labels,
        'test_images' :test_images,
        'test_labels' :test_labels

    }




    index = random.randrange(0, len(test_images))
    print(data_file.display(data['test_images'][0]))
    #reshape_data(data)

    return data



def reshape_data(data):
  im_train = np.array(data['train_images'])
  im_train = np.reshape(im_train, (-1, 3, 28, 28))
  im_train = np.transpose(im_train, (0,2,3,1))
  data['train_images'] = im_train
  im_test = np.array(data['test_images'])
  im_test = np.reshape(im_test, (-1, 3,28, 28))
  im_test = np.transpose(im_test, (0,2,3,1))
  data['test_images'] = im_test
  return data





def main():
  data_sets = load_data()
  print(data_sets['train_images'].__sizeof__())
  print(data_sets['train_labels'].__sizeof__())
  print(data_sets['test_images'].__sizeof__())
  print(data_sets['test_labels'].__sizeof__())





if __name__ == '__main__':
  main()
'''



