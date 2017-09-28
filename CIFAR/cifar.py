import cPickle
import numpy as np

def unpickle(file):  
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

def convert_images(vector):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Reshape the array to 4-dimensions.
    images = vector.reshape([3, 32, 32, -1])
    
    #images = vector.reshape([-1, 3, 32, 32])

    # Reorder the indices of the array.
    images = images.transpose([1, 2, 0, 3])
    #images = images.transpose([0, 2, 3, 1])

    return images

# zero out indices of a vector for a data matrix
# for a given mask
def hideData(data, mask):
    # copy the data
    newData = data.copy()
    
    # get indices from the mask
    x_idx = np.where([mask==1])[1]
    y_idx = np.where([mask==0])[1]
    
    # apply the mask
    newData[y_idx, :] = 0
    
    return newData, data[x_idx, :], data[y_idx, :]

# generates a 3072 element mask (32 x 32 x 3) with columns 
# deleted from the right side of the image towards the left
def generateColumnMask(delColumns):
    mask = np.ones((3, 32, 32))
    mask[:, :, (32 - delColumns):] = 0
    maskVec = mask.reshape(3072)
    return maskVec

# generate a 3072 element mask (32 x 32 x 3) with a square, 
# with side length (sideLength), zero'd out of the middle
def generateCenterSquareMask(sideLength):
    mask = np.ones((3, 32, 32))
    leftIdx = (32 - sideLength)/2
    rightIdx = (32 + sideLength)/2
    mask[:, leftIdx:rightIdx, leftIdx:rightIdx] = 0
    maskVec = mask.reshape(3072)
    return maskVec

# let's get the data for removed squares for CIFAR10 data
def returnSquareData(sideLength):
    train, test, train_labels, test_labels = returnCIFARdata()
    
    train_hideCenter, Xtrain_hideCenter, Ytrain_hideCenter = hideData(train, generateCenterSquareMask(sideLength))
    test_hideCenter, Xtest_hideCenter, Ytest_hideCenter = hideData(test, generateCenterSquareMask(sideLength))

    return train_hideCenter, Xtrain_hideCenter, Ytrain_hideCenter, test_hideCenter, Xtest_hideCenter, Ytest_hideCenter

# let's get the data for halves for CIFAR10 data
def returnHalfData(ncols):
    train, test, train_labels, test_labels = returnCIFARdata()
    
    train_hideRight, Xtrain_hideRight, Ytrain_hideRight = hideData(train, generateColumnMask(ncols))
    test_hideRight, Xtest_hideRight, Ytest_hideRight = hideData(test, generateColumnMask(ncols))

    return train_hideRight, Xtrain_hideRight, Ytrain_hideRight, test_hideRight, Xtest_hideRight, Ytest_hideRight

# parse the CIFAR data files
# CIFAR10 is a 32 by 32 image with 3 color channels
def returnCIFARdata():
    train_images = np.zeros((0, 3072))
    train_labels = np.zeros(0)
    test_images = np.zeros((0, 3072))
    test_labels = np.zeros(0)

    # read in the training data
    for j in range(5):
      d = unpickle('data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      train_images = np.concatenate((x, train_images), axis=0)
      train_labels = np.concatenate((y, train_labels))

    # read in the test data
    d = unpickle('test_batch')
    test_images = np.concatenate((test_images, d['data']), axis=0)
    test_labels = np.concatenate((test_labels, d['labels']))

    # have images in vector form
    train_images_vec = np.copy(train_images.T)/255.0
    test_images_vec = np.copy(test_images.T)/255.0

    # convert images from [3072] to [32, 32, 3]
    #train_images = convert_images(train_images)
    #test_images = convert_images(test_images)

    size = train_images_vec.shape[0]
    n_train = train_images_vec.shape[1]
    n_test = test_images_vec.shape[1]

    print '----CIFAR10 dataset loaded----'
    print 'Train data: %d x %d' %(size, n_train)
    print 'Test data: %d x %d' %(size, n_test)

    # convert labels from class numbers to one hot labels
    train_labels_onehot = np.zeros((10, n_train))
    test_labels_onehot = np.zeros((10, n_test))
    for i in range(n_train):
        idx = int(train_labels[i])
        train_labels_onehot[idx, i] = 1
    for i in range(n_test):
        idx = int(test_labels[i])
        test_labels_onehot[idx, i] = 1

    return train_images_vec, test_images_vec, train_labels, test_labels