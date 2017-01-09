# -----------------------------------------------------------------------------
#     MNIST processing
# -----------------------------------------------------------------------------

import numpy as np

# packing and unpacking images
def packcw(A,nr,nc):
    x = (A.T).reshape(nr*nc,1)
    return x

def unpackcw(x,nr,nc):
    A = x.reshape(nc,nr)
    return A.T

def packrw(A,nr,nc):
    x = A.reshape(nr*nc,1)
    return x

def unpackrw(x,nr,nc):
    A = x.reshape(nr,nc)
    return A

# generates a 784 element mask with columns deleted
# from the right side of the image towards the left
def generateColumnMask(delColumns):
    mask = np.ones((28, 28))
    mask[:, (28 - delColumns):] = 0
    maskVec = packcw(mask, 28, 28)
    return maskVec

# generate a 784 element mask with a square, with side
# length (sideLength), zero'd out of the middle
def generateCenterSquareMask(sideLength):
    mask = np.ones((28, 28))
    leftIdx = (28 - sideLength)/2
    rightIdx = (28 + sideLength)/2
    mask[leftIdx:rightIdx, leftIdx:rightIdx] = 0
    maskVec = packcw(mask, 28, 28)
    return maskVec

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

# get statistics for mmse
def getStatistics(data, vectorMask):
    # get mean and covariance of original data
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    
    # get indices for X, Y parts of data
    x_idx = np.where([vectorMask == 0])[1]
    y_idx = np.where([vectorMask == 1])[1]
        
    # apply masks
    u_x = mean[x_idx]
    u_y = mean[y_idx]
    cov_x = cov[x_idx]
    cov_x = cov_x[:, x_idx]
    cov_yx = cov[y_idx]
    cov_yx = cov_yx[:, x_idx]
    cov_y = cov[y_idx]
    cov_y = cov_y[:, y_idx]
    
    # return statistics
    return u_x, u_y, cov_x, cov_yx

# let's get the data for halves
def returnHalfData(endBuffer):
    train = np.load('MNISTcwtrain1000.npy')
    train = train.astype(float)/255
    test = np.load('MNISTcwtest100.npy')
    test = test.astype(float)/255

    if (endBuffer):
        train = np.concatenate((train, train[:, 0:50]), axis = 1)
        #test = np.concatenate((test, test[:, 0:50]), axis = 1)

    size = train.shape[0]
    n_train = train.shape[1]
    n_test = test.shape[1]

    print '----MNIST dataset loaded----'
    print 'Train data: %d x %d' %(size, n_train)
    print 'Test data: %d x %d' %(size, n_test)

    train_hideRight, Xtrain_hideRight, Ytrain_hideRight = hideData(train, generateColumnMask(14))
    test_hideRight, Xtest_hideRight, Ytest_hideRight = hideData(test, generateColumnMask(14))

    return train_hideRight, Xtrain_hideRight, Ytrain_hideRight, test_hideRight, Xtest_hideRight, Ytest_hideRight

# let's get the data for halves
def returnSquareData(endBuffer):
    train = np.load('MNISTcwtrain1000.npy')
    train = train.astype(float)/255
    test = np.load('MNISTcwtest100.npy')
    test = test.astype(float)/255

    if (endBuffer):
        train = np.concatenate((train, train[:, 0:50]), axis = 1)
        #test = np.concatenate((test, test[:, 0:50]), axis = 1)

    size = train.shape[0]
    n_train = train.shape[1]
    n_test = test.shape[1]

    print '----MNIST dataset loaded----'
    print 'Train data: %d x %d' %(size, n_train)
    print 'Test data: %d x %d' %(size, n_test)

    train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle = hideData(train, generateCenterSquareMask(5))
    test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = hideData(test, generateCenterSquareMask(5))

    return train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle

# mix data half square in middle removed, half rightside removed
def returnMixData(endBuffer):
  train_hideRight, Xtrain_hideRight, Ytrain_hideRight, \
  test_hideRight, Xtest_hideRight, Ytest_hideRight = returnHalfData(endBuffer=False)

  train_hideMiddle, Xtrain_hideMiddle, Ytrain_hideMiddle, \
  test_hideMiddle, Xtest_hideMiddle, Ytest_hideMiddle = returnSquareData(endBuffer=endBuffer)

  train_mix = np.concatenate((train_hideRight, train_hideMiddle), axis =1)
  #Xtrain_mix = np.concatenate((Xtrain_hideRight, Xtrain_hideMiddle), axis =1)
  #Ytrain_mix = np.concatenate((Ytrain_hideRight, Ytrain_hideMiddle), axis =1)
  test_mix = np.concatenate((test_hideRight, test_hideMiddle), axis =1)
  #Xtest_mix = np.concatenate((Xtest_hideRight, Xtest_hideMiddle), axis =1)
  #Ytest_mix = np.concatenate((Ytest_hideRight, Ytest_hideMiddle), axis =1)

  train_truth_1, test_truth_1 = returnData(endBuffer=False)
  train_truth_2, test_truth_2 = returnData(endBuffer=endBuffer)

  train_truth = np.concatenate((train_truth_1, train_truth_2), axis =1)
  test_truth = np.concatenate((test_truth_1, test_truth_2), axis =1)

  return train_mix, test_mix, train_truth, test_truth

# let's get the data
def returnData(endBuffer):
    train = np.load('MNISTcwtrain1000.npy')
    train = train.astype(float)/255
    test = np.load('MNISTcwtest100.npy')
    test = test.astype(float)/255

    if (endBuffer):
        train = np.concatenate((train, train[:, 0:50]), axis = 1)
        #test = np.concatenate((test, test[:, 0:50]), axis = 1)

    size = train.shape[0]
    n_train = train.shape[1]
    n_test = test.shape[1]

    print '----MNIST dataset loaded----'
    print 'Train data: %d x %d' %(size, n_train)
    print 'Test data: %d x %d' %(size, n_test)

    return train, test
