from matplotlib import pyplot as plt
import numpy as np
from mnist import MNIST
# load dataset
mdata = MNIST()
trainX, trainY = mdata.load(	'data_emnist/emnist-digits-train-images-idx3-ubyte',
                                'data_emnist/emnist-digits-train-labels-idx1-ubyte')
testX, testY = mdata.load(	    'data_emnist/emnist-digits-test-images-idx3-ubyte',
                                'data_emnist/emnist-digits-test-labels-idx1-ubyte')
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)
trainX = trainX.reshape(trainX.shape[0], 28, 28)
testX = testX.reshape(testX.shape[0], 28, 28)
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
for i in range(9):
 # define subplot
 plt.subplot(330 + 1 + i)
 # plot raw pixel data
 plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()