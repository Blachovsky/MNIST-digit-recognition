# evaluate the deep model on the test dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

# load train and test dataset
def load_dataset():
	# load dataset
	testX = np.load('my_data/digits_x_test.npy')
	testY = np.load('my_data/digits_y_test.npy')
	testX = np.array(testX)
	testY = np.array(testY)
	testX = testX.reshape(testX.shape[0], 28, 28,1)
	# one hot encode target values
	testY = to_categorical(testY)
	return testX, testY

# scale pixels
def prep_pixels(test):
	# convert from integers to floats
	test_norm = test.astype('float32')
	# normalize to range 0-1
	test_norm = test_norm / 255.0
	# return normalized images
	return test_norm

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	testX, testY = load_dataset()
	# prepare pixel data
	testX = prep_pixels(testX)
	# load model
	model = load_model('models/my_data_model.h5')
	# evaluate model on test dataset
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
run_test_harness()