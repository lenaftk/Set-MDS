from mlxtend.data import loadlocal_mnist
import numpy as np

'''
Load MNIST dataset into numpy array,  and save in csv files.
'''


X, y = loadlocal_mnist(
        images_path='/home/lena/diplomatiki/06_datasets/01_mnist_digits/train-images-idx3-ubyte', 
        labels_path='/home/lena/diplomatiki/06_datasets/01_mnist_digits/train-labels-idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

np.savetxt(fname='/home/lena/diplomatiki/06_datasets/01_mnist_digits/images.csv', 
        X=X, delimiter=',', fmt='%d')
np.savetxt(fname='/home/lena/diplomatiki/06_datasets/01_mnist_digits/labels.csv', 
        X=y, delimiter=',', fmt='%d')