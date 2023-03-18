import numpy as np
import cv2
import os

IMG_DIR = '/home/lena/diplomatiki/06_datasets/03_coil20/coil-20-proc'

for img in os.listdir(IMG_DIR):
        img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
        img_array = img_array.astype(int)
        img_array = (img_array.flatten())

        img_array  = img_array.reshape(-1, 1).T

        print(img_array)

        with open( '/home/lena/diplomatiki/06_datasets/03_coil20/coil-20-proc/output.csv', 'ab') as f:

            np.savetxt(f, img_array, delimiter=",")