import numpy as np 

emb = np.load('Mnist_2000points_bootstrapped_landmarks_2part_True.npz')

emb1 = np.load('Mnist_2000points_full_search_landmarks_False.npz')
# emb1 = np.load('Mnist_2000points_full_search_landmarks_True.npz')
print(emb.files)

print(emb['error'])
print(emb1['error'])
print(emb['total_time'])
# print(emb1['total_time'])