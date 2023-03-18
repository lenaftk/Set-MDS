import  matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from gentlemandata_landmarks import shapes

data_type = 'swissroll' # shape to generate
dim = 3  # shape dimension. Leave 3
distance = 'geodesic' # d_goal is calculated using geodesic or euclidean distance. Useful for MDS
npoints = 10000 # Number of points to generate
n_neighbors = 12 # Neighbors are used for geodesic distance calculation
noise_std = 0 # Amount of noise in the data
landmarks = 10

xs, d_goal,  color = (shapes.DataBuilder()
                 .with_dim(dim)
                 .with_distance(distance)
                 .with_noise(noise_std)
                 .with_npoints(npoints)
                 .with_neighbors(n_neighbors)
                 .with_type(data_type)
                 .with_landmarks(landmarks)
                 .build())
# colors = np.full((landmarks),'black')
np.savez(distance +'_10land_' + str(npoints), xs =xs , d_goal = d_goal, color = color)