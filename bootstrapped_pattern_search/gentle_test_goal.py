
from mds_len import MDS
import  matplotlib.pyplot as plt
import time
import sys
import numpy as np
sys.path.insert(0,'../')
from gentlemandata.gentlemandata import shapes
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_datasets.data_loader import data_loader

from mds_fast import (
    distance_matrix,
    update_distance_matrix,
    c_pertub_error as best_pertubation,
    mse as mse1d,
    mse2 as mse2d,
)

data_type = 'swissroll' # shape to generate
dim = 3  # shape dimension. Leave 3
distance = 'geodesic' # d_goal is calculated using geodesic or euclidean distance. Useful for MDS
npoints = 1000 #[500,2500,5000,15000]# Number of points to generate
n_neighbors = 12 # Neighbors are used for geodesic distance calculation
noise_std = 0 # Amount of noise in the data
landmarks = 100

xs, d_goal, color = data_loader(dataset_name = 'swissroll', npoints=npoints, distance='geodesic', memmap = False, n_neighbors=n_neighbors)
print(data_type, dim , distance, npoints, n_neighbors)

# xs, d_goal, color = (shapes.DataBuilder()    
#                 .with_dim(dim)
#                 .with_distance(distance)
#                 .with_noise(noise_std)
#                 .with_npoints(npoints)
#                 .with_neighbors(n_neighbors)
#                 .with_type(data_type)
#                 .with_memmap(False)
#                 .build())


godzilla = MDS(
    n_components=2,
    starting_radius=10,
    explore_dim_percent= 1.0,
    prob_init= 0.8,
    prob_step = 0.1,
    verbose=1,
    mode = 'full_search',
    dataset = 'swissroll',
    n_landmarks=100
)

start = time.time()
x_r, error , turn, time_error = godzilla.fit_transform(d_goal)
print("x_r shape", x_r.shape)
end = time.time()
print("Total MDS time : ", end - start )
#print([item[0] for item in time_error])


fig = plt.figure()
#  fig.suptitle('BS MDS', fontsize=14, fontweight='bold')    
plt.title(' points ')
ax = plt.axes(projection='3d')
ax.scatter(x_r[:, 0], x_r[:, 1],c=color, cmap=plt.cm.Spectral)
plt.title("BS MDS  sec)")
plt.show()
plt.savefig('1')



