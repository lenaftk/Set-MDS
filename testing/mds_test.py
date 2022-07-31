from hashlib import md5
import sys
from tarfile import GNUTYPE_LONGNAME
import time
import numpy as np
sys.path.insert(0,'../')
sys.path.insert(0,'../mds')
sys.path.insert(0,'../set_mds')
sys.path.insert(0,'../synthetic_data')
from mds_len import MDS
from set_mds import split
from create_synthetic_dataset import SyntheticDataset
# from split import split

# Create Synthetic dataset
k=2
n_samples = 3
dataset1 = SyntheticDataset(n_samples=n_samples,
                            n_components=3,
                            k=k)
d_goal = dataset1.create_synthetic_dataset()
print("result = \n", d_goal)

# MDS routine
godzilla = MDS(
    n_components=2,
    starting_radius=10,
    explore_dim_percent= 1.0,
    prob_init= 0.8,
    prob_step = 0.1,
    verbose=1,
    mode = 'full_search',
#    n_landmarks = 100
)
start = time.time()
xs, error, d_current = godzilla.fit_transform(d_goal)
end = time.time()
print("Total MDS time : ", end - start)
print("Total MDS error : ", error)
#print([item[0] for item in time_error])

##### Set MDS routine. Use the d_current and the xs

def _midpoint(xs):
    midpoint = np.zeros((1,xs.shape[1]))
    for ii in range(0,xs.shape[1]):
        midpoint[0][ii] = xs[:,ii].sum()
    midpoint = midpoint/xs.shape[0]
    return midpoint

midpoint = _midpoint(xs)
print(midpoint)


xs = np.concatenate((xs,midpoint)) #add midpoint in the xs array, but consider maybe a different array splited_points(?)

split(xs,d_current, d_goal, error,k, n_samples)