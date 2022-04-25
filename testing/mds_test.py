import sys
import time
import numpy as np
sys.path.insert(0,'../')
sys.path.insert(0,'../mds')
sys.path.insert(0,'../set-mds')
sys.path.insert(0,'../synthetic_data')
from mds_len import MDS
from create_synthetic_dataset import SyntheticDataset


# Create Synthetic dataset
dataset1 = SyntheticDataset(n_samples=1000,
                            n_components=3,
                            k=100)
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
x_r, error, d_current = godzilla.fit_transform(d_goal)
end = time.time()
print("Total MDS time : ", end - start )
print("Total MDS error : ", error)
#print([item[0] for item in time_error])

# Set MDS routine. Use the d_current

