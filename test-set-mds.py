
import sys
import time
import numpy as np
sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'./synthetic_data')
sys.path.insert(0,'./set_mds')

from bootstrapped_pattern_search.mds_len import MDS
from set_mds_2 import (add_midpoint, set_mds)
#apo edw fortwnei ta datasets...
from synthetic_data.create_synthetic_dataset import SyntheticDataset
# from split import split

# Create Synthetic dataset
k= 2
n_sets = 18
dataset1 = SyntheticDataset(n_samples=n_sets,
                            n_components=2,
                            k=k,
                            seed=42
                            )
d_goal = dataset1.create_synthetic_dataset_2()
print("\n \n \nresult = \n", d_goal.shape)


# MDS routine
godzilla = MDS(
    n_components=2,
    starting_radius=10,
    explore_dim_percent= 1.0,
    prob_init= 1.0,
    prob_step = 0.0,
    verbose=1,
    mode = 'full_search',
    dataset='set-mds-synthetic-dataset',
   # n_landmarks = 100,
    random_state=20
)

start = time.time()
xs, error, turn, time_error, d_current = godzilla.fit_transform(d_goal)
end = time.time()
print("Total MDS time : ", end - start)
print("Total MDS error : ", error)
# #print([item[0] for item in time_error])

# ##### Set MDS routine. Use the d_current and the xs


set_mds(xs,d_current, d_goal, error, k, n_sets)