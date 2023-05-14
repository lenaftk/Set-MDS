import numpy as np
from scipy.spatial import distance
import sys
import time

sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'../')
sys.path.insert(0,'./set_mds')
sys.path.insert(0,'./mds')
sys.path.insert(0,'./compute_mds_set_error')

from new_set_mds_help.set_mds import mds_set
from new_set_mds_help.mds import mds
from new_set_mds_help.compute_mds_set_error import compute_mds_set_error
from synthetic_data.create_synthetic_dataset import SyntheticDataset

# Create Synthetic dataset
savefigg = 'false'
k=10 #set for experiment
n_sets=11  #set for experiment
dataset1 = SyntheticDataset(n_samples=n_sets, 
                            n_components=2,
                            k=k
                            # seed=42
                            )
d_goal,sets = dataset1.create_random_synthetic_dataset()
print("\n \n \nresult = \n", d_goal.shape, sets)

K=n_sets+k
x,ind = mds_set(d_goal, K, dim=2)

print(x,ind)

