import sys
import time
import numpy as np
sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'./synthetic_data')
sys.path.insert(0,'./set_mds')

from bootstrapped_pattern_search.mds_len import MDS
from set_mds_1 import (add_midpoint, set_mds)
#apo edw fortwnei ta datasets...
from synthetic_data.create_synthetic_dataset import SyntheticDataset
# from split import split

#python program to check if a directory exists
import os
path = "./savefigs"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")


# # Create Synthetic dataset
# savefigg = 'false'
# k=3 #set for experiment
# n_sets=6  #set for experiment
# dataset1 = SyntheticDataset(n_samples=n_sets, 
#                             n_components=2,
#                             k=k,
#                             seed=42,
#                             xs=[[-1,1],[-1,2],[7,2],[-3,-3],[6,5],[6,-1],[5,4],[-5,-5],[-2,1]],  #set for experiment
#                             sets_text=["0","1","2","3","4","5","0","1","2"]  #set for experiment
#                             )
# d_goal = dataset1.create_synthetic_dataset()
# print("\n \n \nresult = \n", d_goal.shape)

# Create Synthetic dataset
savefigg = 'false'
k=1 #set for experiment
n_sets=10 #set for experiment
dataset1 = SyntheticDataset(n_samples=n_sets, 
                            n_components=2,
                            k=k
                            )
d_goal,dnk,sets = dataset1.create_random_synthetic_dataset()
print("\n \n \nresult = \n", d_goal,"\n d_sets = \n",dnk, "\n sets = \n",sets)




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
    random_state=20,
    savefig = savefigg
)

start = time.time()
xs, error, turn, time_error, d_current = godzilla.fit_transform(d_goal)
end = time.time()
print("Total MDS time : ", end - start)
print("Total MDS error : ", error)
# #print([item[0] for item in time_error])

# ##### Set MDS routine. Use the d_current and the xs


set_mds(xs,d_current, d_goal, error, k, n_sets, savefigg,sets)
