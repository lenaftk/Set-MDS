
#!/usr/bin/env python
# coding: utf-8
#import pyximport; pyximport.install(pyimport = True)
from tkinter import N
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
import time
import random
import sys

from set_mds_fast import (
    distance_matrix,
    sets_distance_matrix
)

    def initialize_random_points(n_samples, n_components,init,random_state):
        if random_state == None:
            random_state = random.randint(1,10000)  ######to prosthesa giati den ginotan random, alliws den xriazete!!!
        random_state = check_random_state(random_state)  ###isws prepei na to valw polles fores, na tsekareis ligo an allazoun
        xs = init if init is not None else random_state.rand(n_samples, n_components)
        return xs


xs = initialize_random_points(3, 4, None, 20)

D = distance_matrix(xs)

random_state = check_random_state(21)
xs = np.vstack((xs,random_state.rand(1,4)))

sets =[[0],[1],[2,3]]

Dsets = sets_distance_matrix() 


#### me kapoion tropo tha orisw ta set estw oti einai [(0)],[(1)], [(2),(3)]
print(xs)
print(D)
