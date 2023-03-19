from os import setsid
from re import L
from tkinter import N
from sklearn.utils import check_array, check_random_state
import time
import random
import numpy as np

from set_mds_fast import (
    distance_matrix,
    # distance_matrix_between_sets
)

class SyntheticDataset(object):
    def __init__(self,
                 n_samples = None,
                 n_components = None,
                 k = None,
                 seed=42,
                 init = None):
        self.n_samples = n_samples
        self.n_components = n_components
        self.k = k
        self.seed = seed
        self.init = init
        self.sets = []
        self.Dn_k = None   #calculate distance between (n+k) x (n+k)
        self.Dn_n = None   #calculate distance between n_sets

    def distance_matrix_between_sets(self):
        D = np.zeros((self.n_samples,self.n_samples))

        #initialize D matrix with NxN distance matrix
        for ii in range(self.n_samples):
            for jj in range(ii+1, self.n_samples):
                D[ii][jj] = self.Dn_k[ii][jj]
                D[jj][ii] = self.Dn_k[jj][ii]

        for ii in range(self.n_samples+self.k):
            for jj in range(self.n_samples+1,self.n_samples+self.k):
                if self.sets[ii] == self.sets[jj]:
                    continue
                if D[self.sets[ii]][self.sets[jj]] > self.Dn_k[ii][jj]:
                    D[self.sets[ii]][self.sets[jj]] = self.Dn_k[ii][jj]
                    D[self.sets[jj]][self.sets[ii]] = self.Dn_k[ii][jj]
        return D          


    def create_synthetic_dataset(self):
        #create n+k random points
        if self.seed == None:
            self.seed = random.randint(1,10000)
        self.seed = check_random_state(self.seed)
        xs = self.seed.rand(self.n_samples + self.k, self.n_components)

        #calculate distance between n+k random poins
        self.Dn_k = distance_matrix(xs)
        # print(self.Dn_k)

        #assign points n+k points to n_sets
        for ii in range(self.n_samples):
            self.sets.append(ii)
        for jj in range(self.k):
            self.sets.append(random.randint(0,self.k-1))
        
        #calculate distance between sets
        self.Dn_n = self.distance_matrix_between_sets()
        return self.Dn_n  





# #### calling the function
# dataset1 = SyntheticDataset(n_samples=3,
#                             n_components=2,
#                             k=3)
# Dn_n = dataset1.create_synthetic_dataset()
# print("result = \n", Dn_n)












