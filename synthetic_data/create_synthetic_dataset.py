from sklearn.utils import check_array, check_random_state
import matplotlib.pyplot as plt
import time
import random
import numpy as np

# from set_mds_fast import (
#     distance_matrix,
#     # distance_matrix_between_sets
# )

class SyntheticDataset(object):
    def __init__(self,
                 n_samples = None,
                 n_components = None,
                 k = None,
                 seed = None,
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

        #calculate final values of NxN
        for ii in range(self.n_samples+self.k):
            for jj in range(self.n_samples,self.n_samples+self.k):
                if self.sets[ii] == self.sets[jj]:
                    continue
                if D[self.sets[ii]][self.sets[jj]] > self.Dn_k[ii][jj]:
                    D[self.sets[ii]][self.sets[jj]] = self.Dn_k[ii][jj]
                    D[self.sets[jj]][self.sets[ii]] = self.Dn_k[ii][jj]
        return D          



    def create_synthetic_dataset(self):
        #create n+k random points
        print(self.seed)
        if self.seed == None:
            self.seed = random.randint(1,10000)
        self.seed = check_random_state(self.seed)
        xs = self.seed.rand(self.n_samples + self.k, self.n_components)
      

        #calculate distance between n+k random poins
        self.Dn_k = distance_matrix(xs)  # distance_matrix function is written in cython
        print(self.Dn_k)

        #assign points n+k points to n_sets
        for ii in range(self.n_samples):
            self.sets.append(ii)
        for jj in range(self.k):
            if self.seed == None:
                self.sets.append(random.randint(0,self.k-1))
            else:
                self.sets.append(jj)
        print(self.sets)

        #calculate distance between sets
        self.Dn_n = self.distance_matrix_between_sets()
        return self.Dn_n  

    def distance_matrix(self,A):
        nrow = A.shape[0]
        ncol = A.shape[1]
        D = np.zeros((nrow, nrow), np.double)
        
        for ii in range(nrow):
            for jj in range(ii + 1, nrow):
                tmpss = 0
                for kk in range(ncol):
                    diff = A[ii, kk] - A[jj, kk]
                    tmpss += diff * diff
                tmpss = np.sqrt(tmpss)
                D[ii, jj] = tmpss
                D[jj, ii] = tmpss
        return D
    
    def create_synthetic_dataset_2(self):
        #create n+k random points
        print(self.seed)
        if self.seed == None:
            self.seed = random.randint(1,10000)
        self.seed = check_random_state(self.seed)
        xs = self.seed.rand(self.n_samples + self.k, self.n_components)
        print("xs", xs)

        xs= [[10,5],[-11,2],[10,-1],[-10,5],[-10,1]]

        #xs= [[10,5],[-11,2],[10,-1],[9,2],[5,3],[7,2],[10,1],[-5,-2],[5,-2],[6,6],[-6,6],[6,7],[-9,2],[-7,1],[-5,2],[2,3],[4,5],[-4,7],[-10,5],[-10,1]]

        xs=np.array(xs)
        print("xs", xs)
        x=xs[:,0]
        y=xs[:,1]
        text = [str(i%3) for i in range(5)]  
        #text = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","0","8"]
        # text = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","0","15"] == to vriskei swsta

        # for ii in range(15):
        #     text.append[0]
        # plotting scatter plot
        plt.scatter(x, y)
        ax = plt.gca()
        ax.set_xlim([-13, 13])
        ax.set_ylim([-7, 7])
        plt.title("Initial points")
        # Loop for annotation of all points
        for i in range(len(x)):
            plt.annotate(text[i], (x[i]+0.002, y[i] + 0.002))

        ## adjusting the scale of the axes
        plt.savefig(f'./savefigs/initial.png')
        plt.close()
    

        #calculate distance between n+k random poins
        self.Dn_k = self.distance_matrix(xs)  # distance_matrix function is written in cython
        print(self.Dn_k)

        #assign points n+k points to n_sets
        for ii in range(self.n_samples):
            self.sets.append(ii)
        for jj in range(self.k):
            if self.seed == None:
                self.sets.append(random.randint(0,self.k-1))
            else:
                self.sets.append(jj)
        print(self.sets)

        #calculate distance between sets
        self.Dn_n = self.distance_matrix_between_sets()
        return self.Dn_n  





# #### calling the function
# dataset1 = SyntheticDataset(n_samples=3,
#                             n_components=2,
#                             k=3)
# Dn_n = dataset1.create_synthetic_dataset()
# print("result = \n", Dn_n)












