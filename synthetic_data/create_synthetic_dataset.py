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
                 xs = None,
                 sets_text = None,
                 init = None):
        self.n_samples = n_samples
        self.n_components = n_components
        self.k = k
        self.seed = seed
        self.init = init
        self.sets = []
        self.Dn_k = None   #calculate distance between (n+k) x (n+k)
        self.Dn_n = None   #calculate distance between n_sets
        self.xs = xs
        self.sets_text = sets_text

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
                   print("----ii=", ii,"jj=",jj)
                   continue
                if self.Dn_k[ii][jj] < D[self.sets[ii]][self.sets[jj]]:
                    D[self.sets[ii]][self.sets[jj]] = self.Dn_k[ii][jj]
                    D[self.sets[jj]][self.sets[ii]] = self.Dn_k[ii][jj]

        return D          
    

    def create_random_synthetic_dataset(self):
        #self.xs= np.random.rand(self.n_samples + self.k, self.n_components)
        self.xs = np.random.uniform(low=-1.0, high=1.0, size=(self.n_samples + self.k, self.n_components))
        #assign points n+k points to n_sets
        for ii in range(self.n_samples):
            self.sets.append(ii)
        for jj in range(self.k):
            if self.seed == None:
                self.sets.append(random.randint(0,self.n_samples-1))
            else:
                self.sets.append(jj)
        print(self.sets)
        self.sets_text = [str(x) for x in self.sets]
        self.Dn_k = self.distance_matrix(self.xs)  # distance_matrix function is written in cython
        print(self.Dn_k)
        self.Dn_n = self.distance_matrix_between_sets()
                ### plotting scatter plot

        x=self.xs[:,0]
        y=self.xs[:,1]
        plt.scatter(x, y)
        ax = plt.gca()
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        plt.title("Initial points")

        # Loop for annotation of all points
        for i in range(len(x)):
            plt.annotate(self.sets_text[i], (x[i]+0.00002, y[i] + 0.00002))

    


        
        # Separate the first k points from the rest
        blue_points = self.xs[:self.n_samples]
        red_points = self.xs[self.n_samples:]
    # Create the scatter plot with blue and red points
        plt.scatter([p[0] for p in blue_points], [p[1] for p in blue_points], color='yellow')
        plt.scatter([p[0] for p in red_points], [p[1] for p in red_points], color='orange')


        ## adjusting the scale of the axes
        plt.savefig(f'./savefigs/initial.png')
        plt.close()
        ###

        # Separate the first k points from the rest
        blue_points = self.xs[:self.n_samples]
        red_points = self.xs[self.n_samples:]
        return self.Dn_n,self.Dn_k, self.sets





    def create_synthetic_dataset_generic(self):
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

    
    
    def create_synthetic_dataset(self):
        #create n+k points

        self.xs=np.array(self.xs)
        print("self.xs = ", self.xs)

        ### plotting scatter plot
        x=self.xs[:,0]
        y=self.xs[:,1]
        plt.scatter(x, y)
        ax = plt.gca()
        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        plt.title("Initial points")

        # Loop for annotation of all points
        for i in range(len(x)):
            plt.annotate(self.sets_text[i], (x[i]+0.002, y[i] + 0.002))

        # Separate the first k points from the rest
        blue_points = self.xs[:self.n_samples]
        red_points = self.xs[self.n_samples:]
    
    # Create the scatter plot with blue and red points
        plt.scatter([p[0] for p in blue_points], [p[1] for p in blue_points], color='blue')
        plt.scatter([p[0] for p in red_points], [p[1] for p in red_points], color='red')


        ## adjusting the scale of the axes
        plt.savefig(f'./savefigs/initial.png')
        plt.close()
        ###
    

        #calculate distance between n+k (xs)
        self.Dn_k = self.distance_matrix(self.xs)  # distance_matrix function is written in cython
        print(self.Dn_k)

        #assign points n+k points to n_sets, manually
        self.sets = [eval(i) for i in self.sets_text]
        print(self.sets)
        
        
        #calculate distance between sets
        self.Dn_n = self.distance_matrix_between_sets()
        print("self-sets=", self.sets)
        print("self.Dn_k= \n",self.Dn_k,"\n self.Dn_n= \n", self.Dn_n)
        return self.Dn_n  





# #### calling the function
# dataset1 = SyntheticDataset(n_samples=3,
#                             n_components=2,
#                             k=3)
# Dn_n = dataset1.create_synthetic_dataset()
# print("result = \n", Dn_n)












