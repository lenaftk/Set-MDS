import numpy as np
from scipy.spatial import distance
import sys
import time

sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'../')
sys.path.insert(0,'./set_mds')
sys.path.insert(0,'./mds')
sys.path.insert(0,'./compute_mds_set_error')

from new_set_mds_help.mds import mds
from new_set_mds_help.compute_mds_set_error import compute_mds_set_error

def mds_set(d, K, dim=2):
    if dim not in [2, 3]:
        raise ValueError("Dimension of space to project to should be either 2 or 3")
    if K <= d.shape[0]:
        raise ValueError("Number of x vectors should be larger than matrix size")
    
    N = d.shape[0]
    
    # STEP 1: randomly assign x in 0-1, 0-1
    x = np.zeros((K, dim))
    x[:N, :] = mds(d, dim)
    ind = np.zeros(K, dtype=int) # ind stores the mapping between x elements and sets
    ind[:N] = np.arange(1, N+1)
    print(ind)

    print("MDS HAS FINISHED")
    print(x)
    

    # STEP 2: compute error
    E = np.zeros(N)
    for i in range(N):
        E[i] = compute_mds_set_error(d, x, ind, i+1)

    print("E",E)

    # Split & Iterate
    points_org = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]]) # reduce computation time by 50%
    v = np.cos(np.pi/4)
    points_split_org = np.array([[0, 1], [1, 0], [v, v], [v, -v]]) # no need to split in other directions - symmetric
    Np = len(points_org)
    
    # Split the most promising point
    Kcur = N
    while Kcur < K:  # split to desired number of total elements
        step_split = 0.03
        points_split = step_split * points_split_org
        for i in range(N):
            E[i] = compute_mds_set_error(d, x, ind, i+1)
        
        for i in range(Kcur):
            xorg = x[i].copy()
            ind[Kcur] = ind[i]
            print("ind Kcur",ind)
            Eloc_split = np.zeros(Np)
            for j in range(Np):
                x[Kcur, :] = xorg + points_split[j, :]
                x[i, :] = xorg - points_split[j, :]
                print(i,ind[i],d)
                Eloc_split[j] = compute_mds_set_error(d, x, ind, ind[i])
                x[i, :] = xorg.copy()  # reset
            ord = np.argmin(Eloc_split)
            Esplit = Eloc_split[ord] - E[ind[i]-1]
            split_ord = ord
            x[i, :] = xorg - points_split[split_ord, :]
            x[Kcur, :] = xorg + points_split[split_ord, :]
            ind[Kcur] = ind[i] # bug fix for splits > N !!!
            Kcur = Kcur + 1
            for i in range(N):
                E[i] = compute_mds_set_error(d, x, ind, i+1)
            print(f"Splitting ({Kcur-N} out of {K-N}): set {ind[ord]} with members: {np.where(ind==ind[ord])[0]+1}")
            print(f"Error after split: {np.sum(E)}")
            
            # Iterate only on recently split set x 3 times and then once on all
            print("Iterating on recent split")
            Enew = np.sum(E)
            iter = 0
            while iter == 0 or Eorg > Enew:
                iter = iter + 1
                Eorg = Enew
                step = 0.01 #
    return x,ind
    print(ind)