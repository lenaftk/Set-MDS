import numpy as np
from scipy.spatial import distance
import sys
import time
sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'./synthetic_data')
sys.path.insert(0,'./set_mds')
#python program to check if a directory exists
import os
path = "./savefigs"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")


def mds(d, dim=2):
    """
    Multi-Dimensional Scaling (MDS) algorithm to compute 2D or 3D vectors
    that minimize the given distance matrix.
    
    Args:
        d (numpy.ndarray): NxN distance matrix
        dim (int): Dimension of space to project to, should be either 2 or 3 (default=2)
        
    Returns:
        x (numpy.ndarray): Nx2 or Nx3 vectors
        
    Raises:
        ValueError: If input distance matrix is not square or has size less than 3
        ValueError: If dimension of space to project to is not 2 or 3
    """
    if d.shape[0] != d.shape[1]:
        raise ValueError('Input distance matrix d should be square!')
    if d.shape[0] < 3:
        raise ValueError('Matrix size should be at least 3')
    if dim not in [2, 3]:
        raise ValueError('Dimension of space to project to should be either 2 or 3')
        
    N = d.shape[0]
    
    # Step 1: randomly assign x in 0-1, 0-1 or 0-1, 0-1, 0-1
    x = np.random.rand(N, dim)
    
    # Step 2: compute error
    def compute_mds_error(d, x, i):
        return np.sum((np.linalg.norm(x[i] - x, axis=1) - d[i])**2)
    
    E = np.zeros(N)
    for i in range(N):
        E[i] = compute_mds_error(d, x, i)
    
    # Step 3: Iterate
    v = np.cos(np.pi/4)
    points_org = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]])
    Np = points_org.shape[0]
    Enew = np.sum(E)
    iter = 0
    while ((iter == 0) or (Eorg > Enew)):
        iter += 1
        Eorg = Enew
        step = max(0.01, 0.1/(np.sqrt(iter)))
        points = step * points_org
        xorg = np.copy(x)
        for i in range(N):
            E[i] = compute_mds_error(d, x, i)
            Eloc = np.zeros(Np)
            for j in range(Np):
                x[i] = xorg[i] + points[j]
                Eloc[j] = compute_mds_error(d, x, i)
            ord = np.argmin(Eloc)
            x[i] = xorg[i] + points[ord]
            Enew = Enew - 2 * (E[i] - Eloc[ord])
            E[i] = Eloc[ord]
        print(iter, '(', step, '): ', Enew)
    return x

def compute_mds_set_error(d, x, ind, i=None):
    print("HEY")
    N = d.shape[0]
    K = x.shape[0]
    dim = x.shape[1]
    E = 0
    valid_x_ind = np.nonzero(ind)
    
    if i is not None:  # Faster version
        iind = np.nonzero(ind == i)[0]   #epistrefei tis theseis ston ind sugkekrimenou set 
        tmp = np.zeros((K,))
       # print("i is not none"," i=",i," ind=",ind," iind=",iind," temp=",tmp)

        if dim == 2:
            for k in valid_x_ind[0]:
                j = k
                tmp[j] = np.min(np.sqrt((x[iind, 0] - x[j, 0]) ** 2 + (x[iind, 1] - x[j, 1]) ** 2))
                
            for j in range(N):
                jind = np.nonzero(ind == j)[0]
                E += (np.min(tmp[jind]) - d[i, j]) ** 2

        elif dim == 3:
            for k in valid_x_ind[0]:
                j = k
                tmp[j] = np.min(np.sqrt((x[iind, 0] - x[j, 0]) ** 2 + (x[iind, 1] - x[j, 1]) ** 2 + (x[iind, 2] - x[j, 2]) ** 2))
                
            for j in range(N):
                jind = np.nonzero(ind == j)[0]
                E += (np.min(tmp[jind]) - d[i, j]) ** 2
    else:  # Slower version
        dext = np.zeros((K, K))
        
        for k in valid_x_ind[0]:
            i = k
            for l in valid_x_ind[0]:
                j = l
                dext[i, j] = np.sqrt(np.sum((x[i, :] - x[j, :]) ** 2))
                
        for i in range(N):
            iind = np.nonzero(ind == i)[0]
            for j in range(N):
                jind = np.nonzero(ind == j)[0]
                E += (np.min(np.min(dext[iind][:, jind])) - d[i, j]) ** 2

    return E

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
    ind[:N] = np.arange(0, N)


    # STEP 2: compute error
    E = np.zeros(N)
    for i in range(N):
        E[i] = compute_mds_set_error(d, x, ind, i)

    print(E)

    # Split & Iterate
    points_org = np.array([[0, 1], [-1, 0], [1, 0], [0, -1]]) # reduce computation time by 50%
    v = np.cos(np.pi/4)
    points_split_org = np.array([[0, 1], [1, 0], [v, v], [v, -v]]) # no need to split in other directions - symmetric
    Np = len(points_org)
    
    # Split the most promising point
    Kcur = N
    while Kcur < K:  # split to desired number of total elements
        print(Kcur , "K=",K, N)
        step_split = 0.03
        points_split = step_split * points_split_org
        print("points split",points_split)
        for i in range(N):
            E[i] = compute_mds_set_error(d, x, ind, i)
        
        for i in range(1,Kcur+1):
            print("Kcur= ",Kcur)
            print(x[i])
            xorg = x[i].copy()
            print( Kcur,i, ind,x)
            ind[Kcur+1] = ind[i]
            Eloc_split = np.zeros(Np)
            for j in range(Np):
                x[Kcur, :] = xorg + points_split[j, :]
                x[i, :] = xorg - points_split[j, :]
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
                E[i] = compute_mds_set_error(d, x, ind, i)
            print(f"Splitting ({Kcur-N} out of {K-N}): set {ind[ord]} with members: {np.where(ind==ind[ord])[0]}")
            print(f"Error after split: {np.sum(E)}")
            
            # Iterate only on recently split set x 3 times and then once on all
            print("Iterating on recent split")
            Enew = np.sum(E)
            iter = 0
            while iter == 0 or Eorg > Enew:
                iter = iter + 1
                Eorg = Enew
                step = 0.01 #



            


from synthetic_data.create_synthetic_dataset import SyntheticDataset
# from split import split


# Create Synthetic dataset
savefigg = 'false'
k=3 #set for experiment
n_sets=7  #set for experiment
dataset1 = SyntheticDataset(n_samples=n_sets, 
                            n_components=2,
                            k=k,
                            seed=42,
                            xs=[[-1,1],[-1,2],[7,2],[-3,-3],[6,5],[6,-1],[7,1],[5,4],[-5,-5],[-2,1]],  #set for experiment
                            sets_text=["0","1","2","3","4","5","6","0","1","2"]  #set for experiment
                            )
d_goal = dataset1.create_synthetic_dataset()
print("\n \n \nresult = \n", d_goal.shape)

K=10
mds_set(d_goal, K, dim=2)

