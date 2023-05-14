import numpy as np
import sys
import time
sys.path.insert(0,'./bootstrapped_pattern_search')
sys.path.insert(0,'../')
sys.path.insert(0,'./set_mds')
sys.path.insert(0,'./mds')
sys.path.insert(0,'./compute_mds_set_error')

from new_set_mds_help.mds import mds

def compute_mds_set_error(d, x, ind, i=None):
    N = d.shape[0]
    K = x.shape[0]
    dim = x.shape[1]
    E = 0
    valid_x_ind = np.nonzero(ind)
    
    if i is not None:  # Faster version
      
        iind = np.nonzero(ind == i)[0]
        tmp = np.zeros((K,))
        tmp = np.zeros(x.shape[0])

        
        if dim == 2:
            for k in valid_x_ind[0]:
                j = k
                tmp[j] = np.min(np.sqrt((x[iind, 0] - x[j, 0]) ** 2 + (x[iind, 1] - x[j, 1]) ** 2))
                
            for j in range(1,N):
                jind = np.nonzero(ind == j)[0]
                # print("ind=",j, ind)
                # print("jind=",jind)
                # print(E)
                # print("tmp",min(tmp[jind]))
                E += (np.min(tmp[jind]) - d[i-1, j]) ** 2
        elif dim == 3:
            for k in valid_x_ind[0]:
                j = k
                tmp[j] = np.min(np.sqrt((x[iind, 0] - x[j, 0]) ** 2 + (x[iind, 1] - x[j, 1]) ** 2 + (x[iind, 2] - x[j, 2]) ** 2))
                
            for j in range(N):
                jind = np.nonzero(ind == j)[0]
                E += (np.min(tmp[jind]) - d[i-1, j]) ** 2
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