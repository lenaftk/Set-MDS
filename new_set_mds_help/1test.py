import numpy as np
import time
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


dim = 2
N = 10
extension_factor = 2
K = extension_factor * N
dext = np.empty((K, K))
d = np.empty((N, N))
x = np.random.rand(K, dim)
N1 = round(0.4 * N)
N2 = round(0.7 * N)
N3 = round(0.9 * N)
# ind = np.concatenate([np.arange(1, N1 + 1), np.arange(N1 + 2, N2 + 2), np.arange(N2 + 3, N3 + 3), np.arange(N3 + 4, N + 1)])
lis = ["1","2","3","4","5","6","7","8","9","10","1","1","1","2","2","3","3","4","5","6"]
ind = [int(i) for i in lis]
print('ela')
print(ind)

if len(ind) != K:
    raise ValueError('Choose a multiple of 10 for N to avoid rounding issues')

print(x[0],dext)
for i in range(K):
    for j in range(K):
        
        dext[i, j] = np.sqrt(np.sum((x[i, :] - x[j, :]) ** 2))
        print(dext[i,j])

for i in range(1,N):
    iind = np.where(ind == i)[0]
    print(i,ind,iind)
    for j in range(1,N):
        jind = np.where(ind == j)[0]
        print(iind,jind)
        d[i, j] = np.min(np.min(dext[iind, jind]))

print(dext)
print(d)

t0 = time.time()
new_x1 = mds(d, 2)
t1 = time.time()
new_x2, new_ind2 = mds_set(d, K, 2)
# t2 = time.time()
# new_x3, new_ind3 = mds_set_known_index(d, K, ind, 2)
# t3 = time.time()
# new_x4, new_ind4 = mds_set_fast(d, K, 2)
# t4 = time.time()
# new_x5, new_ind5 = mds_set_very_fast(d, K, 2)
# t5 = time.time()
no0,no2=[]
for i in range(N):
    no0[i] = len(np.where(ind == i)[0])
    no2[i] = len(np.where(new_ind2 == i)[0])
    # no3[i] = len(np.where(new_ind3 == i)[0])
    # no4[i] = len(np.where(new_ind4 == i)[0])
    # no5[i] = len(np.where(new_ind5 == i)[0])

d2 = np.sum(np.diff(no2 - no0) != 0)
# d3 = np.sum(np.diff(no3 - no0) != 0)
# d4 = np.sum(np.diff(no4 - no0) != 0)
# d5 = np.sum(np.diff(no5 - no0) != 0)

E2 = compute_mds_set_error(d, new_x2, new_ind2)
# E3 = compute_mds_set_error(d, new_x3, new_ind3)
# E4 = compute_mds_set_error(d, new_x4, new_ind4)
# E5 = compute_mds_set_error(d, new_x5, new_ind5)

N
d = [d2]  # [d2, d3, d4, d5]
E = [E2]  # [E2, E3, E4, E5]
t = np.diff([t1])[0] / 60  # diff([s1,s2,s3,s4,s5])/60

# split statistics
# no = np.empty((N,))
# for i in range(N):
#     no[i] = len(np.where(new_ind == i)[0])

# new_x = new_x5
# new_ind = new