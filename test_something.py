
import numpy as np
from scipy.spatial.distance import cdist

# Generate some example points
xs = np.array([[1, 2], [3, 4], [5, 6]])

# Compute the distance matrix
distance_matrix = cdist(xs, xs)

print("Distance matrix:")
print(distance_matrix)


def change_point_to_distance_matrix(xs,d_current,point):
    xs_columns = xs.shape[1]
    xs_rows = xs.shape[0]
    for ii in range(xs_rows):
        tmp=0
        for jj in range(xs_columns):
            diff = xs[ii][jj]-xs[point][jj]
            tmp += diff*diff
        tmp = np.sqrt(tmp)
        d_current[ii][point] = tmp
        d_current[point][ii] = tmp
    return d_current


xs1 = np.array([[1, 2], [10, 8], [5, 6]])
d_current = change_point_to_distance_matrix(xs1,distance_matrix,1)
print(d_current)

xs2 = np.array([[1, 2], [10, 8], [10, -6]])
d_current = change_point_to_distance_matrix(xs2,distance_matrix,xs.shape[0]-1)
print(d_current)