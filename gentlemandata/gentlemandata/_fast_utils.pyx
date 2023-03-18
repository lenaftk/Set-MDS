#!python
#cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import numpy as np
cimport numpy as np
from numpy cimport ndarray as nd_arr
from cython.parallel cimport prange

# don't use np.sqrt - the sqrt function from the C standard library is much
# faster
from libc.math cimport sqrt


def distance_matrix_memmap(double[:, :] A, str filename):
    cdef:
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, kk
        np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, nrow), np.double)
        fp = np.memmap(filename,dtype='float64', mode='w+', shape=(nrow,nrow))
        double tmpss, diff

    for ii in range(nrow):
        for jj in range(ii + 1, nrow):
            tmpss = 0
            for kk in range(ncol):
                diff = A[ii, kk] - A[jj, kk]
                tmpss += diff * diff
            tmpss = sqrt(tmpss)
            fp[ii, jj] = tmpss
            fp[jj, ii] = tmpss
    del fp
    return

cpdef distance_matrix_landmarks_memmap(double[:, :] A, int  n_landmarks, str filename):
    cdef:
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, kk
        np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, n_landmarks), np.double)
        fp = np.memmap(filename,dtype='float64', mode='w+', shape=(nrow,n_landmarks))
        double tmpss, diff

    for ii in range(nrow):
        for jj in range(n_landmarks):
            tmpss = 0
            for kk in range(ncol):
                diff = A[ii, kk] - A[jj, kk]
                tmpss += diff * diff
            tmpss = sqrt(tmpss)
            fp[ii, jj] = tmpss
    del fp
    return

def distance_matrix(double[:, :] A):
    cdef:
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, kk
        np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, nrow), np.double)
        double tmpss, diff

    for ii in range(nrow):
        for jj in range(ii + 1, nrow):
            tmpss = 0
            for kk in range(ncol):
                diff = A[ii, kk] - A[jj, kk]
                tmpss += diff * diff
            tmpss = sqrt(tmpss)
            D[ii, jj] = tmpss
            D[jj, ii] = tmpss
            
    return D


cpdef distance_matrix_landmarks(double[:, :] A, int  n_landmarks):
    cdef:
        Py_ssize_t nrow = A.shape[0]
        Py_ssize_t ncol = A.shape[1]
        Py_ssize_t ii, jj, kk
        np.ndarray[np.float64_t, ndim=2] D = np.zeros((nrow, n_landmarks), np.double)
        double tmpss, diff

    for ii in range(nrow):
        for jj in range(n_landmarks):
            tmpss = 0
            for kk in range(ncol):
                diff = A[ii, kk] - A[jj, kk]
                tmpss += diff * diff
            tmpss = sqrt(tmpss)
            D[ii, jj] = tmpss
    return D