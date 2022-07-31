#!/usr/bin/env python
# coding: utf-8
#import pyximport; pyximport.install(pyimport = True)
from re import S
from this import d
from tkinter.dialog import DIALOG_ICON
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
import time
import random
import sys

from setmds_fast import (
    distance_matrix,
    distance_matrix_landmarks,
    update_distance_matrix,
    update_distance_matrix_landmarks,
    distance_matrix_nk,
    c_pertub_error as best_pertubation,
    mse as mse1d,
    mse2 as mse2d,
    mse2_landmarks as mse2d_landmarks
)
from common import timemethod

# k_set matrix. this matrix will be 1d, and will show in which set the point belongs. 
# all_sets matrix.   This matrix will be 2d. in the 1st row (not 0), will have the index of the points that belong in set1 etc..
#      mallon tha arxikoipoihsw ton pinaka me -1 kai me diastaseis (k,k)
# xs[n+samples +kj] and i will find the corellated point.

def _matrix_initialization(n_samples,k,d_current):
    set_point_error = np.zeros(n_samples) #pinakas pou tha krataei to error, kai tha prepei na epilegw to kalitero. Na ginei max value kai oxi miden.
    k_set = np.negative(k) #min_value, which set
    d_current_nk = np.zeros((n_samples+k,n_samples+k))
    for ii in range(n_samples):
        for jj in range(ii+1,n_samples):
            d_current_nk[ii][jj] = d_current[ii][jj]
            d_current_nk[jj][ii] = d_current[ii][jj]
    return set_point_error, k_set, d_current_nk



def split(xs, d_current, d_goal, error, k,n_samples): ###thelw na vrw se poio set anikei to middlepoint
    set_point_errors, k_set, d_current_nk = _matrix_initialization(n_samples,k,d_current)
    print("oooo\n",d_current,"\neee\n",d_current_nk)
    print(xs.shape[0])
    for kj in range(k):
        for point in range(n_samples):
            # k_set[kj] = point 
            # prev_error = mse1d(d_current[point], d_goal[point]) # vriskw to palio error. Twra prepei na vrw to kainourio.
            d_current_n = distance_matrix_nk(xs,d_current_nk)
  
           # new_error[point] = mse2d_set(xs, d_goal[point], point, kj, all_sets) #prepei na parw olous tous sindiasmous twn sets.......
            print(point)
    print(d_current_n)













# def _log_iteration(turn, radius, prev_error, error):
#     print("Turn {0}: Radius {1}: (prev, error decrease, error): "
#           "({2}, {3}, {4})"
#           .format(turn, radius, prev_error, prev_error - error, error))

# def _radius_update(radius, error, prev_error, tolerance=1e-4):
#     if error >= prev_error or prev_error - error <= error * tolerance:
#         return radius * 0.5
#     return radius

# def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
#     if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
#         return points
#     if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
#         return points
#     keep = int(points.shape[0] * keep_percent)
#     return np.random.choice(points, size=keep, replace=False)

# def _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points,d_goal,d_current,xs,
#     prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose):
#     while turn <= max_iter  and radius > radius_barrier:      #oso exw akoma epanalipseis& i aktina ine sta oria
#         turn += 1    #+1 epanalipsi11111
#         radius = _radius_update(  #update tin aktina
#             radius, error, prev_error, tolerance=radius_update_tolerance)
#         prev_error = error
#         filtered_points = _point_sampling(points, keep_percent=sample_points)  #epilogi simeiwn
#         for point in filtered_points:    #gia kathe shmeio apo ta epilegmena simeia
#             point_error = mse1d(d_goal[point], d_current[point])
#             optimum_error, optimum_k, optimum_step = best_pertubation(
#                 xs,
#                 radius,
#                 d_current,
#                 d_goal,
#                 point,
#                 prob_thr,
#                 prob_step,
#                 prob_matrix,
#                 turn,
#                 percent=explore_dim_percent,
#                 n_jobs=n_jobs
#             )
#             if (point_error > optimum_error):
#                 error -= (point_error - optimum_error)
#                 d_current = update_distance_matrix(
#                 xs, d_current, point, optimum_step, optimum_k)
#                 xs[point, optimum_k] += optimum_step
#         if verbose >= 2:
#             _log_iteration(turn, radius, prev_error, error)
#     return xs, error, d_current

# def set_mds(d_goal,
#             d_current,
#             xs,
#             n_components=2,
#             starting_radius=1.0,
#             radius_update_tolerance=1e-4,
#             sample_points=1.0,
#             explore_dim_percent=1.0,
#             max_iter=1000,
#             radius_barrier=1e-3,
#             n_jobs=1, 
#             verbose=0,
#             random_state=None,
#             prob_thr=0.2,
#             prob_step=0.05,
#             prob_init=None,
#             allow_bad_moves='true',
#             ):
#     n_samples = d_goal.shape[0]
#     points = np.arange(n_samples)
#     radius = starting_radius
#     turn = 0
#     error = mse2d(d_goal, d_current) ### to error borw na to perasw san parameter kai na min to ipologisw ksana
#     prev_error = np.Inf
#     if verbose:
#         print("Starting Error : {}".format(error))
#     prob_matrix = np.full((n_samples, 2 * n_components), prob_init) #$$$$$$$$$$$$$


#     for x in xs


    
#     # xs, error, d_current = _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points, d_goal,d_current,xs,
#     # prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose)


#     return xs, error, d_current

# class SetMDS(BaseEstimator):
#     def __init__(self,
#                  n_components=2,
#                  starting_radius=1.0,
#                  max_iter=1000,
#                  radius_barrier=1e-3,
#                  explore_dim_percent=1.0,
#                  sample_points=1.0,
#                  radius_update_tolerance=1e-4,
#                  verbose=0,
#                  random_state=None,
#                  n_jobs=1,
#                  dissimilarity='precomputed',
#                  prob_thr = 0.2,
#                  prob_step = 0.05,
#                  prob_init = 1.0,
#                  n_landmarks = None,
#                  allow_bad_moves ='true',
#                  mode = 'bootstrapped',
#                  dataset = None
#                 ):
#         self.radius_update_tolerance = radius_update_tolerance
#         self.sample_points = sample_points
#         self.n_components = n_components
#         self.starting_radius = starting_radius
#         self.max_iter = max_iter
#         self.radius_barrier = radius_barrier
#         self.explore_dim_percent = explore_dim_percent
#         self.num_epochs = 0
#         self.verbose = verbose
#         self.random_state = random_state
#         self.n_jobs = 1
#         self.dissimilarity = dissimilarity
#         self.allow_bad_moves = allow_bad_moves
#         self.n_landmarks = n_landmarks
#         self.prob_thr = prob_thr
#         self.mode = mode
#         self.dataset = dataset
#         available_modes = ['full_search', 'bootstrapped']
#         if mode in available_modes:
#             self.mode = mode
#         else:
#             raise NotImplementedError("Mode: {} is not yet available."
#                                       " Try one of: {} instead"
#                                       "".format(mode, available_modes))

#         if mode in ['full_search']:
#             print("In full_search mode initial probability for each dimension must be 1.0 and probability step must be 0."
#                     "Assigning...")
#             self.prob_step = 0.
#             self.prob_init = 1.0
#         else:
#             self.prob_step = prob_step
#             self.prob_init = prob_init

#     @timemethod
#     def fit_transform(self, X, d_current, d_goal):
#         X = X.astype(np.float64)    ##o X einai tipou float64
#         X = check_array(X)          # elegxos gia na einai ola ta stoixeia tou X coble( oxi Inf)
#     #TREXEI PSMDS
#         if self.n_landmarks != None:
#             d_goal = d_goal[:,0:self.n_landmarks].copy(order='C')  # d_goal = (N,n_landrm)
#         self.embedding_, self.error_, self.d_current = set_mds(
#             d_goal,
#             d_current,
#             init=X,
#             n_components=self.n_components,
#             starting_radius=self.starting_radius,
#             max_iter=self.max_iter,
#             sample_points=self.sample_points,
#             explore_dim_percent=self.explore_dim_percent,
#             radius_update_tolerance=self.radius_update_tolerance,
#             radius_barrier=self.radius_barrier,
#             n_jobs=self.n_jobs,
#             verbose=self.verbose,
#             random_state=self.random_state,
#             prob_thr=self.prob_thr,
#             prob_step=self.prob_step,
#             prob_init=self.prob_init,   
#             allow_bad_moves=self.allow_bad_moves
#             )            
#         return self.embedding_, self.error_, self.d_current
        
#     def fit(self, X, init=None):   
#         self.fit_transform(X, init=init)
#         return self