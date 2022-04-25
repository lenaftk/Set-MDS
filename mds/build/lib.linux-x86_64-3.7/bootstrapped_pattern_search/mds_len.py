#!/usr/bin/env python
# coding: utf-8
#import pyximport; pyximport.install(pyimport = True)
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
import time
import random
import sys

from mds_fast import (
    distance_matrix,
    distance_matrix_landmarks,
    update_distance_matrix,
    update_distance_matrix_landmarks,
    c_pertub_error as best_pertubation,
    mse as mse1d,
    mse2 as mse2d,
    mse2_landmarks as mse2d_landmarks
)
from common import timemethod

def _log_iteration(turn, radius, prev_error, error):
    print("Turn {0}: Radius {1}: (prev, error decrease, error): "
          "({2}, {3}, {4})"
          .format(turn, radius, prev_error, prev_error - error, error))

def _radius_update(radius, error, prev_error, tolerance=1e-4):
    if error >= prev_error or prev_error - error <= error * tolerance:
        return radius * 0.5
    return radius

def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
    if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
        return points
    if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
        return points
    keep = int(points.shape[0] * keep_percent)
    return np.random.choice(points, size=keep, replace=False)

def initialize_random_points(n_samples, n_components,init,random_state):
    if random_state == None:
        random_state = random.randint(1,10000)  ######to prosthesa giati den ginotan random, alliws den xriazete!!!
    random_state = check_random_state(random_state)  ###isws prepei na to valw polles fores, na tsekareis ligo an allazoun
    xs = init if init is not None else random_state.rand(n_samples, n_components)
    return xs

def _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points,d_goal,d_current,xs,
    prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose):
    while turn <= max_iter  and radius > radius_barrier:      #oso exw akoma epanalipseis& i aktina ine sta oria
        turn += 1    #+1 epanalipsi11111
        radius = _radius_update(  #update tin aktina
            radius, error, prev_error, tolerance=radius_update_tolerance)
        prev_error = error
        filtered_points = _point_sampling(points, keep_percent=sample_points)  #epilogi simeiwn
        for point in filtered_points:    #gia kathe shmeio apo ta epilegmena simeia
            point_error = mse1d(d_goal[point], d_current[point])
            optimum_error, optimum_k, optimum_step = best_pertubation(
                xs,
                radius,
                d_current,
                d_goal,
                point,
                prob_thr,
                prob_step,
                prob_matrix,
                turn,
                percent=explore_dim_percent,
                n_jobs=n_jobs
            )
            if (point_error > optimum_error):
                error -= (point_error - optimum_error)
                d_current = update_distance_matrix(
                xs, d_current, point, optimum_step, optimum_k)
                xs[point, optimum_k] += optimum_step
        if verbose >= 2:
            _log_iteration(turn, radius, prev_error, error)
    return xs, error, d_current

def pattern_search_mds(d_goal,
                       init=None,
                       n_components=2,
                       starting_radius=1.0,
                       radius_update_tolerance=1e-4,
                       sample_points=1.0,
                       explore_dim_percent=1.0,
                       max_iter=1000,
                       radius_barrier=1e-3,
                       n_jobs=1, 
                       verbose=0,
                       random_state=None,
                       prob_thr=0.2,
                       prob_step=0.05,
                       prob_init=None,
                    #    n_landmarks = None,
                       allow_bad_moves='true',
                      ):
    n_samples = d_goal.shape[0]
    n_landmarks = d_goal.shape[1]
    xs = initialize_random_points(n_samples, n_components,init,random_state) #$$$$$$$$ iswsw xreiastei na allaksw n_samples kai landmarks
    d_current = distance_matrix(xs[0:n_landmarks,:])
    points = np.arange(n_landmarks)
    radius = starting_radius
    turn = 0
    error = mse2d(d_goal, d_current)
    prev_error = np.Inf
    if verbose:
        print("Starting Error : {}".format(error))

    prob_matrix = np.full((n_samples, 2 * n_components), prob_init) #$$$$$$$$$$$$$
    
    xs, error, d_current = _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points,d_goal,d_current,xs,
    prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose)

    

    if (n_samples != n_landmarks):
        d_current = np.concatenate((d_current,distance_matrix_landmarks(xs,n_landmarks)), axis=0) 
        points = np.arange(n_landmarks,n_samples)
        radius = starting_radius
        turn = 0
        error += mse2d_landmarks(d_goal[n_landmarks:n_samples], d_current[n_landmarks:n_samples]) 
        prev_error = np.Inf
        # xs,error = _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points,d_goal,d_current,xs,
        # prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose)
         
        if verbose:
            print("Starting Error 2: {}".format(error))
        xs, error, d_current = _mds_iterations(turn,max_iter,radius,radius_barrier,error,prev_error,radius_update_tolerance,points,sample_points,d_goal,d_current,xs,
        prob_thr,prob_step,prob_matrix,explore_dim_percent,n_jobs,verbose)

        if verbose:
            print("Ending Error  {}".format(error))

    return xs, error

class MDS(BaseEstimator):

    def __init__(self,
                 n_components=2,
                 starting_radius=1.0,
                 max_iter=1000,
                 radius_barrier=1e-3,
                 explore_dim_percent=1.0,
                 sample_points=1.0,
                 radius_update_tolerance=1e-4,
                 verbose=0,
                 random_state=None,
                 n_jobs=1,
                 dissimilarity='precomputed',
                 prob_thr = 0.2,
                 prob_step = 0.05,
                 prob_init = 1.0,
                 n_landmarks = None,
                 allow_bad_moves ='true',
                 mode = 'bootstrapped',
                 dataset = None
                ):
        self.radius_update_tolerance = radius_update_tolerance
        self.sample_points = sample_points
        self.n_components = n_components
        self.starting_radius = starting_radius
        self.max_iter = max_iter
        self.radius_barrier = radius_barrier
        self.explore_dim_percent = explore_dim_percent
        self.num_epochs = 0
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = 1
        self.dissimilarity = dissimilarity
        self.allow_bad_moves = allow_bad_moves
        self.n_landmarks = n_landmarks
        self.prob_thr = prob_thr
        self.mode = mode
        self.dataset = dataset
        available_modes = ['full_search', 'bootstrapped']
        if mode in available_modes:
            self.mode = mode
        else:
            raise NotImplementedError("Mode: {} is not yet available."
                                      " Try one of: {} instead"
                                      "".format(mode, available_modes))

        if mode in ['full_search']:
            print("In full_search mode initial probability for each dimension must be 1.0 and probability step must be 0."
                    "Assigning...")
            self.prob_step = 0.
            self.prob_init = 1.0
        else:
            self.prob_step = prob_step
            self.prob_init = prob_init

    @timemethod
    def fit_transform(self, X, init=None):
        X = X.astype(np.float64)    ##o X einai tipou float64
        X = check_array(X)          # elegxos gia na einai ola ta stoixeia tou X coble( oxi Inf)
        d_goal = (X if self.dissimilarity == 'precomputed'  #X=X an exw idi to X upologismeno
                  else distance_matrix(X))   # alliws upologise to distance matrix tou X
       
       
    #TREXEI PSMDS
        if self.n_landmarks != None:
            d_goal = d_goal[:,0:self.n_landmarks].copy(order='C')  # d_goal = (N,n_landrm)
        self.embedding_, self.error_ = pattern_search_mds(
            d_goal,
            init=init,
            n_components=self.n_components,
            starting_radius=self.starting_radius,
            max_iter=self.max_iter,
            sample_points=self.sample_points,
            explore_dim_percent=self.explore_dim_percent,
            radius_update_tolerance=self.radius_update_tolerance,
            radius_barrier=self.radius_barrier,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            prob_thr=self.prob_thr,
            prob_step=self.prob_step,
            prob_init=self.prob_init,
            # n_landmarks = self.n_landmarks,
            allow_bad_moves=self.allow_bad_moves
            )            

        return self.embedding_, self.error_


    def fit(self, X, init=None):   
        self.fit_transform(X, init=init)
        return self