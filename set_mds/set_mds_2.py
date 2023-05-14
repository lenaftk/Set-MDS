#!/usr/bin/env python
# coding: utf-8
#import pyximport; pyximport.install(pyimport = True)


#the one with the middle. Vlepoume oti to error stin arxi den nienai  mikrotero aapo prin, kai meta den mikrainei  kiolas kialo epilegoume lathos set eksarxis.
 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
import time
import random
import sys
import matplotlib.pyplot as plt
import imageio

from setmds_fast import (
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

# k_set matrix. this matrix will be 1d, and will show in which set the point belongs. 
# all_sets matrix.   This matrix will be 2d. in the 1st row (not 0), will have the index of the points that belong in set1 etc..
#      mallon tha arxikoipoihsw ton pinaka me -1 kai me diastaseis (k,k)
# xs[n+samples +kj] and i will find the corellated point.

def _radius_update(radius, error, prev_error, tolerance=1e-4):
    if error >= prev_error or prev_error - error <= error * tolerance:
        return radius * 0.5
    return radius

def add_randompoint(xs):
    random_state = random.randint(1,10000)
    random_state = check_random_state(random_state)
    random_point = random_state.rand(1,xs.shape[1])
    xs[xs.shape[1]-1] = random_point 

    return xs

def add_new_point(xs):
    new_line=np.zeros((1,xs.shape[1]))
    xs_ = np.concatenate((xs,new_line)) 
    print(xs_)
    return xs_

def change_randompoint(xs):
    random_state = random.randint(1,10000)
    random_state = check_random_state(random_state)
    random_point = random_state.rand(1,xs.shape[1])
    xs[xs.shape[0]-1,:]= random_point
    return xs

def add_midpoint(xs):
    midpoint = np.zeros((1,xs.shape[1]))
    for ii in range(0,xs.shape[1]):
        midpoint[0][ii] = xs[:,ii].sum()
    midpoint = midpoint/xs.shape[0]
    xs_ = np.concatenate((xs,midpoint)) 
    return xs_

def _point_sampling(points, keep_percent=1.0, turn=-1, recalculate_each=-1):
    if keep_percent > 1.0 or 1.0 - keep_percent < 1e-5:
        return points
    if turn > 0 and recalculate_each > 0 and turn % recalculate_each == 0:
        return points
    keep = int(points.shape[0] * keep_percent)
    return np.random.choice(points, size=keep, replace=False)

def _matrix_initialization(n_samples,k,d_current):
    set_point_error = np.zeros(n_samples) #pinakas pou tha krataei to error, kai tha prepei na epilegw to kalitero. Na ginei max value kai oxi miden.
    k_set = np.negative(k) #min_value, which set
    d_current_nk = np.zeros((n_samples+k,n_samples+k))
    for ii in range(n_samples):
        for jj in range(ii+1,n_samples):
            d_current_nk[ii][jj] = d_current[ii][jj]
            d_current_nk[jj][ii] = d_current[ii][jj]
    return set_point_error, k_set, d_current_nk

def change_point_to_distance_matrix(xs,d_current):
    xs_columns = xs.shape[1]
    xs_rows = xs.shape[0]
    for ii in range(xs_rows):
        tmp=0
        for jj in range(xs_columns):
            diff = xs[ii][jj]-xs[xs_rows-1][jj]
            tmp += diff*diff
        tmp = np.sqrt(tmp)
        d_current[ii][xs_rows-1] = tmp
        d_current[xs_rows-1][ii] = tmp
    return d_current

def init_point_to_distance_matrix(d_current):
     #insert new row & column
    a= np.zeros((d_current.shape[0]+1,1))
    b= np.zeros((d_current.shape[0]))
    d_current = np.insert(d_current,d_current.shape[0], b, axis= 0)
    d_current = np.hstack((d_current, a))
    return d_current

def update_d_sets_matrix(d_current,d_sets,point,sets):
 #   print("--d_current",d_current)
 #   print("--d_sets",d_sets)
 #   print(point)
   # print(d_current.shape[0])

    for ii in range(d_current.shape[0]):
        if d_current[point][ii] < d_sets[sets[point]][sets[ii]]:
            d_sets[sets[point]][sets[ii]] = d_current[point][ii]
            d_sets[sets[ii]][sets[point]] = d_current[point][ii]
   # print("--d_sets",d_sets)
    return d_sets

def best_pertub(sets,
                d_sets,
                xs,
                radius,
                d_current,
                d_goal,
                point,
                turn,
                percent,
                n_jobs):
                print("se periptwsi anagkis...")


def set_mds(xs, d_current, d_goal, error, k, n_samples, savefig): 
    savefig_number=0
    init_error = error
    d_current_initial=d_current.copy()
    radius_update_tolerance=1e-20
    max_iter=1000
    radius_barrier=1e-10
    sample_points=1.0
    explore_dim_percent=1.0
    n_jobs=1
    starting_radius=3.0
    #define first n sets
    sets=[]
    for ii in range(n_samples):
        sets.append(int(ii))
    #initialize d_sets
    d_sets=d_current.copy()
    #start spliting
    for kj in range(k):
        print("START!!!", kj)
        # print("Now we are in the k= ",kj)
        # temp_set_error=[]

        #xs = add_randompoint(xs) ##or midpoint
        xs = add_midpoint(xs) ##or midpoint 
        #xs = add_new_point(xs)
        #xs = change_randompoint(xs)
        d_current = init_point_to_distance_matrix(d_current)
        d_current = change_point_to_distance_matrix(xs,d_current)
        print("xs= \n", xs)
        print(d_current)
        sets.append(int(0))

        temp_set_error=[]
        for set in range(n_samples):
            sets[n_samples + kj]=set
            #print("kj, SET IS",kj,set,sets)
            temp_d_sets = d_sets.copy() #arxikopoiw pali ton pinaka
            temp_set_error.append(error) #arxikopoiw to error
            for ii in range(n_samples+kj+1):
                #print("1.\n",d_current)
                #print("2.\n",d_sets)
                if(d_current[n_samples+kj][ii]  < d_sets[set][sets[ii]]):
                    #print("yes",d_current[n_samples+kj][ii],d_sets[set][sets[ii]],n_samples+kj,ii)
                    temp_d_sets[set][sets[ii]] = d_current[n_samples+kj][ii]
                    temp_d_sets[sets[ii]][set] = d_current[n_samples+kj][ii]
            temp_set_error[set] = error - mse1d(d_goal[set], d_sets[set]) + mse1d(d_goal[set], temp_d_sets[set]) 
            error_temp = mse2d(d_goal,temp_d_sets)

            # print("1.\n",d_goal)
            # print("2.\n",d_sets)
            # print("3.\n",temp_d_sets)
            # print("temp_set_error[set]", set, temp_set_error)
            
        #choose the set
        inactive=0
        while inactive == 0: 
            winner_set = temp_set_error.index(min(temp_set_error))
            print("winner error=", temp_set_error[winner_set], "from",temp_set_error, "prev-error= ", error, "winner_set= ",winner_set)
            sets[n_samples + kj] = winner_set
            error = temp_set_error[winner_set]

            ## update distance matrix
            for ii in range(n_samples+kj+1):  #[0,1,2,0,1]
                if(d_current[n_samples+kj][ii]  < d_sets[winner_set][sets[ii]]):
                        d_sets[winner_set][sets[ii]] = d_current[n_samples+kj][ii]
                        d_sets[sets[ii]][winner_set] = d_current[n_samples+kj][ii]
            print("winner_set & updated distance matrix", winner_set)
            print(d_sets)
                
                ##

            points = np.arange(n_samples+kj+1)

            radius=starting_radius
            turn=0
            prev_error = np.Inf
            inactive= 0
            while turn <= max_iter and radius > radius_barrier:      #oso exw akoma epanalipseis& i aktina ine sta oria
                # print("turn = ", turn)
                turn += 1    #+1 epanalipsi
                radius = _radius_update(  #update tin aktina
                    radius, error, prev_error, tolerance=radius_update_tolerance)
                # print(prev_error,error)
                prev_error = error
                filtered_points = _point_sampling(points, keep_percent=sample_points)  #epilogi simeiwn
                for point in filtered_points:    #gia kathe shmeio apo ta epilegmena simeia
                    point_error = mse1d(d_goal[sets[point]], d_sets[sets[point]])
                    # print("d_goal ", d_goal[sets[point]], "\n d_sets",d_sets[sets[point]])
                    optimum_error, optimum_k, optimum_step = best_pertubation(
                        n_samples,
                        sets,
                        d_sets,
                        xs,
                        radius,
                        d_current,
                        d_goal,
                        point,
                        turn,
                        percent=explore_dim_percent,
                        n_jobs=n_jobs
                    )
                    # print("ok ", point,point_error,optimum_error,optimum_k,optimum_step)
                    # if(point_error == optimum_error):
                        # print("oulala!!! \n \n \n"  )
                    # print("pointerror= ", point_error,"optimum_error= ",optimum_error, "error", error)
                    if (point_error > optimum_error):
                        inactive = 1
                        print("yes")
                        error -= (point_error - optimum_error)
                        d_current = update_distance_matrix(
                            xs, d_current, point, optimum_step, optimum_k)
                    # print(point, d_current)
                        d_sets = update_d_sets_matrix(
                            d_current,d_sets,point, sets)
                        xs[point, optimum_k] += optimum_step
                    if inactive == 0:
                        temp_set_error[winner_set]=np.inf
        
                # Preparing dataset
                if savefig == 'true':
                    x=xs[:,0]
                    y=xs[:,1]
                    # text = ["0","1","2","0","1","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2"]

                    text = [sets[i] for i in range(len(xs))]  
                    # plotting scatter plot
                    plt.scatter(x, y)
                    ax = plt.gca()
                    plt.title("SET MDS. Turn= " + str(turn) +  " Point= " + str(point) )
                    ax.set_xlim([-11, 25])
                    ax.set_ylim([-10, 10])
                    
                    # Loop for annotation of all points
                    for i in range(len(x)):
                        plt.annotate(text[i], (x[i]+0.002, y[i] + 0.002))
    
                    ## adjusting the scale of the axes
                    savefig_number+=1
                    plt.savefig(f'./savefigs/set-mds-{savefig_number}.png')
                    plt.close()

 


    # print(sets, init_error, error)
    # if(error<init_error):
    #     print("yes")
    # print("d_current_initial \n", d_current_initial)
    # print(" d_current \n", d_current)
    # print("d_sets \n",d_sets)
    print(sets)
    print(xs)

            
