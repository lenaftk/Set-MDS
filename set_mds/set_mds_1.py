#!/usr/bin/env python
# coding: utf-8
#import pyximport; pyximport.install(pyimport = True)

### periexei prin tin allagi me to inactive.
#
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
from scipy.spatial.distance import cdist
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

def find_midpoint(xs):
    midpoint = np.zeros((1,xs.shape[1]))
    for ii in range(0,xs.shape[1]):
        midpoint[0][ii] = xs[:,ii].sum()
    midpoint = midpoint/xs.shape[0]
    return midpoint[0]

def generate_random_point(xs):
    min_coord = -1 #min(min(row) for row in xs)
    max_coord = 1 #max(max(row) for row in xs)
    point = []
    for j in range(xs.shape[1]):
        coord = random.uniform(min_coord, max_coord)
        point.append(coord)
    return np.array(point)


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

def calculate_d_sets_matrix(d_current,d_sets,point,sets):
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

def points_creator(xs):
    points=[]
    #add_midpoint
    points.append(find_midpoint(xs))
    #generate 5 random points
    for ii in range(5):
        points.append(generate_random_point(xs))
    return points

def calculate_set_mds_error(error,xs,sets,i,d_current):
    d_current(xs,xs)

    # N = d.shape[0]
    # K = x.shape[0]
    # dim = x.shape[1]
    # E = 0
    # valid_x_ind = np.nonzero(ind)
    # iind = np.nonzero(ind == i)[0]
    # tmp = np.zeros((K,))        
    # if dim == 2:
    #     for k in valid_x_ind[0]:
    #         j = k
    #         tmp[j] = np.min(np.sqrt((x[iind, 0] - x[j, 0]) ** 2 + (x[iind, 1] - x[j, 1]) ** 2))
            
    #     for j in range(1,N):
    #         jind = np.nonzero(ind == j)[0]
    #         E += (np.min(tmp[jind]) - d[i-1, j]) ** 2

def count_elements(lst):
    counts = {}
    for x in lst:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    print(counts)
    return counts

def compare_elements(dict1,dict2,k,n,file):
    true_ = 0
    false_ = 0
    correct_split=0
    for key, value in dict1.items():
        if dict1.get(key) == dict2.get(key):
            print(f"{key} -> {value}", " True")
            file.write("{} -> {}  True \n".format(key,value))
            if dict1.get(key) > 1:
                print("correct split")
                correct_split+=1
            true_ += 1
        else:
            print(f"{key} -> Algorithm found {value}, but it was ", dict2.get(key))
            file.write("{} -> Algorithm found {}, but it was {} \n".format(key,value,dict2.get(key)))
            false_ +=1
            if dict2.get(key) >1 and dict1.get(key)>1:
                if dict2.get(key)>dict1.get(key):
                    correct_split+=dict2.get(key)-dict1.get(key)
                else: 
                    correct_split+=dict2.get(key)-1
                    

    print("\nTRUE=", true_, ", FALSE=",false_, "PERCENTAGE= ",(true_/(false_+true_))*100,"%", ", k(splits)=", k, ", n(sets)=", n)
    file.write("\nTRUE= {}, FALSE= {} ,PERCENTAGE= {}%\n".format(true_,false_,(true_/(false_+true_))*100))
    file.write("\ncorrect_splits= {},PERCENTAGE= {}% \n".format(correct_split,correct_split/k *100))
    file.write("\nk(splits)={}, n(sets)= {} \n".format(k,n))

def find_different_values(dict1, dict2):
    # find key-value pairs in dict1 that are not in dict2
    diff1 = {k: dict1[k] for k in set(dict1) - set(dict2) if dict1[k] != dict2.get(k)}

    # find key-value pairs in dict2 that are not in dict1
    diff2 = {k: dict2[k] for k in set(dict2) - set(dict1) if dict2[k] != dict1.get(k)}

    # return the combined set of different key-value pairs
    return {**diff1, **diff2}

def set_mds(xs, d_current, d_goal, error, k, n_samples, savefig,init_sets): 
### initializations and parameters
    savefigs_number=0
    radius_update_tolerance=1e-20
    max_iter=100
    radius_barrier=1e-10
    sample_points=1.0
    explore_dim_percent=1.0
    n_jobs=1
    starting_radius=2.5
 ###
    #define first n sets
    sets=[]
    
    for ii in range(n_samples):
        sets.append(int(ii))
    #initialize d_sets
    d_sets=d_current.copy()

    #start spliting
    print("MDS has finished with Error:", error)
    for kj in range(k):
        pr_error=error
        # error = np.Inf
        print("START!!! \n kj= ", kj, " \n error= ", error)
        #initialization for new point
        xs = add_new_point(xs)
        d_current = init_point_to_distance_matrix(d_current)
        sets.append(int(-1))
        #print("First \n", d_current) 

        # v= 0.03*np.cos(np.pi/4)
        # vector = [[v, v], [-v,-v],[-v, v], [v,-v]]
        # Evec = np.zeros((len(vector)))
        # for i in range(xs.shape[0]-1):   #spaw ola ta simeia 
        #     sets[-1]=sets[i]             #orise to trexon set
        #     xorg = xs[i].copy()          #apothikeuw to simeio, gia epanafora tou argotera
        #     for j in range(len(vector)):
        #         xs[i] = xs[i] + vector[j]            #kounima trexon simeiou
        #         xs[n_samples+kj] = xs[i] - vector[j]           #kounima kainouriou simeiou
        #         Evec[j] = calculate_set_mds_error(error,xs,sets,i,d_current)   
        #         print("outside\n", d_current)     
        #         xs[i]=xorg                           #epanafora

            

        
        #first_experiment_with_random_points()
        points_to_try_as_splits = points_creator(xs)
        temp_set_error=[]
        saved_errors_per_point=[]
        saved_winner_set_per_point=[]
        for label in range(n_samples):
            temp_set_error.append(error) #arxikopoiw to error
        for count,split_point in enumerate(points_to_try_as_splits):
            xs[xs.shape[0]-1,:]= split_point
            #print("xs=", xs)
            for label in range(n_samples):
                sets[n_samples + kj]=label
                temp_d_sets = d_sets.copy() #arxikopoiw pali ton pinaka
                temp_d_current = change_point_to_distance_matrix(xs,d_current,xs.shape[0]-1)
                for ii in range(n_samples+kj+1):
                    if(temp_d_current[n_samples+kj][ii] < d_sets[label][sets[ii]]):
                        temp_d_sets[label][sets[ii]] = temp_d_current[n_samples+kj][ii]
                        temp_d_sets[sets[ii]][label] = temp_d_current[n_samples+kj][ii]
                #print("set= ",set_, len(temp_set_error),temp_set_error)
                #temp_set_error[label] = pr_error - mse1d(d_goal[label], d_sets[label]) + mse1d(d_goal[label], temp_d_sets[label]) 
                temp_set_error[label] = mse2d(d_goal,temp_d_sets)
                
                
            #choose the set
            winner_set = temp_set_error.index(min(temp_set_error))
            #print("Count: ", count, "winner set is ",winner_set)
            # if kj==0:
            #     winner_set = 0
            # if kj==1:
            #     winner_set = 1
            # if kj==2:
            #     winner_set = 2

            #print("winner error=", temp_set_error[winner_set], "from",temp_set_error, "prev-error= ", error, winner_set, sets)
            #sets[n_samples + kj] = winner_set
            error = temp_set_error[winner_set]
            saved_errors_per_point.append(error)
            saved_winner_set_per_point.append(winner_set)
        
        winner_point = saved_errors_per_point.index(min(saved_errors_per_point))
        winner_set = saved_winner_set_per_point[winner_point]
        sets[n_samples + kj] = winner_set
        #print("Final Winner set", winner_set)
        
    # update distance matrix
        xs[xs.shape[0]-1,:]= points_to_try_as_splits[winner_point]
        d_current = change_point_to_distance_matrix(xs,d_current,xs.shape[0]-1)
        for ii in range(n_samples+kj+1):  #[0,1,2,0,1]
            if(d_current[n_samples+kj][ii]  < d_sets[winner_set][sets[ii]]):
                    d_sets[winner_set][sets[ii]] = d_current[n_samples+kj][ii]
                    d_sets[sets[ii]][winner_set] = d_current[n_samples+kj][ii]
        #print("winner_set & updated distance matrix", winner_set)
           # print(d_sets)
        #
        #print("I have found it!")

        points = np.arange(n_samples+kj+1)

        radius=starting_radius
        turn=0
        prev_error = np.Inf
        inactive= 0
        while turn <= max_iter and radius > radius_barrier:      #oso exw akoma epanalipseis& i aktina ine sta oria
            #print("Turn = ", turn, "Error= ",error)
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
                    #print("yes")
                    error -= (point_error - optimum_error)
                    d_current = update_distance_matrix(
                        xs, d_current, point, optimum_step, optimum_k)
                   # print(point, d_current)
                    d_sets = update_d_sets_matrix(
                        d_current,d_sets,point, sets)
                    xs[point, optimum_k] += optimum_step


        
                # Preparing dataset
                if savefig == 'true':
                    x=xs[:,0]
                    y=xs[:,1]
                    text = [sets[i] for i in range(len(xs))]  
                    # plotting scatter plot
                    plt.scatter(x, y)
                    ax = plt.gca()
                    plt.title("SET MDS. Turn= " + str(turn) +  " Point= " + str(point) )
                    ax.set_xlim([-1.2, 1.2])
                    ax.set_ylim([-1.2, 1.2])
                    
                    # Loop for annotation of all points
                    for i in range(len(x)):
                        plt.annotate(text[i], (x[i]+0.002, y[i] + 0.002))
                    # Separate the first k points from the rest
                    blue_points = xs[:n_samples]
                    red_points = xs[n_samples:]
    
                    # Create the scatter plot with blue and red points
                    plt.scatter([p[0] for p in blue_points], [p[1] for p in blue_points], color='blue')
                    plt.scatter([p[0] for p in red_points], [p[1] for p in red_points], color='red')
    
                    ## adjusting the scale of the axes
                    savefigs_number+=1
                    plt.savefig(f'./savefigs/set-mds-{savefigs_number}.png')
                    plt.close()


    # print(sets, init_error, error)
    # if(error<init_error):
    #     print("yes")
    # print("d_current_initial \n", d_current_initial)
    # print(" d_current \n", d_current)
    # print("d_sets \n",d_sets)
    #print(xs)

    print(d_goal-d_sets)
 
    print("error",error)
    dict1=count_elements(sets)
    dict2=count_elements(init_sets)
    file = open("./experiments/25 May/results n_sets={} k={}.txt".format(n_samples,k), "w")
    
    compare_elements(dict1,dict2,k,n_samples,file)
    sets.sort()
    init_sets.sort()
    print("\n algo sets",sets)
    print(" init sets",init_sets)
    file.write("\nalgo sets {} \n".format(sets))
    file.write("init sets {} ".format(init_sets))

    return sets
  
            
