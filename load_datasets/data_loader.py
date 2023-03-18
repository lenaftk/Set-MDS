import os
import sys
import numpy as np
from numpy import genfromtxt
from load_datasets.load_embeddings import load_word_vectors
from gentlemandata.gentlemandata.shapes import DataBuilder

'''
Load following datasets with data_name: 'Mnist_digits', 'Glove', 'Pen_digits' , 'Coil_20'
Load following synthetic datasets with data_name: data_name in {
                                    'sphere',
                                    'cut-sphere',
                                    'ball',
                                    'random',
                                    'spiral',
                                    'spiral-hole',
                                    'swissroll',
                                    'swisshole',
                                    'toroid-helix',
                                    's-curve',
                                    'punctured-sphere',
                                    'gaussian',
                                    'clusters-3d',
                                    'twin-peaks',
                                    'corner-plane'
                                }
don't forget to set the path for each dataset :
'''
#set paths for each dataset
mnist_digits_data_path = '/home/lena/diplomatiki/06_datasets/01_mnist_digits/images.csv'
mnist_digits_labels_path = '/home/lena/diplomatiki/06_datasets/01_mnist_digits/labels.csv'
glove_path = '/home/lena/diplomatiki/06_datasets/06_glove/glove.42B.300d.txt'
pen_digits_path = '/home/lena/diplomatiki/06_datasets/02_pen_digits/pendigits.tra'
coil20_path = '/home/lena/diplomatiki/06_datasets/03_coil20/coil-20-proc/output.csv'
men_path = '/home/lena/diplomatiki/06_datasets/07_MEN-word-similarity/MEN/MEN_dataset_lemma_form_full.json'



def data_loader(dataset_name=None,  #
                dim=3,
                distance='geodesic',
                npoints = 1000,
                n_neighbors = 66,
                noise_std = 0, 
                memmap = False
                ):
    print("Hey! This is data loader ... ")
    dataset_data = None
    if dataset_name == 'Mnist_digits':
        dataset_data = genfromtxt(mnist_digits_data_path, delimiter=',', max_rows=npoints)
        dataset_labels = genfromtxt(mnist_digits_labels_path, delimiter=',', max_rows=npoints)
        color = dataset_labels.astype(int)
    
    elif dataset_name == 'Glove':
        word2idx, dataset_data = load_word_vectors(glove_path, take=npoints)
        color = ['blue']

    ### Sto pen_digits paizei na thelei diaforetiki metaxeirisi twn data. Check it.
    elif dataset_name == 'Pen_digits':
        dataset_data = genfromtxt(pen_digits_path, delimiter=',', max_rows=npoints, usecols=range(0,16))
        dataset_labels = genfromtxt(pen_digits_path, delimiter=',', max_rows=npoints, usecols=(16))
        color = dataset_labels.astype(int)

        #coil exei 1440 fotossss
    elif dataset_name == 'Coil_20':
        dataset_data = genfromtxt(coil20_path, delimiter=',', max_rows=npoints)
        num = 0
        color = []
        for ii in range(0,int(npoints/72)):
            num = num + 3
            for jj in range(0,72):
                color.append(num)
        for ii in range(0, npoints % 72):
                color.append(num+3)

        
    
    if dataset_data is None:  # this is for gentlemandata
        print("I WILL PRINT GENTLEMANDATA")
        data_type = dataset_name
        dim=3
        xs, d_goal, color = (DataBuilder()
                .with_dim(dim)
                .with_distance(distance)
                .with_noise(noise_std)
                .with_npoints(npoints)
                .with_neighbors(n_neighbors)
                .with_type(data_type)
                .with_memmap(memmap)
                .build())
        return xs, d_goal, color


    else:    
    #create NON synthetic data
        dim = dataset_data.shape[1]
        data_type = 'real'
        xs, d_goal, color = (DataBuilder()    
                                        .with_dim(dim)
                                        .with_distance(distance)
                                        .with_noise(noise_std)
                                        .with_npoints(npoints)
                                        .with_neighbors(n_neighbors)
                                        .with_points(dataset_data)
                                        .with_type(data_type)
                                        .with_memmap(memmap)
                                        .build())
        return dataset_data, d_goal, color