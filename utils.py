# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:25:20 2021

@author: klepy
"""
import numpy as np
import torch.nn as nn
import math

def align(arr,M,K):
    num_train = arr.shape[0]
    temp = np.zeros([num_train,M,K+1])
    for i in range(num_train):
        for k in range(K+1):
            v = arr[i,:]
            vec = v[M*k:M*(k+1)]
            temp[i,:,k] = vec
    return temp

def prepare(train_real,train_imag):
    M = 10
    K = (train_real.shape[1]//(2*M)) -1
    Ntr = train_real.shape[0]
    # Real part
    x_real = np.asarray(train_real.iloc[:,:M*(K+1)])
    y_real = np.asarray(train_real.iloc[:,M*(K+1):])
    # Imag part
    x_imag = np.asarray(train_imag.iloc[:,:M*(K+1)])
    y_imag = np.asarray(train_imag.iloc[:,M*(K+1):])
    
    #%% 
    x_r = align(x_real,M,K)
    x_i = align(x_imag,M,K)
    y_r = align(y_real,M,K)
    y_i = align(y_imag,M,K)
    
    x_cat = np.zeros([Ntr,2,M,K+1])
    y_cat = np.zeros([Ntr,2,M,K+1])
    
    x_cat[:,0,:,:] = x_r[:]
    x_cat[:,1,:,:] = x_i[:]
    y_cat[:,0,:,:] = y_r[:]
    y_cat[:,1,:,:] = y_i[:]
    return x_cat, y_cat

def stat(x):
    mean_r = []
    mean_i = []
    std_r = []
    std_i = []
    for i in range(len(x)):
        mean_r.append(np.mean(x[i,0,:,:]))
        std_r.append(np.std(x[i,0,:,:]))
        mean_i.append(np.mean(x[i,1,:,:]))
        std_i.append(np.std(x[i,1,:,:]))
    m_r = sum(mean_r) / len(mean_r)
    m_i = sum(mean_i) / len(mean_i)
    std_r = sum(std_r) / len(std_r)
    std_i = sum(std_i) / len(std_i)
    return m_r,m_i,std_r,std_i

def standard(x,m_r,m_i,std_r,std_i):
    x_n = np.zeros_like(x)
    for i in range(len(x)):
        x_n[i,0,:,:] = (x[i,0,:,:] - m_r) / std_r
        x_n[i,1,:,:] = (x[i,1,:,:] - m_i) / std_i
    return x_n

def minmax(x):
    x_n = np.zeros_like(x)
    for i in range(len(x)):
        x_n[i,0,:,:] = (x[i,0,:,:] - np.amin(x[i,0,:,:])) / (np.amax(x[i,0,:,:])-np.amin(x[i,0,:,:]))
        x_n[i,1,:,:] = (x[i,1,:,:] - np.amin(x[i,1,:,:])) / (np.amax(x[i,1,:,:])-np.amin(x[i,1,:,:]))   
    return x_n

def minmax_col(x):
    x_n = np.zeros_like(x)
    for i in range(len(x)):
        x_n[i,0,:,0] = (x[i,0,:,0] - np.amin(x[i,0,:,0])) / (np.amax(x[i,0,:,0])-np.amin(x[i,0,:,0]))
        x_n[i,0,:,1::] = (x[i,0,:,1::] - np.amin(x[i,0,:,1::])) / (np.amax(x[i,0,:,1::])-np.amin(x[i,0,:,1::]))
        x_n[i,1,:,0] = (x[i,1,:,0] - np.amin(x[i,1,:,0])) / (np.amax(x[i,1,:,0])-np.amin(x[i,1,:,0]))
        x_n[i,1,:,1::] = (x[i,1,:,1::] - np.amin(x[i,1,:,1::])) / (np.amax(x[i,1,:,1::])-np.amin(x[i,1,:,1::]))
    return x_n
       
def weigths_init_kaiming(lyr):
    """
    Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=0.1)
        nn.init.constant_(lyr.bias.data, 0.0)


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)
