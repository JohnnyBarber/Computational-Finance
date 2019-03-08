#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:08:34 2019

@author: meixiangui
"""
import matplotlib.pyplot as plt
import math 
import numpy as np
import pandas as pd

def StockPrices(S0, r, sd, T, paths, steps):
    dt = T/steps
    # Generate stochastic process and its antithetic paths
    Z = np.random.normal(0, 1, paths//2 * (steps+1)).reshape(paths//2,(steps+1))
    Z_inv = -Z
    dWt = np.sqrt(dt) * Z
    dWt_inv = np.sqrt(dt) * Z_inv
    # bind the normal and antithetic Wt
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    # define the initial value of St
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*(sd**2))*dt + sd*dWt[:, i - 1])
          
    return St[:, 1:]

stock = StockPrices(36.0,0.06,0.2,0.5,1000,100)

def laguerrePloy(S,k):
    n=len(S)
    x1=np.exp(-0.5*S)
    x2=np.exp(-0.5*S)*(1-S)
    x3=np.exp(-0.5*S)*(1-2*S+S**2/2)
    x4=np.exp(-0.5*S)*(1-3*S+1.5*(S**2)-S**3/6)
    if k == 1:
       return x1
    elif k==2:
       return np.array([x1,x2]).reshape(2,n)
    elif k==3:
       return np.array([x1,x2,x3]).reshape(3,n)
    else:
       return np.array([x1,x2,x3,x4]).reshape(4,n)

def hermite(S,k):
    n=len(S)
    x1=[1]*n
    x2=2*S
    x3=4*S**2-2
    x4=8*S**3-12*S
    if k == 1:
       return x1
    elif k==2:
       return np.array([x1,x2]).reshape(2,n)
    elif k==3:
       return np.array([x1,x2,x3]).reshape(3,n)
    else:
       return np.array([x1,x2,x3,x4]).reshape(4,n)
    
def monomials(S,k):
    n=len(S)
    x1=[1]*n
    x2=S
    x3=S**2 
    x4=S**3
    if k == 1:
       return x1
    elif k==2:
       return np.array([x1,x2]).reshape(2,n)
    elif k==3:
       return np.array([x1,x2,x3]).reshape(3,n)
    else:
       return np.array([x1,x2,x3,x4]).reshape(4,n)
# LSMC process
def LSMC(S0, r, sd, T, paths, steps,K,k, methods):
    St = StockPrices(S0, r, sd, T, paths, steps)
    dt = T/steps
    # build discount factor
    discount = np.tile(np.exp(-r*dt* np.arange(1, 
                                    steps + 1, 1)), paths).reshape((paths, steps))
    # initialize payoffs matrix
    payoffs = np.zeros((paths, steps))
    payoffs[:,steps - 1] = np.maximum(K - St[:,steps - 1], 0)
    # initialize continue value matrix
    contvalue = payoffs
    # initialize stopping time matrix
    index = np.zeros((paths, steps))
    # initialize continuation value matrix
    for j in reversed(range(steps - 1)):
        # Find in the money paths
        in_money = np.where(K - St[:, j] > 0)[0]
        #  Use x which are in the money
        if methods == 'laguerre':
            X = laguerrePloy(St[in_money, j], k)
        elif methods == 'hermite':
            X = hermite(St[in_money, j], k)
        elif methods == 'monomials':
            X = monomials(St[in_money, j], k)           
        Y = payoffs[in_money, j + 1]/np.exp(r*dt)
        index[in_money,j] = 1
        # Find Least Square Beta
        A = np.dot(X, X.T)
        b = np.dot(X, Y)
        Beta = np.dot(np.linalg.pinv(A), b)
        # find full x  
        if methods == 'laguerre':
            x = laguerrePloy(St[:, j], k)  
        elif methods == 'hermite':
            x = hermite(St[:, j], k)      
        elif methods == 'monomials':
            x = monomials(St[:, j], k)  
        # find continue value
        contvalue[:,j] = np.dot(x.T, Beta)
        # update decision rule
        index[:, j] = np.where(np.maximum(K - St[:, j],[0]*paths)  - contvalue[:,j] >= 0, 1, 0)
    # Find the first occurence of 1, indicating the earlist exercise date
    first_exercise = np.argmax(index, axis = 1) 
    index = np.zeros((len(first_exercise), steps))
    index[np.arange(len(first_exercise)), first_exercise] = 1
    # update payoff matrix
    payoffs[:, j] =  np.maximum(np.maximum(K - St[:, j],[0]*paths),contvalue[:,j])
#     discount back the payoff
    option = np.mean(np.sum(index*discount*payoffs, axis = 1))
    return option
    
option= LSMC(36.0,0.06,0.2,0.5,1000,100,40.0,2,'hermite')     

S0=36.0; r=0.06; sd=0.2; T=0.5; paths=1000; steps=100; K=40.0; k=2; methods='monomials'

#    j=steps-2
#    while j>0 :     
#        payoffs[:,j] = payoffs[:,j+1]*np.exp(-r*dt)
#        in_money = np.where(payoffs[:, j] > 0)[0]
#        out_money = np.asarray(list(set(np.arange(paths)) 
#                                        - set(in_the_money_n)))
#        
#        # Find Least Square Beta
#        if methods == 'laguerre':
#            X = laguerrePloy(St[:,j], k)
#            
#        elif methods == 'hermite':
#            X = hermite(St[:,j], k)
#            
#        elif methods == 'monomials':
#            X = monomials(St[:,j], k)
#            
#        A = np.dot(X, X.T)
#        b = np.dot(X, payoffs[:,j])
#        Beta = np.dot(np.linalg.pinv(A), b)
#        
#        cont_value[in_the_money_n,i] =  np.dot(X, Beta)
          
    
#    # initialize payoffs matrix and decision matrix 
#    payoffs = np.zeros((paths, steps))
#    index = np.zeros((paths, steps))
#    for i in range(paths):
#        if K-St[i,-1]>0:
#           payoffs[i,-1]=K-St[i,-1]
#           index[i,-1]=1
#        else:
#           payoffs[i,-1]=0.0
#           index[i,-1]=0.0
    
    
    
    
    
    
    
    
    
    
    