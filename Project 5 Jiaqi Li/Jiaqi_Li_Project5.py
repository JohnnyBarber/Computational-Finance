# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:00:20 2019

@author: Jiaqi Li
"""
import numpy as np
import time
import matplotlib.pyplot as plt

#def f_S_path(n,s0,r,dt,sigma,path):
#    All = np.zeros(shape = (path,n+1))
#    All[:,0] = [s0]*path
#    for j in range(int(path/2)):
#        Z1 = np.array(np.random.normal(0,1,n))
#        Z2 = np.array([-z for z in Z1])
#        dW1 = np.sqrt(dt)*Z1
#        dW2 = np.sqrt(dt)*Z2
#        for i in range(n):
#                x1 = np.exp((r-0.5*sigma**2)*dt+sigma*dW1[i])
#                All[2*j,i+1] = All[2*j,i]*x1
#                x2 = np.exp((r-0.5*sigma**2)*dt+sigma*dW2[i])
#                All[2*j+1,i+1] = All[2*j+1,i]*x2
#    return All

def StockPrices(S0, r, sd, T, paths, steps):
    dt = T/steps
    # Generate stochastic process and its antithetic paths
    Z = np.random.normal(0, 1, paths//2 * (steps)).reshape(paths//2,(steps))
    Z_inv = -Z
    dWt = np.sqrt(dt) * Z
    dWt_inv = np.sqrt(dt) * Z_inv
    # bind the normal and antithetic Wt
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    # define the initial value of St
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (steps):
        St[:, i+1] = St[:, i]*np.exp((r - 1/2*(sd**2))*dt + sd*dWt[:, i])
    return St[:,1:]

def f_reg(x,k,method):
    x = np.array(x)
    n = len(x)
    if method == "Laguerre":
        if k == 1:
            R1 = np.exp(-x/2)
            return R1
        elif k == 2:
            R1 = np.exp(-x/2)
            R2 = np.exp(-x/2)*(1-x)
            return np.array([R1,R2]).reshape(2,n)
        elif k == 3:
            R1 = np.exp(-x/2)
            R2 = np.exp(-x/2)*(1-x)
            R3 = np.exp(-x/2)*(1-2*x+x**2/2)
            return np.array([R1,R2,R3]).reshape(3,n)
        elif k == 4:
            R1 = np.exp(-x/2)
            R2 = np.exp(-x/2)*(1-x)
            R3 = np.exp(-x/2)*(1-2*x+x**2/2)
            R4 = np.exp(-x/2)*(1-3*x+3*x**2/2-x**3/6)
            return np.array([R1,R2,R3,R4]).reshape(4,n)
    if method == "Hermite":
        if k == 1:
            R1 = [1]*n
            return R1
        elif k == 2:
            R1 = [1]*n
            R2 = 2*x
            return np.array([R1,R2]).reshape(2,n)
        elif k == 3:
            R1 = [1]*n
            R2 = 2*x
            R3 = 4*x**2-2
            return np.array([R1,R2,R3]).reshape(3,n)
        elif k == 4:
            R1 = [1]*n
            R2 = 2*x
            R3 = 4*x**2-2
            R4 = 8*x**3-12*x
            return np.array([R1,R2,R3,R4]).reshape(4,n)
    if method == "Monomials":
        if k == 1:
            R1 = [1]*n
            return R1
        elif k == 2:
            R1 = [1]*n
            R2 = x
            return np.array([R1,R2]).reshape(2,n)
        elif k == 3:
            R1 = [1]*n
            R2 = x
            R3 = x**2
            return np.array([R1,R2,R3]).reshape(3,n)
        elif k == 4:
            R1 = [1]*n
            R2 = x
            R3 = x**2
            R4 = x**3
            return np.array([R1,R2,R3,R4]).reshape(4,n)

# LSMC process
def f_APLS(S0, r, sd, T, paths, steps, K, k, methods):
    dt = T/steps
    St = StockPrices(S0, r, sd, T, paths, steps)
    # initialize payoffs matrix
    payoffs = np.zeros((paths, steps))
    payoffs[:,steps - 1] = np.maximum(K - St[:,steps - 1], 0)
    # initialize stopping time matrix
    index = np.zeros((paths, steps))
    index[:,steps-1] = np.where(payoffs[:,steps - 1]> 0, 1, 0)
    discI = np.array(index[:,steps-1])
    Ot = np.array(np.maximum(K - St[:,steps-1],0))
    # initialize continuation value matrix
    for j in reversed(range(steps - 1)):
        payoffs[:,j] = np.maximum(K - St[:, j],0)
        # Find in the money paths
        In = np.where(Ot*np.exp(-r*dt*discI) > 0)[0]
        #  Use x which are in the money
        X = f_reg(St[In, j], k, methods)         
        Y = np.array(Ot*np.exp(-r*dt*discI))[In]
        # Find Least Square Beta
        A = np.dot(X, X.T)
        b = np.dot(X, Y)
        Beta = np.dot(np.linalg.inv(A), b)
        # find full x  
        x = f_reg(St[:, j], k, methods) 
        # find continue value
        expected = np.dot(x.T, Beta)
        # update decision rule
        index[:, j] = np.where(np.array(payoffs[:,j]) - np.array(expected) > 0, 1, 0)
        for l in range(paths):
            if index[l,j] == 1:
                discI[l] = 1
                Ot[l] = K - St[l,j]
            elif index[l,j] == 0:
                discI[l] += 1
        payoffs[:,j] = np.maximum(np.array(payoffs[:,j]),np.maximum(np.array(expected),0))
    # Find the first occurence of 1, indicating the earlist exercise date
    first_exercise = np.argmax(index, axis = 1) 
    index = np.zeros(shape = (paths, steps))
    index[np.arange(paths), first_exercise] = 1
    option = 0
    temp = index*payoffs
    for q in range(steps):
        option += np.mean(temp[:,q])*np.exp(-r*dt*(q+1))
    return option

#parameters-------------------------------------------------------
N = 10000
K = 40
h = 100
r = 0.06
sigma = 0.2
S01 = [36,40,44]
T1 = [0.5,1,2]
k1 = [2,3,4]

start = time.time()
#a----------------------------------------------------------------
method = "Laguerre"
payoffa = np.zeros((3,9))
payoffa[0,0] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[0],method)
payoffa[0,1] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[1],method)
payoffa[0,2] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[2],method)
payoffa[0,3] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[0],method)
payoffa[0,4] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[1],method)
payoffa[0,5] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[2],method)
payoffa[0,6] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[0],method)
payoffa[0,7] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[1],method)
payoffa[0,8] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[2],method)
payoffa[1,0] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[0],method)
payoffa[1,1] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[1],method)
payoffa[1,2] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[2],method)
payoffa[1,3] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[0],method)
payoffa[1,4] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[1],method)
payoffa[1,5] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[2],method)
payoffa[1,6] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[0],method)
payoffa[1,7] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[1],method)
payoffa[1,8] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[2],method)
payoffa[2,0] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[0],method)
payoffa[2,1] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[1],method)
payoffa[2,2] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[2],method)
payoffa[2,3] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[0],method)
payoffa[2,4] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[1],method)
payoffa[2,5] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[2],method)
payoffa[2,6] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[0],method)
payoffa[2,7] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[1],method)
payoffa[2,8] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[2],method)

#b----------------------------------------------------------------
method = "Hermite"
payoffb = np.zeros((3,9))
payoffb[0,0] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[0],method)
payoffb[0,1] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[1],method)
payoffb[0,2] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[2],method)
payoffb[0,3] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[0],method)
payoffb[0,4] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[1],method)
payoffb[0,5] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[2],method)
payoffb[0,6] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[0],method)
payoffb[0,7] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[1],method)
payoffb[0,8] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[2],method)
payoffb[1,0] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[0],method)
payoffb[1,1] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[1],method)
payoffb[1,2] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[2],method)
payoffb[1,3] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[0],method)
payoffb[1,4] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[1],method)
payoffb[1,5] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[2],method)
payoffb[1,6] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[0],method)
payoffb[1,7] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[1],method)
payoffb[1,8] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[2],method)
payoffb[2,0] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[0],method)
payoffb[2,1] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[1],method)
payoffb[2,2] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[2],method)
payoffb[2,3] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[0],method)
payoffb[2,4] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[1],method)
payoffb[2,5] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[2],method)
payoffb[2,6] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[0],method)
payoffb[2,7] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[1],method)
payoffb[2,8] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[2],method)

#c----------------------------------------------------------------
method = "Monomials"
payoffc = np.zeros((3,9))
payoffc[0,0] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[0],method)
payoffc[0,1] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[1],method)
payoffc[0,2] = f_APLS(S01[0],r,sigma,T1[0],N,h,K,k1[2],method)
payoffc[0,3] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[0],method)
payoffc[0,4] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[1],method)
payoffc[0,5] = f_APLS(S01[0],r,sigma,T1[1],N,h,K,k1[2],method)
payoffc[0,6] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[0],method)
payoffc[0,7] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[1],method)
payoffc[0,8] = f_APLS(S01[0],r,sigma,T1[2],N,h,K,k1[2],method)
payoffc[1,0] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[0],method)
payoffc[1,1] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[1],method)
payoffc[1,2] = f_APLS(S01[1],r,sigma,T1[0],N,h,K,k1[2],method)
payoffc[1,3] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[0],method)
payoffc[1,4] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[1],method)
payoffc[1,5] = f_APLS(S01[1],r,sigma,T1[1],N,h,K,k1[2],method)
payoffc[1,6] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[0],method)
payoffc[1,7] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[1],method)
payoffc[1,8] = f_APLS(S01[1],r,sigma,T1[2],N,h,K,k1[2],method)
payoffc[2,0] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[0],method)
payoffc[2,1] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[1],method)
payoffc[2,2] = f_APLS(S01[2],r,sigma,T1[0],N,h,K,k1[2],method)
payoffc[2,3] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[0],method)
payoffc[2,4] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[1],method)
payoffc[2,5] = f_APLS(S01[2],r,sigma,T1[1],N,h,K,k1[2],method)
payoffc[2,6] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[0],method)
payoffc[2,7] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[1],method)
payoffc[2,8] = f_APLS(S01[2],r,sigma,T1[2],N,h,K,k1[2],method)

end = time.time()
print(end - start)

#2------------------------------------------------------------------------------
#a----------------------------------------------------------------
def f_FSEP(S0,r,sigma,t,T,paths,steps,method):
    dt = T/steps
    price = StockPrices(S0,r,sigma,T,paths,steps)
    K = price[:,int(0.2/dt)]
    option = np.mean(np.maximum(K-price[:,steps-1],0))*np.exp(-r)
    return option

EuroP_FS = f_FSEP(65.0,0.06,0.2,0.2,1,100000,100,"Monomials")

#b----------------------------------------------------------------
def f_FSAP(S0,r,sd,k,t,T,paths,steps,methods):
    dt = T/steps
    St = StockPrices(S0, r, sd, T, paths, steps)
    stop = int(0.2/dt)
    K = St[:,stop]
    # initialize payoffs matrix
    payoffs = np.zeros((paths, steps))
    payoffs[:,steps - 1] = np.maximum(K - St[:,steps - 1], 0)
    # initialize stopping time matrix
    index = np.zeros((paths, steps))
    index[:,steps-1] = np.where(payoffs[:,steps - 1]> 0, 1, 0)
    discI = np.array(index[:,steps-1])
    Ot = np.array(np.maximum(K - St[:,steps-1],0))
    # initialize continuation value matrix
    for j in reversed(range(stop,steps - 1)):
        payoffs[:,j] = np.maximum(K - St[:, j],0)
        # Find in the money paths
        In = np.where(Ot*np.exp(-r*dt*discI) > 0)[0]
        #  Use x which are in the money
        X = f_reg(St[In, j], k, methods)         
        Y = np.array(Ot*np.exp(-r*dt*discI))[In]
        # Find Least Square Beta
        A = np.dot(X, X.T)
        b = np.dot(X, Y)
        Beta = np.dot(np.linalg.inv(A), b)
        # find full x  
        x = f_reg(St[:, j], k, methods) 
        # find continue value
        expected = np.dot(x.T, Beta)
        # update decision rule
        index[:, j] = np.where(np.array(payoffs[:,j]) - np.array(expected) > 0, 1, 0)
        for l in range(paths):
            if index[l,j] == 1:
                discI[l] = 1
                Ot[l] = K[l] - St[l,j]
            elif index[l,j] == 0:
                discI[l] += 1
        payoffs[:,j] = np.maximum(np.array(payoffs[:,j]),np.maximum(np.array(expected),0))
    # Find the first occurence of 1, indicating the earlist exercise date
    first_exercise = np.argmax(index, axis = 1) 
    index = np.zeros(shape = (paths, steps))
    index[np.arange(paths), first_exercise] = 1
    option = 0
    temp = index*payoffs
    for q in range(steps):
        option += np.mean(temp[:,q]*np.exp(-r*dt*(q+1)))
    return option

AmeriP_FS = f_FSAP(65,0.06,0.2,2,0.2,1,100000,100,"Monomials")
