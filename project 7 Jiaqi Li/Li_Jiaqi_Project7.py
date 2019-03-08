# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:56:07 2019

@author: Jiaqi Li
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si

#Problem 1-----------------------------------------------------
S0 = 10
sigma = 0.2
r = 0.04
dt = 0.002
dX = sigma*np.array([np.sqrt(dt),np.sqrt(3*dt),np.sqrt(4*dt)])
K = 10
T = 0.5
steps = int(T/dt)
S = np.arange(4,17,1)

###############B-S######################
def f_BS(S0,type):
    K=10.0
    T=0.5
    sigma=0.2
    r=0.04
    d_1=(np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma * np.sqrt(T))
    d_2=d_1-sigma*np.sqrt(T)
    if type=="call":
        option=(S0*si.norm.cdf(d_1,0.0,1.0)-K*np.exp(-r*T)*si.norm.cdf(d_2,0.0,1.0))
    if type=="put":
        option=(K*np.exp(-r*T)*si.norm.cdf(-d_2, 0.0, 1.0)-S0*si.norm.cdf(-d_1,0.0,1.0))
    return option

BS_EFD = np.zeros(13)
for i in range(13):
    BS_EFD[i] = f_BS(S[i],"put")

###############EFD######################
def f_EFD(S0,dX,sigma,dt,r,K,steps):
    Pu = dt*(sigma**2/(2*dX**2)+(r-sigma**2/2)/(2*dX))
    Pm = 1-dt*sigma**2/(dX**2) - r*dt
    Pd = dt*(sigma**2/(2*dX**2)-(r-sigma**2/2)/(2*dX))
    
    TerP = np.arange(np.log(100)+dX,np.log(1)-dX,-dX)
    index = np.where(np.exp(TerP)<S0)[0][0]
    N = len(TerP)
    A = np.zeros(shape = (N,N))
    move = np.array([0,1,2])
    A[0,move] = [Pu,Pm,Pd]
    A[1,move] = [Pu,Pm,Pd]
    for i in range(1,(N-3)):
        A[i+1,move+i] = [Pu,Pm,Pd]
    A[N-2,move+N-3] = [Pu,Pm,Pd]
    A[N-1,move+N-3] = [Pu,Pm,Pd]
    
    F = np.maximum(K - np.exp(TerP),0)
    
    B = np.zeros(N)
    B[-1] = np.exp(TerP[-2])-np.exp(TerP[-1])
    
    for i in range(steps):
        F = np.dot(A,F)+B
    option = np.mean([F[index-1],F[index]])
    return option

optionE = np.zeros((3,13))
for i in range(3):
    for j in range(13):
        optionE[i,j] = f_EFD(S[j],dX[i],sigma,dt,r,K,steps)

Err_EFD = BS_EFD - optionE
        
###############IFD######################
def f_IFD(S0,dX,sigma,dt,r,K,steps):
    Pu = -0.5*dt*(sigma**2/(dX**2)+(r-sigma**2/2)/(dX))
    Pm = 1+dt*sigma**2/(dX**2) + r*dt
    Pd = -0.5*dt*(sigma**2/(dX**2)-(r-sigma**2/2)/(dX))
    
    TerP = np.arange(np.log(20)+dX,np.log(1)-dX,-dX)
    index = np.where(np.exp(TerP)<S0)[0][0]
    N = len(TerP)
    A = np.zeros(shape = (N,N))
    move = np.array([0,1,2])
    A[0,[0,1]] = [1,-1]
    A[1,move] = [Pu,Pm,Pd]
    for i in range(N-3):
        A[i+1,move+i] = [Pu,Pm,Pd]
    A[-2,[-3,-2,-1]] = [Pu,Pm,Pd]
    A[-1,[-2,-1]] = [1,-1]
    
    F = np.maximum(K - np.exp(TerP),0)
    
    B = np.zeros(N)
    B[1:-1]= F[1:-1]
    B[-1] = np.exp(TerP[-2])-np.exp(TerP[-1])
    
    for i in range(steps):
        F = np.dot(np.linalg.inv(A),B)
        B = np.zeros(N)
        B[1:-1]= F[1:-1]
        B[-1] = np.exp(TerP[-2])-np.exp(TerP[-1])
    option = np.mean([F[index-1],F[index]])
    return option

optionI = np.zeros((3,13))
for i in range(3):
    for j in range(13):
        optionI[i,j] = f_IFD(S[j],dX[i],sigma,dt,r,K,steps)

Err_IFD = BS_EFD - optionI

###############CNFD######################
def f_CNFD(S0,dX,sigma,dt,r,K,steps):
    Pu = -1/4*dt*(sigma**2/(dX**2)+(r-sigma**2/2)/(dX))
    Pm = 1+dt*sigma**2/(2*dX**2) + r*dt/2
    Pd = -1/4*dt*(sigma**2/(dX**2)-(r-sigma**2/2)/(dX))
    
    TerP = np.arange(np.log(20)+dX,np.log(1)-dX,-dX)    
    F = np.maximum(K - np.exp(TerP),0)
    
    index = np.where(np.exp(TerP)<S0)[0][0]
    N = len(TerP)
    move = np.array([0,1,2])
    X = np.zeros(shape = (N,N))
    X[0,[0,1]] = 0
    X[1,move] = [-Pu,-(Pm-2),-Pd]
    for i in range(1,(N-3)):
        X[i+1,move+i] = [-Pu,-(Pm-2),-Pd]
    X[-2,move+N-3] = [-Pu,-(Pm-2),-Pd]
    
    Y = np.zeros(N)
    Y[-1] = np.exp(TerP[-2])-np.exp(TerP[-1])
    
    A = np.zeros(shape = (N,N))
    A[0,[0,1]] = [1,-1]
    A[1,move] = [Pu,Pm,Pd]
    for i in range(1,(N-3)):
        A[i+1,move+i] = [Pu,Pm,Pd]
    A[-2,move+N-3] = [Pu,Pm,Pd]
    A[-1,[-2,-1]] = [1,-1]
    
    B = np.dot(np.linalg.inv(A),X)
    D = np.dot(np.linalg.inv(A),Y)
    
    for i in range(steps):
        F = np.dot(B,F)+D
    
    option = np.mean([F[index],F[index-1]])
    return option

optionCN = np.zeros((3,13))
for i in range(3):
    for j in range(13):
        optionCN[i,j] = f_CNFD(S[j],dX[i],sigma,dt,r,K,steps)

Err_CNFD = BS_EFD - optionCN

#Problem 2-----------------------------------------------------
S0 = 10
sigma = 0.2
r = 0.04
dt = 0.002
dS = 0.25
K = 10
T = 0.5
steps = int(T/dt)
###############Generalized Finite-Difference######################
def f_A_PDE(a,dS,r,sigma,dt,S0,K,steps,name):
    TerP = np.arange(0,20+dS,dS)
    index = np.abs(TerP-S0).argmin()
    n = len(TerP)
    
    j = np.arange(1,n-1,1)
    alpha = a
    a1 = 0.5*((sigma**2)*(j**2)-r*j)*(1-alpha)
    a2 = -1/dt-((sigma**2)*(j**2)+r)*(1-alpha)
    a3 = 0.5*((sigma**2)*(j**2)+r*j)*(1-alpha)
    b1 = 0.5*((sigma**2)*(j**2)-r*j)*alpha
    b2 = 1/dt-((sigma**2)*(j**2)+r)*alpha
    b3 = 0.5*((sigma**2)*(j**2)+r*j)*alpha
    
    P = np.maximum(np.round(K - TerP,4),0)
    C = np.maximum(np.round(TerP - K,4),0)
    
    A = np.zeros(shape = (n,n))
    move = np.array([0,1,2])
    A[0,[0,1]] = [1,-1]
    for i in range(n-2):
        A[i+1,move+i] = [a1[i],a2[i],a3[i]]
    A[-1,[-2,-1]] = [1,-1]
    
    B = np.zeros(shape = (n,n))
    for i in range(n-2):
        B[i+1,move+i] = [-b1[i],-b2[i],-b3[i]]
    
    eC = np.zeros(n)
    eC[0] = TerP[0]-TerP[1]
    eP = np.zeros(n)
    eP[0] = TerP[-2]-TerP[-1]
    
    for i in range(steps):
        C = np.dot(np.linalg.inv(A),np.dot(B,C)+eC)
        C = np.maximum(C,np.maximum(np.round(TerP - K,4),0))
        P = np.dot(np.linalg.inv(A),np.dot(B,P)+eP)
        P = np.maximum(P,np.maximum(np.round(K - TerP,4),0))
        
    if name == "C":
        return C[index]
    else:
        return P[index]

optionC = np.zeros((9,13))
optionP = np.zeros((9,13))
S = np.arange(4,17,1)
dS = [0.25,1,1.25]
a = [1,0,0.5]

pS = [0,0,0,1,1,1,2,2,2]
pa = [0,1,2,0,1,2,0,1,2]
for i in range(9):
    for j in range(len(S)):
        optionC[i,j] = f_A_PDE(a[pa[i]],dS[pS[i]],r,sigma,dt,S[j],K,steps,"C")
for i in range(9):
    for j in range(len(S)):
        optionP[i,j] = f_A_PDE(a[pa[i]],dS[pS[i]],r,sigma,dt,S[j],K,steps,"P")

name = ["explicit method","implicit method","Crank-Nicolson"]
Shape = ["s","v","o"]
plt.figure(1,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionC[i,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Call with dS = 0.25")
plt.xlabel("S0")
plt.ylabel("Option Prices")

plt.figure(2,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionC[i+3,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Call with dS = 1")
plt.xlabel("S0")
plt.ylabel("Option Prices")

plt.figure(3,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionC[i+6,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Call with dS = 1.25")
plt.xlabel("S0")
plt.ylabel("Option Prices")

plt.figure(4,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionP[i,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Put with dS = 0.25")
plt.xlabel("S0")
plt.ylabel("Option Prices")

plt.figure(5,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionP[i+3,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Put with dS = 1")
plt.xlabel("S0")
plt.ylabel("Option Prices")

plt.figure(6,figsize = (8,6))
for i in range(3):
    plt.plot(np.arange(4,17,1),optionP[i+6,:],marker = Shape[i],label = name[i])
plt.legend()
plt.title("American Put with dS = 1.25")
plt.xlabel("S0")
plt.ylabel("Option Prices")

