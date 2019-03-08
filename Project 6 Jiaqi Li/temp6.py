#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:31:15 2019

@author: meixiangui
"""

import matplotlib.pyplot as plt
import math 
import numpy as np

#Q1 (a) Lookback Option 
# define the stock process
def Stock_price(S0, r, sd, T, paths, steps):
    dt = T/steps
    # Generate stochastic process and its antithetic paths
#    np.random.seed(999)
    z = np.random.normal(0, 1, paths/2 * steps).reshape(paths/2,steps)
    z_inv = -z
    dwt = np.sqrt(dt) * z
    dwt_inv = np.sqrt(dt) * z_inv
    # bind the normal and antithetic Wt
    dwt = np.concatenate((dwt, dwt_inv), axis=0)
    # define the initial value of St
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*(sd**2))*dt + sd*dwt[:, i - 1])    
    return St[:, 1:]

def lb_option(S0, r, sd, T, paths, steps, K, type):
    St=Stock_price(S0, r, sd, T, paths, steps) 
    if type=='call':
        call=np.zeros(paths)
        for i in range(paths):
            call[i]=np.maximum(max(St[i])-K,0.0)
        option_price=np.exp(-r*T)*np.mean(call)
    elif type=='put':
        put=np.zeros(paths)
        for i in range(paths):
            put[i]=np.maximum(K-min(St[i]),0.0)
        option_price=np.exp(-r*T)*np.mean(put)
    return option_price

S0=98.0; r=0.03;T=1.0; paths=10000; steps=100; K=100.0
sd=np.arange(0.12,0.52,0.04)
lb_call = [lb_option(S0, r, m, T, paths, steps, K, 'call')for m in sd]
lb_put = [lb_option(S0, r, m, T, paths, steps, K, 'put') for m in sd]

plt.figure(figsize=(6,4))
plt.plot(sd,lb_call,color='orange')
plt.xlabel('Volitality')
plt.ylabel('European Call Price')
plt.title('European Lookback Call Option')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(sd,lb_put,color='green')
plt.xlabel('Volitality')
plt.ylabel('European Put Price')
plt.title('European Lookback Put Option')
plt.show()

#******************************************
# Q2
def loan(r0,delta,lambda2,L0,t,T):
    n=12*T
    R=r0+delta*lambda2
    r=R/12
    PMT=(L0*r)/(1.0-1.0/((1+r)**n))
    a=PMT/r
    b=PMT/(r*(1+r)**n)
    c=1+r
    loan =a-b*c**(12*t)
    return loan[1:]

#loan=[loan(r0,delta,lambda2,L0,m,T) for m in t]
#test = loan(r0,delta,lambda2,L0,t,T)

def S(lambda2,T):
    paths=10000;steps=60
    dt = T/steps
    Nt=np.clip(np.random.poisson(lambda2*dt,(paths,steps)),0.0,1.0)
    S=np.argmax(Nt,axis=1)*dt
    no_default = np.where(np.sum(Nt, axis = 1) == 0.0)
    S[no_default] = 100.0
    return S
#s=S(paths,steps,lambda2)
    
#def Q(r0,delta,lambda1,lambda2,L0,t,T,paths,steps,q):
#    V0=20000.0;mu=-0.1;gamma=-0.4;
#    dt=T/steps
#    z = np.random.normal(0, 1, paths*steps).reshape(paths,steps)
#    dwt = np.sqrt(dt) * z
#    # define the initial value of Vt
#    Vt = np.zeros((paths, steps + 1))
#    Vt[:, 0] = V0
#    for j in range (1, steps + 1):
#          Vt[:, j] = (Vt[:, j - 1]*np.exp((mu - 1/2*np.power(sd, 2))*dt \
#            +sd*dwt[:,j-1])*(1.0+gamma * np.random.poisson(lambda1*dt,paths)))   
#    Vt=Vt[:, 1:]
##   t = [np.arange(0,T+dt,dt)]
#    # find loan balance and recovery rate over time
#    Loan= loan(r0,delta,lambda2,L0,t,T)
#    residual_collateral = np.tile(np.array(Loan)*np.array(q),paths).reshape((paths,steps))
#    default = np.where(Vt-residual_collateral<= 0.0, 1.0, 0.0)
#    # Find stop time index
#    Q = np.argmax(default, axis = 1)*dt
#    # If there is no default
#    no_default = np.where(np.sum(default, axis = 1) == 0.0)
#    Q[no_default] = 100.0
#    return Q   

def option(lambda1 = 0.2,lambda2 = 0.4,T=5.0):
    #****************
    V0=20000.0;sd=0.2;paths=10000;steps=60;mu=-0.1;gamma=-0.4;
    T=5.0;r0=0.02;delta=0.25;L0=22000.0;epl=0.95;alpha=0.7
    dt = T/steps
    # caculate q1
    t= np.arange(0,T+dt,dt)
    beta=(epl-alpha)/T
    q1=[alpha+beta*m for m in t[1:]]
    # caculate the value of collateral
    # Generate stochastic process
    z = np.random.normal(0, 1, paths*steps).reshape(paths,steps)
    dwt = np.sqrt(dt) * z
    # define the initial value of Vt
    Vt = np.zeros((paths, steps + 1))
    Vt[:, 0] = V0
    for j in range (1, steps + 1):
          Vt[:, j] = (Vt[:, j - 1]*np.exp((mu - 1/2*np.power(sd, 2))*dt \
            +sd*dwt[:,j-1])*(1.0+gamma * np.random.poisson(lambda1*dt,paths)))   
    Vt=Vt[:, 1:]
    #***********************
    # caculate loan and payoffs
    ###############################################################
#   t = [np.arange(0,T+dt,dt)]
    # find loan balance and recovery rate over time
    Loan= np.round(loan(r0,delta,lambda2,L0,t,T),4)
    residual_collateral = np.tile(Loan*q1,paths).reshape((paths,steps))
    default = np.where(Vt-residual_collateral<= 0.0, 1.0, 0.0)
    # Find stop time index
    q = np.argmax(default, axis = 1)*dt
    # If there is no default
    no_default = np.where(np.sum(default, axis = 1) == 0.0)
    q[no_default] = 100.0

    Nt=np.clip(np.random.poisson(lambda2*dt,(paths,steps)),0.0,1.0)
    s=np.argmax(Nt,axis=1)*dt
    no_default2 = np.where(np.sum(Nt, axis = 1) == 0.0)
    s[no_default2] = 100.0
    ###############################################################
    
    payoffs = np.zeros(paths)
#    q=Q(r0,delta,lambda2,L0,t,T,paths,steps,q1)
#    s=S(lambda2,T)
    n=12*T
    R=r0+delta*lambda2
    r=R/12
    PMT=(L0*r)/(1.0-1.0/(1+r)**n)
    a=PMT/r
    b=PMT/(r*(1+r)**n)
    c=1+r
    count = 0.0
    for i in range(paths):
        if q[i]==100.0 and s[i]==100.0:
           payoffs[i]=0.0
        elif q[i]<=s[i]:
             payoffs[i]=np.maximum((a-b*c**(12*q[i]))-epl*Vt[i,int(q[i]/dt)],0.0) \
            *np.exp(-r0*q[i])
             count += 1.0
        elif q[i]>s[i]:
             payoffs[i]=np.abs((a-b*c**(12*s[i]))-epl*Vt[i,int(s[i]/dt)]) \
            *np.exp(-r0*s[i])
             count += 1.0
    option_value=round(np.mean(payoffs),4)
    prob = count/paths
    #************************
    tau = np.zeros(paths)         
    for i in range(paths):
        tau[i] = min(min(s[i],q[i]),5.0)
    i = 0
    while i < paths:
        if tau[i] == 5.0:
           tau = np.delete(tau,i)
           paths = paths - 1
        i += 1
    exp_t=round(np.mean(tau),4)
    return option_value,prob,exp_t


print('option value, default prob and expected default time are separately:')
print(option(lambda1=0.2,lambda2 =0.4,T=5.0))

l1 = np.arange(0.05,0.45,0.05)
l2 = np.arange(0, 0.9, 0.1)
time = np.arange(3.0, 9.0, 1.0)       

#plot the figure 
#(a)   
list1 = np.zeros((8,6))
list2 = np.zeros((9,6))

for j in range(6):
    for i in range(8):
        list1[i,j] = option(lambda1 = l1[i], T = time[j])[0]

for j in range(6):
    for i in range(9):
        list2[i,j] = option(lambda2 = l2[i], T = time[j])[0]

#******************
plt.figure(figsize=(6, 8))
for i in range(8):
    plt.plot(list1[:,j],linestyle='--', 
             marker='o', label = '$\lambda1 = $' + str(l1[i]))
plt.legend()
plt.xlabel('time')
plt.ylabel('Default Option Value')
plt.title('Default Option Value with Different $\lambda_1$')
plt.show()

plt.figure(figsize=(6, 8))
for i in range(9):
    plt.plot(list2[i,:], linestyle='--', 
             marker='o', label = '$\lambda2 = $' + str(l2[i]))
plt.legend()
plt.xlabel('T')
plt.ylabel('Default Option Value ($)')
plt.title('Default Option Value with Different $\lambda_2$')
plt.show()

#(b)


    
        

















