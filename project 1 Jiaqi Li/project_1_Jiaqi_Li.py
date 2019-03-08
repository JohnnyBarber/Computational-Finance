# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:56:02 2019

@author: Jiaqi Li
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

#1--------------------------------
#Set random number generators 
a = 7**5
b = 0
m = 2**31-1

#Creat a list to store random numbers with U[0,1]
unif = [None] * 10001
unif[0] = 1

#This loop is for generating random numbers with U[0,1]
i = 1
while i < 10001:
    unif[i] = np.mod(a*unif[i-1]+b,m)
    i = i+1
unif = [x/m for x in unif]

#Delete the first observation x_0 and keep x_1 to x_10000
del unif[0]

#Draw the histogram of the sample uniform distribution
plt.figure()
ax1 = plt.hist(unif, normed = True, bins = 30)
plt.title("Uniform Distribution")
plt.ylabel("probability")
plt.xlabel("value")

print("mean: ", np.mean(unif))
print("std: ", np.std(unif))

#Use built-in function to generate uniform distribution
build = np.random.uniform(0,1,10000)
print("build in function mean: ", np.mean(build))
print("build in function std: ",np.std(build))

#2-----------------------------------
#Creat a list to store random numbers from the discrete distribution
dist = [None] * 10000

#Set probabilities for different outcomes from the distribution
p1 = 0.3; p2 = 0.35; p3 = 0.2; p4 = 0.15

#This loop is for generating the discrete distribution
i = 0
while i < 10000:
    if unif[i] <= p1:
        dist[i] = -1
    elif p1 < unif[i] <= p1+p2:
        dist[i] = 0
    elif p1+p2 < unif[i] <= p1+p2+p3:
        dist[i] = 1
    elif p1+p2+p3 < unif[i] <= p1+p2+p3+p4:
        dist[i] = 2
    i = i + 1
    
print("mean: ", np.mean(dist))
print("std: ", np.std(dist))

#Draw the histogram
plt.figure()
ax2 = plt.hist(dist, normed = True)
plt.title("Discrete Distribution")
plt.ylabel("probability")
plt.xlabel("value")

#3-----------------------------------
#Set seed so that each time we can gerenate the same sequence 
#of random numbers for each loop that is used to generate
#binomial random numbers
#By setting such a seed, our result will be consistent and easy
#to study with
random.seed(9)

#Creat a list to store random numbers from the bernoulli distribution
ber = [None] * 44
p = 0.64

#Creat a list to store random numbers from the binomail distribution
B = [None]*1000

#This loop will generate 1000 binomially distributed random numbers
x = 1
while x < 1001:
    u = [None] * 45
    #generate random number for x_0 in each loop
    u[0] = random.randint(1,100)
    #generate a set of 44 uniformly distributed random numbers
    #each set of these numbers will generate 1 random number
    #with Binomial(44,0.64)
    i = 1
    while i < 45:
        u[i] = np.mod(a*u[i-1]+b,m)
        i = i+1
    del u[0]
    j = 0
    while j < 44:
        if u[j]/m <= p:
            ber[j] = 1
        else:
            ber[j] = 0
        j = j + 1
    B[x-1] = sum(ber)
    x = x + 1

#Draw the histogram
plt.figure()
ax3 = plt.hist(B, normed = True)
plt.title("Binomial Distribution")
plt.ylabel("probability")
plt.xlabel("value")

#Compute P(X>=40)
P = sum(1 for i in B if i >= 40)

#Compute theoretical value of P(X>=40) to compare with the empirical result
k = 40
n = 44
P_T = 0
while k < 45:
    P_true = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))*(p**k)*((1-p)**(n-k))
    P_T = P_T + P_true
    k = k + 1
print("Empirical probability =", P)
print("Theoretical Probability =", P_T,"is approximately 0")

        
#4----------------------------------
lam = 1.5

#Use the uniform distributed random numbers created in question 1
#to generate a exponential distribution
U_4 = [1-x for x in unif]
X_4 = -1/lam*np.log(U_4)

#Compute P(X>=1) and P(X>=4)
P_1 = sum(1 for i in X_4 if i >= 1)/10000
P_4 = sum(1 for i in X_4 if i >= 4)/10000
print("P(X >= 1) =", P_1, ", P(X >= 4) =", P_4)
mu_4 = np.mean(X_4)
sigma_4 = np.std(X_4)
print("mean =", mu_4, ", std =", sigma_4)

#Draw the histogram
plt.figure()
ax4 = plt.hist(X_4, normed = True, bins = 30)
plt.title("Exponential Distribution")
plt.ylabel("probability")
plt.xlabel("value")

#5-------------------------------------------------
#Creat a function that can generate random number with U[0,1]
#with sample size n and initial value x_0
def f_unif(n,x_0):
    U = [None] * (n+1)
    U[0] = x_0
    i = 1
    while i < (n+1):
        U[i] = np.mod(a*U[i-1]+b,m)
        i = i+1
    del U[0]
    U = [x/m for x in U]
    return U

#Generate a uniform distribution with 5000 observations and x_0 = 1
U_5 = f_unif(n = 5000, x_0 = 1)

#Creat 2 lists with length 2500 each
#These 2 lists will be used to store normally distributed random numbers
Z_1 = [None]*2500
Z_2 = [None]*2500

#Here we used Box-Muller Method
for i in range(2500):
    Z_1[i] = np.sqrt(-2*np.log(U_5[2*i]))*np.cos(2*math.pi*U_5[2*i+1])
    Z_2[i] = np.sqrt(-2*np.log(U_5[2*i]))*np.sin(2*math.pi*U_5[2*i+1])
    i = i + 1

#Combine 2 lists to get a normal distribution with 5000 observations
Z_BM = Z_1+Z_2

mu_BM = np.mean(Z_BM)
std_BM = np.std(Z_BM)

print("Mox-Muller method: mean =", mu_BM, ", std =", std_BM)

#Draw the histogram
plt.figure()
ax5 = plt.hist(Z_BM, normed = True, bins = 30)
plt.title("Box-Muller Method")
plt.ylabel("probability")
plt.xlabel("value")

#Here we used Polar-Marsaglia Method
Z_1_1 = [None]*2500
Z_2_2 = [None]*2500
for i in range(2500):
    V_1 = 2*U_5[2*i]-1
    V_2 = 2*U_5[2*i+1]-1
    W = V_1**2+V_2**2
    #drop V_1 and V_2 if W <= 1
    if W <= 1:
        Z_1_1[i] = np.sqrt((-2*np.log(W))/W)*V_1
        Z_2_2[i] = np.sqrt((-2*np.log(W))/W)*V_2
Z_PM = Z_1_1 + Z_2_2
Z_PM = [x for x in Z_PM if x != None]
#Notice that this method will generate less than 5000 observations

mu_PM = np.mean(Z_PM)
std_PM = np.std(Z_PM)

print("Polar-Marsaglia method: mean =", mu_PM, ", std =", std_PM)

#Draw the histogram
plt.figure()
ax6 = plt.hist(Z_PM, normed = True, bins = 30) 
plt.title("Polar-Marsaglia Method")
plt.ylabel("probability")
plt.xlabel("value")

#Now we want to compare time efficiency of the two methods

#The following uniform distribution is used to generate
#2 normal distributions each with 5000 observations
#by using different methods
U_5_test = f_unif(10000,2)

Z_1_test = [None]*2500
Z_2_test = [None]*2500

#Box-Muller Method
#Record starting time
start_time1 = time.time()
for i in range(2500):
    Z_1_test[i] = np.sqrt(-2*np.log(U_5_test[2*i]))*np.cos(2*math.pi*U_5_test[2*i+1])
    Z_2_test[i] = np.sqrt(-2*np.log(U_5_test[2*i]))*np.sin(2*math.pi*U_5_test[2*i+1])
    i = i + 1

#Record ending time and compute time used
time_1 = (time.time() - start_time1)
print("--- %s seconds ---" % time_1)

Z_1_1_test = [None]*2500
Z_2_2_test = [None]*2500

#Record starting time
start_time2 = time.time()
x = 0
for i in range(10000):
    V_1 = 2*U_5_test[2*i]-1
    V_2 = 2*U_5_test[2*i+1]-1
    W = V_1**2+V_2**2
    if W <= 1:
        Z_1_1_test[x] = np.sqrt((-2*np.log(W))/W)*V_1
        Z_2_2_test[x] = np.sqrt((-2*np.log(W))/W)*V_2
        x = x + 1
    #when Z_1_1_test and Z_2_2_test both contain 2500 observations, exit loop
    if x > 2499:
        break

#Record ending time and compute time used
time_2 = (time.time() - start_time2)
print("--- %s seconds ---" % time_2)

print("Box-Muller takes", time_1, "seconds.")
print("Polar-Marsaglia takes", time_2, "seconds.")
