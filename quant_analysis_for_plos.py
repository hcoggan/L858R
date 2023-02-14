# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:48:07 2022

@author: 44749
"""

import numpy as np
from matplotlib import pyplot as plt

v_0 = 80



day7_areas = np.load("day7_areas.npy", allow_pickle=True)
day14_areas = np.load("day14_areas.npy", allow_pickle=True)

day7_vols = [[4*np.pi*np.power(np.sqrt(a/(np.pi)), 3)/3 for a in areas] for areas in day7_areas]
day14_vols = [[4*np.pi*np.power(np.sqrt(a/(np.pi)), 3)/3 for a in areas] for areas in day14_areas]


#first fit single-parameter growth rate
day_7_single_p = [np.average([np.log(v/v_0)/7 for v in vs]) for vs in day7_vols]
day_14_single_p = [np.average([np.log(v/v_0)/14 for v in vs]) for vs in day14_vols]

#print(day_7_single_p)
#print(day_14_single_p)

#now surface-based growth
#day_7_single_p = [np.average([(np.cbrt(v)-np.cbrt(v_0))/7 for v in vs]) for vs in day7_vols]
#day_14_single_p = [np.average([(np.cbrt(v)-np.cbrt(v_0))/14 for v in vs]) for vs in day14_vols]

#print(day_7_single_p)
#print(day_14_single_p)
exp = 0
#return volume from parameters
def logistic(alpha, K, t):
    return v_0*K*np.exp(alpha*t)/((K-v_0)+v_0*np.exp(alpha*t))

#return growth rate from carrying capacity
def alpha(v, K, t):
    frac = v*(K-v_0)/(v_0*(K-v))
    return np.log(frac)/t

#fit growth rate from carrying capacities
#Ks = np.linspace(0, 100, 1000)
#average growth-rates at day 7 and day 14
#avs7 = [np.average([alpha(vol, K, 7) for vol in day7_vols[exp]]) for K in Ks]
#avs14 = [np.average([alpha(vol, K, 14) for vol in day14_vols[exp]]) for K in Ks]


#plot average growth rates against carrying capacity at day 7 and day 14
#plt.plot(Ks, avs7, label="day 7")
#plt.plot(Ks, avs14, label="day 14")
#plt.xlabel("carrying capacity, /um^3 day")
#plt.ylabel("average division rate /day")
#plt.title("Parameter sweep of logistic model for Experiment "+str(exp+1))
#plt.legend()
#plt.show()



#fit death rate from absorption rate
def death(v, beta, t):
    return beta*(np.cbrt(v)-np.cbrt(v_0)*np.exp(-beta*t/3))/(1-np.exp(-beta*t/3))

#betas = np.linspace(-1, 1, 1000)


#fit average death rate at day 7 and day 14
#avs7 = [np.average([death(vol, beta, 7) for vol in day7_vols[exp]]) for beta in betas]
#avs14 = [np.average([death(vol, beta, 14) for vol in day14_vols[exp]]) for beta in betas]

#what is the point at which they match?
#argmin = np.argmin([abs(a7-a14) for (a7, a14) in zip(avs7, avs14)])

#print(betas[argmin], avs7[argmin])

#plt.plot(betas, avs7, label="day 7")
#plt.plot(betas, avs14, label="day 14")
#plt.xlabel("absorption coefficient, / um/day")
#plt.ylabel("average death rate, /day")
#plt.title("Parameter sweep of absorption model for experiment "+str(exp+1))
#plt.legend()
#plt.show()


#calculate the distribution of death rates at the matching point
#alphas_7 = [death(v, betas[argmin], 7) for v in day7_vols[exp]] 
#alphas_14 = [death(v, betas[argmin], 14) for v in day14_vols[exp]]

#plt.hist(alphas_7, density=True, histtype="step", label="from day 7 data")
#plt.hist(alphas_14, density=True, histtype="step", label="from day 14 data")
#plt.xlabel("death rate /day")
#plt.ylabel("probability")
#plt.title("Distribution of death rates for experiment " + str(exp+1))
#plt.legend()
#plt.show()

int_7 = [np.average([np.log(v/v_0)/7 for v in vols]) for vols in day7_vols]
int_14 = [np.average([np.log(v/v_0)/14 for v in vols]) for vols in day14_vols]
#calculate the time-averaged fitnesses
print(int_7, int_14)

#calculate the actual distribution of time-average fitnesses
int_7_dists = [[np.log(v/v_0)/7 for v in vols] for vols in day7_vols]
int_14_dists = [[np.log(v/v_0)/14 for v in vols] for vols in day14_vols]

#plot those distributions

plt.hist(int_14_dists[0], density=True, histtype="step", label="Experiment 1")
plt.hist(int_14_dists[1], density=True, histtype="step", label="Experiment 2")
plt.hist(int_14_dists[2], density=True, histtype="step", label="Experiment 3")
plt.xlabel("time-averaged fitness, /day")
plt.ylabel("probability")
plt.legend()
plt.show()
