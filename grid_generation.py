# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:43:37 2022

@author: 44749
"""

import numpy as np
from matplotlib import pyplot as plt

v_0 = 30
r_0 = np.cbrt(3*v_0/(4*np.pi))
k = 80 #spring force
dt = 0.01 #timestep
T = 1 #timeframe of adjustment

a_0 = np.pi*(r_0**2)

#create an irregular 2D cell grid, roughly circular
N = 500 #no of cells
R = np.sqrt((N*a_0)/np.pi) #radius of roughly expected area
print(R)

#initially distribute cells:
xs, ys = np.zeros(N), np.zeros(N)

for n in range(N):
    x, y = np.random.uniform(-R, R), np.random.uniform(-R, R)
    while np.sqrt(x**2 + y**2) > R:
        x, y = np.random.uniform(-R, R), np.random.uniform(-R, R) #generate new point
    else:
        xs[n], ys[n] = x, y #point is within circle, distribute


#now make them push each other aside
def damped_velocities(xs, ys):
    velocities = np.zeros((N, 2))
    for n in range(N):
        pos_vec = np.array([xs[n], ys[n]])
        r_n = np.linalg.norm(pos_vec)
#        if r_n > R-r_0: #too close to the boundary
#            velocities[n] += -k*(r_n - R + r_0)*pos_vec/r_n #inward force from the circular boundary
        for m in range(n):
            direc = np.array([xs[n]-xs[m], ys[n]-ys[m]])
            r = np.linalg.norm(direc) #distance between two cells
            if r < 2*r_0: #spring force is in effect
                velocities[n] -= k*(r-2*r_0)*direc/r
                velocities[m] += k*(r-2*r_0)*direc/r
    return velocities

#calculate separation statistics
def sep_stats(xs, ys):
    min_seps = []
    av_encroachments = []
    for n in range(N):
        all_seps = []
        encroachments = []
        for m in range(N):
            if n != m:
                direc = np.array([xs[n]-xs[m], ys[n]-ys[m]])
                r = np.linalg.norm(direc) #distance between two cells
                all_seps.append(r) #separation between two cells
                if r < 2*r_0:
                    encroachments.append(2*r_0 - r) #encroachments
        min_seps.append(min(all_seps))
        if len(encroachments) > 0:
            av_encroachments.append(np.average(encroachments)) 
    return (np.average(min_seps), np.average(av_encroachments))

for step in range(int(T/dt)):
    velocities = damped_velocities(xs, ys)
    x_vels, y_vels = np.array([velocities[n][0] for n in range(N)]), np.array([velocities[n][1] for n in range(N)])
    xs += x_vels*dt
    ys += y_vels*dt
    print(step)

av_sep, av_enc = sep_stats(xs, ys)


plt.scatter(xs, ys)
plt.xlabel("um")
plt.ylabel("um")
plt.title("Adjusted positions for "+ str(N) + " cells, k=" +str(k)+", av min sep = " + str(np.around(av_sep/r_0, decimals=2)) + "r_0, av encroach = " + str(np.around(av_enc/r_0, decimals=2)) + "r_0")

grid = [[N, k, dt, T], [xs, ys]]

np.save("2D_irreg_grid.npy", grid)
            
            
                

