# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:55:28 2022

@author: 44749
"""

import numpy as np
from matplotlib import pyplot as plt

v_0 = 80
r_0 = np.cbrt(v_0)

t = 0

n = 30 #number of cells in each direction
N = n**3 #number of cells total


alpha = 0
dT = 0.01

[xs, ys, zs] = np.load("square_grid_coords_30.npy", allow_pickle=True) #load grid
neighbours = np.load("square_neighbour_list_30.npy", allow_pickle=True) #load neighbours

age_of_first_occupations = np.zeros(N) #keep track of the time at which a lattice point was first occupied

gamma = 10 #repulsion of occupied cells
delta = 1.0 #attraction of neighbours
beta = 1 #benefit from empty neighbour cells

senesc_rate = 0.0 #for trials of senescence

h = 1 #steepness of fitness function
contact_inhib_thresh = 25 #number of occupied neighbours at which suppression kicks in

prob_diff = 1 #prob that it is born a stem cell

#reproduce cells
def reproduce(occupations, active_cell_counts, age_of_first_occupations, t):
    for n in range(N): #all of these should have occupation number 1 or 0 after the last adjustment step
        if occupations[n] == 1:
            if active_cell_counts[n] == 1: #is the cell able to actively reproduce?
                if np.random.rand() < senesc_rate*dT:
                    active_cell_counts[n] = 0 #render the cell inactive if we're trialling senescence
                else:
                    occupied_neighbour_no = len(list(filter(lambda i: (occupations[i]==1), neighbours[n]))) #how many neighbours does a cell have?
                    if np.random.rand() < (alpha + beta/(1+ np.exp(-h*(contact_inhib_thresh-occupied_neighbour_no))))*dT: #compute division probability
                        occupations[n] = 2 #cell is double-occupied
                        if np.random.rand() < prob_diff: #is this new cell active?
                            active_cell_counts[n] = 2 
    return occupations, active_cell_counts, age_of_first_occupations


#adjust positions until we have no double occupations
#rule: if a cell becomes double-occupied through, then mark which cell was just added to it
def shift(occupations, active_cell_counts, age_of_first_occupations, type_just_added, t):
    keep_adjusting = True
    while keep_adjusting: 
        any_double_occupied = False #assume none are double occupied
        for n in range(N): #run through every point in space
            if occupations[n] > 1: #is it double-occupied? 
                any_double_occupied = True
                empty_neighbours = list(filter(lambda i: (occupations[i]==0), neighbours[n])) #are there any empty neighbours?
                if len(empty_neighbours)==0: #if all neighbours are occupied
                    ps = [np.exp(-gamma*occupations[i]) for i in neighbours[n]] #weight by their occupations
                    norm = sum(ps) #normalise
                    ps = [p/norm for p in ps]
                    index = np.random.choice(neighbours[n], size=1, p = ps)[0] #choose one to move to at random
                    occupations[index] += 1 #move it to that lattice-point
                    #are we moving an active cell or an inactive cell?
                    if type_just_added[n] == 1: #active cell was just moved there so choose from between other types
                        prob_active = (active_cell_counts[n]-1)/(occupations[n]-1) #fraction of the remaining cells which are active
                        if np.random.rand() < prob_active:
                            active_cell_counts[n] -= 1 #it is active
                            active_cell_counts[index] += 1 #so move it
                            type_just_added[index] = 1 #type just moved was active
                        else:
                            type_just_added[index] = -1 #we just moved an inactive cell
                    else:
                        if type_just_added[n] == -1:
                            prob_active = (active_cell_counts[n])/(occupations[n]-1) #fraction of the remaining cells which are active (retain an inactive cell)
                            if np.random.rand() < prob_active:
                                active_cell_counts[n] -= 1 #it is active
                                active_cell_counts[index] += 1 #so move it
                                type_just_added[index] = 1 #we just moved an active cell here
                            else:
                                type_just_added[index] = -1 #we just moved an inactive cell here
                        else: #we have no information about what was just added
                            prob_active = active_cell_counts[n]/occupations[n] #what fraction of these cells are active?
                            if np.random.rand() < prob_active:
                                active_cell_counts[n] -= 1 #it is active
                                active_cell_counts[index] += 1 #so move it
                                type_just_added[index] = 1 
                            else:
                                type_just_added[index] = -1
                else:
                    contact_list = [len(list(filter(lambda i: (occupations[i]>0), neighbours[m]))) for m in empty_neighbours] #number of neighbours possessed by every neighbouring empty lattice-point
                    weights = [np.exp(delta*c) for c in contact_list] #weight by number of neighbours
                    norm = sum(weights)
                    ps = [w/norm for w in weights]
                    index = np.random.choice(empty_neighbours, size=1, p=ps)[0]
                    occupations[index] += 1 #move to an empty one at random
                    age_of_first_occupations[index] = t
                    if type_just_added[n] == 1: #active cell was just moved there so choose from between other types
                        prob_active = (active_cell_counts[n]-1)/(occupations[n]-1)
                        if np.random.rand() < prob_active:
                            active_cell_counts[n] -= 1 #it is active
                            active_cell_counts[index] += 1 #so move it
                            type_just_added[index] = 1
                        else:
                            type_just_added[index] = -1 #moving an inactive cell
                    else:
                        if type_just_added[n] == -1:
                            prob_active = (active_cell_counts[n])/(occupations[n]-1)
                            if np.random.rand() < prob_active:
                                active_cell_counts[n] -= 1 #it is active
                                active_cell_counts[index] += 1 #so move it
                                type_just_added[index] = 1
                            else:
                                type_just_added[index] = -1
                        else: #no information about what was just added
                            prob_active = active_cell_counts[n]/occupations[n]
                            if np.random.rand() < prob_active:
                                active_cell_counts[n] -= 1 #it is active
                                active_cell_counts[index] += 1 #so move it
                                type_just_added[index] = 1
                            else:
                                type_just_added[index] = -1
                 #   print("jammed")
                occupations[n] -= 1 #take a cell away from the original cell
        keep_adjusting = any_double_occupied #have we found any double-occupied?
    return occupations, active_cell_counts, type_just_added, age_of_first_occupations


#choose centre roughly

centre = (n**2)*int(n/2) + (n*int(n/2)) + int(n/2) #set 

occupations, active_cell_counts, type_just_added = np.zeros(N), np.zeros(N), np.zeros(N) #set things up- one occupied cell in the centre
occupations[centre], active_cell_counts[centre], age_of_first_occupations[centre], type_just_added[centre] = 1, 1, 0, 1


no_cells = 1
t = 0

pops = [] #track number of cells
overall_radii = [] #track the maximum radius

while no_cells < 10000 and sum(active_cell_counts)>0:
    occupations, active_cell_counts, age_of_first_occupations = reproduce(occupations, active_cell_counts, age_of_first_occupations, t)
 #   print("reproduction")
    occupations, active_cell_counts, type_just_added, age_of_first_occupations = shift(occupations, active_cell_counts, age_of_first_occupations, type_just_added, t)
 #   print("shift")
    no_cells = sum(occupations)
    occupied = list(filter(lambda i: (occupations[i]==1), np.arange(N)))
    radii = [np.sqrt((xs[i]-xs[centre])**2 + (ys[i]-ys[centre])**2) for i in occupied]
    overall_radii.append(max(max(radii), r_0))
    print(no_cells)
    t += dT
    pops.append(no_cells)

print("time = " + str(t))
    
    


#plot cells at end of simulation

line = max(ys)/2

to_plot = list(filter(lambda i: (occupations[i]==1), np.arange(N))) #find all occupied cells
xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [ys[i] for i in to_plot], [zs[i] for i in to_plot]

#colour by neighbour number
cs_to_plot = [len(list(filter(lambda i: (occupations[i]==1), neighbours[n]))) for n in to_plot]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(xs_to_plot, ys_to_plot, zs_to_plot, c=cs_to_plot)

plt.xlabel("um")
plt.ylabel("um")
fig.colorbar(p)
plt.title(str(int(sum(occupations))) +" cells, tension="+str(delta)+", c=" +str(contact_inhib_thresh)+", alpha=" +str(alpha)+", beta="+str(beta)+", h=" + str(h))#", diff_prob=" +str(prob_diff))
#plt.colorbar()
plt.show()

                  
                