#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

v_0 = 80
r_0 = np.cbrt(v_0)

t = 0

n = 30 #number of cells in each direction
N = n**3 #number of cells total


[xs, ys, zs] = np.load("square_grid_coords_30.npy", allow_pickle=True) #load grid
neighbours = np.load("square_neighbour_list_30.npy", allow_pickle=True) #load neighbours

age_of_first_occupations = np.zeros(N) #keep track of the time at which a lattice point was first occupied


# In[65]:


alpha = 0
dT = 0.01
gamma = 10 #repulsion of occupied cells
delta = 1.0 #attraction of neighbours
beta = 1 #benefit from empty neighbour cells

senesc_rate = 0.0 #for trials of senescence

h = 1 #steepness of fitness function
contact_inhib_thresh = 25 #number of occupied neighbours at which suppression kicks in

prob_diff = 1 #prob that it is born a stem cell


# In[66]:



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


# In[19]:


#now run this once
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


# In[ ]:


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

                  


# In[114]:


#parameters of simulations- already performed

paramsets = [[1, 1, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 20, 1, 10000], [1, 5, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 0.01, 20, 1, 10000], [1, 2, 1.0, 0, 0.1, 20, 1, 10000], [1, 2, 1.0, 0, 10, 20, 1, 10000],             [1, 2, 0.1, 0, 1, 20, 1, 10000], [1, 2, 0.5, 0, 1, 20, 1, 10000], [1, 2, 2.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 10, 1, 10000], [1, 2, 1.0, 0, 1, 15, 1, 10000], [1, 2, 1.0, 0, 1, 25, 1, 10000],              [1, 10, 10, 0, -1, 1, 1, 10000], [1, 10, 10, 0, -1, 25, 1, 10000], [0, 1, 1.0, 0, -1, 1, 1, 10000], [1, 1, 1.0, 0, -5, 2, 1, 10000], [1, 0, 1.0, 0, 1, 1, 0.3, 10000], [1, 0, 1.0, 0, 1, 1, 0.5, 10000], [1, 0, 1.0, 0, 1, 1, 0.7, 10000],           [1, 5, 1.0, 0, 1, 15, 0.3, 10000], [1, 5, 1.0, 0, 1, 20, 0.3, 10000], [1, 5, 1.0, 0, 1, 25, 0.3, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000],               [1, 10, 1.0, 0, 1, 10, 1, 10000], [0.5, 10, 1.0, 0, 1, 10, 1, 10000], [0.1, 10, 1.0, 0, 1, 10, 1, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 0.1, 10, 1, 5000], [0, 10, 1.0, 0, 0.5, 10, 1, 5000], [0, 10, 1.0, 0, 1.5, 10, 1, 5000],                [0, 10, 0.1, 0, 1.0, 10, 1, 5000], [0, 1, 0.5, 0, 1.0, 10, 1, 5000], [0, 10, 2.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 15, 1, 5000], [0, 10, 1.0, 0, 1.0, 20, 1, 5000], [0, 5, 1.0, 0, 1.0, 25, 1, 10000],
            [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.5, 1.0, 10, 1, 10000], [1, 0, 1.0, 1.0, 1.0, 10, 1, 5000], [1, 0, 1.0, 2.0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.01, 10, 1.0, 0, 1.0, 10, 1, 5000],
            [0.02, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.03, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.04, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.05, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.06, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.07, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.08, 10, 1.0, 0, 1.0, 10, 1, 5000],
            [0.09, 10, 1.0, 0, 1.0, 10, 1, 5000],  [0.1, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 11, 1, 5000], [0, 10, 1.0, 0, 1.0, 12, 1, 5000], [0, 10, 1.0, 0, 1.0, 13, 1, 5000], [0, 10, 1.0, 0, 1.0, 14, 1, 5000], [0, 10, 1.0, 0, 1.0, 15, 1, 5000], [0, 10, 1.0, 0, 1.0, 16, 1, 5000],
            [0, 10, 1.0, 0, 1.0, 17, 1, 5000], [0, 10, 1.0, 0, 1.0, 18, 1, 5000], [0, 10, 1.0, 0, 1.0, 19, 1, 5000], [0, 10, 1.0, 0, 1.0, 20, 1, 5000], [0, 10, 1.0, 0, 1.0, 21, 1, 5000], [0, 10, 1.0, 0, 1.0, 22, 1, 5000], [0, 10, 1.0, 0, 1.0, 23, 1, 5000], [0, 10, 1.0, 0, 1.0, 24, 1, 5000], [0, 10, 1.0, 0, 1.0, 25, 1, 5000], [0, 10, 1.0, 0, 1.0, 26, 1, 5000],
            [1, 0, 1.0, 0, 1, 10, 1, 10000], [2, 0, 1.0, 0, 1, 10, 1, 10000], [5, 0, 1.0, 0, 1, 10, 1, 10000], [0, 1, 1.0, 0, 1, 10, 1, 5000], [0, 5, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000]]

# 40 is a duplicate of 41

#run last simulations

centre = (n**2)*int(n/2) + (n*int(n/2)) + int(n/2) #set 

print(len(paramsets))


  
for i in range(len(paramsets)):
    if i==45:
        [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
        occupations, active_cell_counts, type_just_added = np.zeros(N), np.zeros(N), np.zeros(N) #set things up- one occupied cell in the centre
        occupations[centre], active_cell_counts[centre], age_of_first_occupations[centre], type_just_added[centre] = 1, 1, 0, 1

        no_cells = 1
        t = 0

        pops = [] #track number of cells
        overall_radii = [] #track the maximum radius

        while no_cells < no_cells_max and sum(active_cell_counts)>0:
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
        np.save("sim_"+str(i)+".npy", [occupations, t])


# In[109]:




# function to take data and produce points and colours to plot

def plotfull(occupations):
    # find all occupied cells
    to_plot = list(filter(lambda i: (occupations[i] == 1), np.arange(N)))
    xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [
        ys[i] for i in to_plot], [zs[i] for i in to_plot]
    # neighbour number
    cs_to_plot = [len(
        list(filter(lambda i: (occupations[i] == 1), neighbours[n]))) for n in to_plot]
    return (xs_to_plot, ys_to_plot, zs_to_plot, cs_to_plot)

# function to produce a cross-section


def plothalf(occupations):
    line = int(max(ys)/2)
    # find all occupied cells
    to_plot = list(
        filter(lambda i: (occupations[i] == 1 and ys[i] > line), np.arange(N)))
    xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [ys[i] for i in to_plot], [zs[i] for i in to_plot]
    # neighbour number
    cs_to_plot = [len(list(filter(lambda i: (occupations[i] == 1), neighbours[n]))) for n in to_plot]
    return (xs_to_plot, ys_to_plot, zs_to_plot, cs_to_plot)


fullticks = [0, 65, 130]

#calculate neighbour statistics

av_no_neighbours = np.zeros(79)
isolated_cell_fraction = np.zeros(79) 

min_neighbours = 26

for i in range(79):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    to_plot = list(filter(lambda i: (occupations[i] == 1), np.arange(N)))
    cs_to_plot = [len(list(filter(lambda i: (occupations[i] == 1), neighbours[n]))) for n in to_plot]
    if min(cs_to_plot) < min_neighbours:
        min_neighbours = min(cs_to_plot)
    av_no_neighbours[i] = np.average(cs_to_plot)
    under_occupied = 0
    for neigh in cs_to_plot:
        if neigh < contact_inhib_thresh:
            under_occupied += 1
    isolated_cell_fraction[i] = under_occupied/len(to_plot)

np.save("av_no_neighbours.npy", av_no_neighbours)
np.save("isolated_cell_fraction.npy", isolated_cell_fraction)
        


# In[122]:


#range of neighbour colorbar should be 1-26
#find range of other stats

av_no_neighbours = np.load("av_no_neighbours.npy", allow_pickle=True)
isolated_cell_fraction = np.load("isolated_cell_fraction.npy", allow_pickle=True)
curv_ranges = np.load("curvature_ranges_sim.npy", allow_pickle=True)

av_empty_neighbours = [26- av for av in av_no_neighbours]

min_av, min_is, min_range, max_av, max_is, max_range = min(av_empty_neighbours), min(isolated_cell_fraction), min(curv_ranges), max(av_empty_neighbours), max(isolated_cell_fraction), max(curv_ranges)

#define colourmaps
curv_norm = mpl.colors.Normalize(vmin=min_range, vmax=max_range)
av_norm = mpl.colors.Normalize(vmin=min_av, vmax=max_av)
is_norm = mpl.colors.Normalize(vmin=min_is, vmax=max_is)

curv_cmap = mpl.cm.get_cmap('Greens')
av_cmap = mpl.cm.get_cmap('Oranges')
#is_cmap = mpl.cm.get_cmap('Oranges')

circle_radius = 0.1  # Adjust the radius of the circles as needed
circle_spacing = 0.3  # Adjust the spacing between circles as needed

# # make plots showing the effect of absorption-dependent growth
# ""
# for i in range(3):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\beta$'+"="+str(beta))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#    # fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i], decimals=3), np.around(av_empty_neighbours[i], decimals=3)]
#     norms = [curv_norm(curv_ranges[i]), av_norm(av_empty_neighbours[i])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i])), av_cmap(av_norm(av_empty_neighbours[i]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.savefig(str(i)+"_stat.png", format="png", bbox_inches='tight', pad_inches=0)
#     fig.savefig(str(i)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

# # set hplots
# for i in range(3, 6):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title("h=" + str(h))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i], decimals=3), np.around(av_empty_neighbours[i], decimals=3)]
#     norms = [curv_norm(curv_ranges[i]), av_norm(av_empty_neighbours[i])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i])), av_cmap(av_norm(av_empty_neighbours[i]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.savefig(str(i)+"_stat.png", format="png", bbox_inches='tight', pad_inches=0)
#     fig.savefig(str(i)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# # set tau-plots
# for i in range(6, 9):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title(r'$\tau$'+"=" + str(delta))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#    # fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i], decimals=3), np.around(av_empty_neighbours[i], decimals=3)]
#     norms = [curv_norm(curv_ranges[i]), av_norm(av_empty_neighbours[i])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i])), av_cmap(av_norm(av_empty_neighbours[i]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.savefig(str(i)+"_stat.png", format="png", bbox_inches='tight', pad_inches=0)
#     fig.savefig(str(i)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

# # set cplots
# for i in range(9, 12):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title("c=" + str(contact_inhib_thresh))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i], decimals=3), np.around(av_empty_neighbours[i], decimals=3)]
#     norms = [curv_norm(curv_ranges[i]), av_norm(av_empty_neighbours[i])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i])), av_cmap(av_norm(av_empty_neighbours[i]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.savefig(str(i)+"_stat.png", format="png", bbox_inches='tight', pad_inches=0)
#     fig.savefig(str(i)+".png", format="png", dpi=300) 
#     plt.show()
#     plt.clf()


# # now make anchorage dependent figure

# for i in range(4):
#     # set parameters and retrieve data
#     i_ = i+12
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+12]
#     occupations, t = np.load("sim_"+str(i+12)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title(r'$\alpha$' + "=" + str(alpha)+", " + r'$\beta$' + "=" + str(beta) +
#                  ", c=" + str(contact_inhib_thresh)+", h="+str(h)+", "+r'$\tau$'+"="+str(delta))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.savefig(str(i)+"_stat.png", format="png", bbox_inches='tight', pad_inches=0)
#     fig.savefig(str(i+12)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# # make a plot showing the effect of differentiation
# for i in range(3):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+16]
#     occupations, t = np.load("sim_"+str(i+16)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title("q="+str(prob_diff))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+16
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     plt.show()
#     fig.savefig(str(i+16)+".eps", format="eps", dpi=300)
#     plt.clf()


# # make a plot showing differentiation with absorption
# for i in range(3):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+19]
#     occupations, t = np.load("sim_"+str(i+19)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title("c="+str(contact_inhib_thresh))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+19
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+19)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

    
# # make example plots for neighbour suppression
# for i in range(2):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+22]
#     occupations, t = np.load("sim_"+str(i+22)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     i_ = i+22
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#     fig.savefig(str(i+22)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

    
# for i in range(2):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+22]
#     occupations, t = np.load("sim_"+str(i+22)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plothalf(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks([60, 130])
#     ax.set_zticks(fullticks)
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#     fig.tight_layout()
#     fig.savefig(str(i+24)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# # make a plot that shows budding behaviour emerging from decreasing alpha
# for i in range(4):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+26]
#     occupations, t = np.load("sim_"+str(i+26)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\alpha$'+"="+str(alpha))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#  #   fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+26
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+26)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

    
# # make a comparison plot for neighbour suppression

# # h plots
# for i in range(3):
#     # set parameters and retrieve data
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+30]
#     occupations, t = np.load("sim_"+str(i+30)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title("h=" + str(h))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+30
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+30)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

    
    
# # set tau-plots
# for i in range(3):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+33]
#     occupations, t = np.load("sim_"+str(i+33)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title(r'$\tau$'+"=" + str(delta))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+33
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+33)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

    
# # set cplots
# for i in range(3):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+36]
#     occupations, t = np.load("sim_"+str(i+36)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title("c=" + str(contact_inhib_thresh))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+36
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+36)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# # this saves as plot 21 for unclear reasons- every figure after that is pushed back a step
# # generate comparison plot for 'light' neighbour suppression
# for i in range(1):
#     # set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+39]
#     occupations, t = np.load("sim_"+str(i+39)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title("c="+str(contact_inhib_thresh))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     fig.savefig("NS.png", format="png")
#     plt.show()
#     plt.clf()


# # make senescence plots
# for i in range(4):
#     # set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+41]
#     occupations, t = np.load("sim_"+str(i+41)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\sigma$'+"="+str(senesc_rate))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#  #   fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i+41
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i+41)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# # make granular alpha plot- you're only using half of them
# for i in range(45, 56):
#     # set parameters and retrieve data-
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\alpha$'+"="+str(alpha))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#  #   fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str("alpha_"+str(i))+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()

# # make probability plots

# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 6}

# def p(n, alpha, beta, h, contact_inhib_thresh):
#     # compute division probability
#     return (alpha + beta/(1 + np.exp(-h*(contact_inhib_thresh-(26-n)))))*dT


# pfig = plt.figure()
# ax1 = pfig.add_subplot(211)

# ns = [i for i in range(26)]
# for h in [0.001, 0.01, 0.1, 1.0, 10]:
#     ps = [p(n, 1, 2, h, 13) for n in ns]
#     ax1.plot(ns, ps, label="h="+str(h))

# ax1.set_xlabel("Number of empty neighbour lattice-points")
# ax1.set_ylabel("Probability")

# ax2 = pfig.add_subplot(212)

# for c in [5, 10, 15, 20, 25]:
#     ps = [p(n, 1, 2, 1, c) for n in ns]
#     ax2.plot(ns, ps, label="c="+str(c))

# ax2.set_xlabel("Number of empty neighbour lattice-points")
# ax2.set_ylabel("Probability")

# pfig.tight_layout()

# pfig.savefig("pfig.png", format="png")
# plt.show()
# plt.clf()

# # make probability figures for Discussion figure

# pabsfig = plt.figure()
# ax = pabsfig.add_subplot(111)
# [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[0]  # we are using the first figure here
# ps = [p(n, alpha, beta, h, contact_inhib_thresh) for n in ns]
# ax.plot(ns, ps)
# ax.set_xlabel("Number of empty neighbour lattice-points")
# ax.set_ylabel("Division probability")

# pabsfig.savefig("pabs.png", format="png", dpi=300)
# plt.show()
# plt.clf()

# #for anchorage probability

# pancfig = plt.figure()
# ax = pancfig.add_subplot(111)
# [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[12]  # we are using the first figure here
# ps = [p(n, alpha, beta, -h, contact_inhib_thresh) for n in ns]
# ax.plot(ns, ps)
# ax.set_xlabel("Number of empty neighbour lattice-points")
# ax.set_ylabel("Division probability")

# pancfig.savefig("panc.png", format="png")
# plt.show()
# plt.clf()


# #now for differentiation
# pdifffig = plt.figure()
# ax = pdifffig.add_subplot(111)
# [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[15]  # we are using the first figure here
# ps = [alpha*dT for n in ns]
# ax.plot(ns, ps)
# ax.set_xlabel("Number of empty neighbour lattice-points")
# ax.set_ylabel("Division probability")

# pdifffig.savefig("pdiff.png", format="png")
# plt.show()
# plt.clf()

# #now for neighbour suppression

# pNSfig = plt.figure()
# ax = pNSfig.add_subplot(111)
# [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[24]  # we are using the first figure here
# ps = [p(n, alpha, beta, h, contact_inhib_thresh) for n in ns]
# ax.plot(ns, ps)
# ax.set_xlabel("Number of empty neighbour lattice-points")
# ax.set_ylabel("Division probability")

# pNSfig.savefig("pNS.png", format="png")
# plt.show()
# plt.clf()

# #now make figure for average number of neighbours
# cs, neighbour_plot, curv_plot = [], [], []
# for i in range(56, 73):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     cs.append(contact_inhib_thresh)
#     neighbour_plot.append(av_empty_neighbours[i])
#     curv_plot.append(curv_ranges[i])
# neighfig = plt.figure()
# ax = neighfig.add_subplot(111)
# ax2 = ax.twinx() #add a secondary axis
# ax.plot(cs, neighbour_plot, color="black")
# ax2.plot(cs, curv_plot, color="red")
# ax.set_xlabel("Threshold number of occupied cell neighbours")
# ax.set_xticks([10, 15, 20, 25])
# ax.set_ylabel("Average number of empty cell neighbours", color="red")
# ax2.set_ylabel("Range of curvature, " r'$\mu m^-1$')
# neighfig.savefig("neighbour_curv.eps", format="eps", dpi=300)
# plt.show()
# plt.clf()


# # set cplots- show c emerging
# for i in range(56, 73):
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     # plot graph and set axes
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_title("c=" + str(contact_inhib_thresh))
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#  #   fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#  #   fig.tight_layout()
#     i_ = i
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str(i)+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()


# for i in range(73, 76):
#     # set parameters and retrieve data-
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\alpha$' + "=" + str(alpha)+", T=" + str(np.around(t, decimals=3)) + " days")
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#    # fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#     #   fig.tight_layout()
#     i_ = i
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str("vartime"+str(i))+".eps", format="eps", dpi=300)
#     plt.show()
#     plt.clf()
    
# for i in range(76, 79):
#     # set parameters and retrieve data-
#     [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
#     occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
#     (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
#     fig = plt.figure()
#     ax = fig.add_subplot(121, projection='3d')
#     # plot in 3D
#     p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
#     # introduce labels
#     ax.set_xlabel(r'$\mu$' + "m")
#     ax.set_ylabel(r'$\mu$' + "m")
#     ax.set_zlabel(r'$\mu$' + "m")
#     ax.set_xticks(fullticks)
#     ax.set_yticks(fullticks)
#     ax.set_zticks(fullticks)
#     ax.set_title(r'$\beta$' + "=" + str(beta)+", T=" + str(np.around(t, decimals=3)) + " days")
#     # make the panes transparent
#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     # make the grid lines transparent
#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#     #   fig.tight_layout()
#     i_ = i
#     ax2 = fig.add_subplot(122)
#     #now print a map of colours with individual stats
#     stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
#     norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
#     colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
#     # Add the circles with numbers
#     # Create a 2D subplot for the circles with numbers
#     for j, stat in enumerate(stats):
#         x_pos = 0.5  # Adjust the x-position of the circles as needed
#         y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
#         ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
#         ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
#     # Set the aspect ratio of the plot to 'equal' for circular circles
#     ax2.set_aspect('equal')
#     ax2.axis('off')
#     fig.tight_layout()
#     fig.savefig(str("vartime"+str(i))+".eps", format="eps", dpi=300)
#     plt.show()
#     print(curv_ranges[i_])
#     plt.clf()

#make final figures
for i in [0, 12, 15, 24]:
    # set parameters and retrieve data-
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i, vmin=1, vmax=26)
    # introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title(r'$\alpha$' + "=" + str(alpha)+", " + r'$\beta$' + "=" + str(beta) +
                ", c=" + str(contact_inhib_thresh)+", h="+str(h)+", "+r'$\tau$'+"="+str(delta))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    #fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
    #   fig.tight_layout()
    i_ = i
    ax2 = fig.add_subplot(122)
    #now print a map of colours with individual stats
    stats = [np.around(curv_ranges[i_], decimals=3), np.around(av_empty_neighbours[i_], decimals=3)]
    norms = [curv_norm(curv_ranges[i_]), av_norm(av_empty_neighbours[i_])]
    colours = [curv_cmap(curv_norm(curv_ranges[i_])), av_cmap(av_norm(av_empty_neighbours[i_]))]
    # Add the circles with numbers
    # Create a 2D subplot for the circles with numbers
    for j, stat in enumerate(stats):
        x_pos = 0.5  # Adjust the x-position of the circles as needed
        y_pos = 0.6 - j * circle_spacing  # Adjust the y-position of the circles as needed
        ax2.add_patch(plt.Circle((x_pos, y_pos), circle_radius, color=colours[j]))
        ax2.text(x_pos, y_pos, str(stat), color='white' if norms[j] > 0.5 else 'black', ha='center', va='center')
    # Set the aspect ratio of the plot to 'equal' for circular circles
    ax2.set_aspect('equal')
    ax2.axis('off')
    fig.tight_layout()
    fig.savefig(str("comp_"+str(i))+".eps", format="eps", dpi=300)
    plt.show()
    plt.clf()
    
#separately plot and save the colorbar

print(curv_ranges)

cmap = mpl.cm.get_cmap('viridis')  
norm = plt.Normalize(vmin=1, vmax=26)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

# Create a figure and axes for the colorbar
fig, ax = plt.subplots()

# Plot the colorbar
cbar = plt.colorbar(sm, cax=ax)
ax.set_aspect('equal')

# Show the colorbar
plt.show()

fig.savefig("colorbar_neighbour.eps", format="eps", dpi=300)


# In[ ]:





# In[ ]:





# In[ ]:




