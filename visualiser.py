# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:48:30 2023

@author: 44749
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.transforms import offset_copy

#set font
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

v_0 = 80
r_0 = np.cbrt(v_0)


n = 30 #number of cells in each direction
N = n**3 #number of cells total


#alpha = 0.0
dT = 0.01

[xs, ys, zs] = np.load("square_grid_coords_30.npy", allow_pickle=True) #load grid
neighbours = np.load("square_neighbour_list_30.npy", allow_pickle=True) #load neighbours

paramsets = [[1, 1, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 20, 1, 10000], [1, 5, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 0.01, 20, 1, 10000], [1, 2, 1.0, 0, 0.1, 20, 1, 10000], [1, 2, 1.0, 0, 10, 20, 1, 10000],\
             [1, 2, 0.1, 0, 1, 20, 1, 10000], [1, 2, 0.5, 0, 1, 20, 1, 10000], [1, 2, 2.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 10, 1, 10000], [1, 2, 1.0, 0, 1, 15, 1, 10000], [1, 2, 1.0, 0, 1, 25, 1, 10000], \
             [1, 10, 10, 0, -1, 1, 1, 10000], [1, 10, 10, 0, -1, 25, 1, 10000], [0, 1, 1.0, 0, -1, 1, 1, 10000], [1, 1, 1.0, 0, -5, 2, 1, 10000], [1, 0, 1.0, 0, 1, 1, 0.3, 10000], [1, 0, 1.0, 0, 1, 1, 0.5, 10000], [1, 0, 1.0, 0, 1, 1, 0.7, 10000],\
           [1, 5, 1.0, 0, 1, 15, 0.3, 10000], [1, 5, 1.0, 0, 1, 20, 0.3, 10000], [1, 5, 1.0, 0, 1, 25, 0.3, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000],\
               [1, 10, 1.0, 0, 1, 10, 1, 10000], [0.5, 10, 1.0, 0, 1, 10, 1, 10000], [0.1, 10, 1.0, 0, 1, 10, 1, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 0.1, 10, 1, 5000], [0, 10, 1.0, 0, 0.5, 10, 1, 5000], [0, 10, 1.0, 0, 1.5, 10, 1, 5000],\
                [0, 10, 0.1, 0, 1.0, 10, 1, 5000], [0, 1, 0.5, 0, 1.0, 10, 1, 5000], [0, 10, 2.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 15, 1, 5000], [0, 10, 1.0, 0, 1.0, 20, 1, 5000], [0, 5, 1.0, 0, 1.0, 25, 1, 10000],
            [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.5, 1.0, 10, 1, 10000], [1, 0, 1.0, 1.0, 1.0, 10, 1, 5000], [1, 0, 1.0, 2.0, 1.0, 10, 1, 5000]]
#40 is a duplicate of 41
    

    
#function to take data and produce points and colours to plot

def plotfull(occupations):
    to_plot = list(filter(lambda i: (occupations[i]==1), np.arange(N))) #find all occupied cells
    xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [ys[i] for i in to_plot], [zs[i] for i in to_plot]
    #neighbour number
    cs_to_plot = [len(list(filter(lambda i: (occupations[i]==1), neighbours[n]))) for n in to_plot]
    return (xs_to_plot, ys_to_plot, zs_to_plot, cs_to_plot)

#function to produce a cross-section
def plothalf(occupations):
    line = int(max(ys)/2)
    to_plot = list(filter(lambda i: (occupations[i]==1 and ys[i] > line), np.arange(N))) #find all occupied cells
    xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [ys[i] for i in to_plot], [zs[i] for i in to_plot]
    #neighbour number
    cs_to_plot = [len(list(filter(lambda i: (occupations[i]==1), neighbours[n]))) for n in to_plot]
    return (xs_to_plot, ys_to_plot, zs_to_plot, cs_to_plot)

fullticks = [0, 65, 130]

#make plots showing the effect of absorption-dependent growth

for i in range(3):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title(r'$\beta$'+"="+str(beta)) 
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dp1=1200)
    
    
#set hplots
for i in range(3, 6):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    #plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title("h=" + str(h))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dp1=1200)


#set tau-plots
for i in range(6, 9):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] =  paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    #plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p= ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title(r'$\tau$'+"=" + str(delta))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dp1=1200)
    
#set cplots
for i in range(9, 12):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    #plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title("c=" + str(contact_inhib_thresh))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dp1=1200)




#now make anchorage dependent figure

for i in range(4):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+12]
    occupations, t = np.load("sim_"+str(i+12)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p= ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title(r'$\alpha$' + "=" + str(alpha)+", "+ r'$\beta$' + "=" + str(beta) +", c=" + str(contact_inhib_thresh)+", h="+str(h)+", "+r'$\tau$'+"="+str(delta))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+12)+".eps", format="eps", dp1=1200)


#make a plot showing the effect of differentiation
for i in range(3):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+16]
    occupations, t = np.load("sim_"+str(i+16)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title("q="+str(prob_diff)) 
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+16)+".eps", format="eps", dp1=1200)


#make a plot showing differentiation with absorption
for i in range(3):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+19]
    occupations, t = np.load("sim_"+str(i+19)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title("c="+str(contact_inhib_thresh)) 
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+19)+".eps", format="eps", dp1=1200)

#make example plots for neighbour suppression
for i in range(4):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+22]
    occupations, t = np.load("sim_"+str(i+22)+".npy", allow_pickle=True)
    if i < 2:
        (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    else:
        (xs_i, ys_i, zs_i, cs_i) = plothalf(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    if i < 2:
        ax.set_yticks(fullticks)
    else:
        ax.set_yticks([60, 130])
    ax.set_zticks(fullticks)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+22)+".eps", format="eps", dp1=1200)


#make a plot that shows budding behaviour emerging from decreasing alpha
for i in range(4):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+26]
    occupations, t = np.load("sim_"+str(i+26)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title(r'$\alpha$'+"="+str(alpha)) 
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+26)+".eps", format="eps", dp1=1200)

#make a comparison plot for neighbour suppression

#h plots
for i in range(3):
    #set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+30]
    occupations, t = np.load("sim_"+str(i+30)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title("h=" + str(h))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+30)+".eps", format="eps", dp1=1200)

#set tau-plots
for i in range(3):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] =  paramsets[i+33]
    occupations, t = np.load("sim_"+str(i+33)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    #plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title(r'$\tau$'+"=" + str(delta))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+33)+".eps", format="eps", dp1=1200)
#set cplots
for i in range(3):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+36]
    occupations, t = np.load("sim_"+str(i+36)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    #plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_title("c=" + str(contact_inhib_thresh))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+36)+".eps", format="eps", dp1=1200)


#this saves as plot 21 for unclear reasons- every figure after that is pushed back a step
#generate comparison plot for 'light' neighbour suppression
for i in range(1):
    #set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] =  paramsets[i+39]
    occupations, t = np.load("sim_"+str(i+39)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    ax = plt.axes(projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title("c="+str(contact_inhib_thresh))
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig("NS.eps", format="eps", dp1=1200)



#make senescence plots
for i in range(4):
    #set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff, no_cells_max] = paramsets[i+41]
    occupations, t = np.load("sim_"+str(i+41)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    #introduce labels
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_title(r'$\sigma$'+"="+str(senesc_rate)) 
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+41)+".eps", format="eps", dp1=1200)

