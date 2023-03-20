# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:48:30 2023

@author: 44749
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.transforms import offset_copy

# set font
font = {'family': 'normal',
        'weight': 'normal',
        'size': 15}

matplotlib.rc('font', **font)

v_0 = 80
r_0 = np.cbrt(v_0)


n = 30  # number of cells in each direction
N = n**3  # number of cells total

dT = 0.01

#alpha = 0.0
dT = 0.01

[xs, ys, zs] = np.load("square_grid_coords_30.npy",
                       allow_pickle=True)  # load grid
neighbours = np.load("square_neighbour_list_30.npy",
                     allow_pickle=True)  # load neighbours

paramsets = [[1, 1, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 20, 1, 10000], [1, 5, 1.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 0.01, 20, 1, 10000], [1, 2, 1.0, 0, 0.1, 20, 1, 10000], [1, 2, 1.0, 0, 10, 20, 1, 10000],\
             [1, 2, 0.1, 0, 1, 20, 1, 10000], [1, 2, 0.5, 0, 1, 20, 1, 10000], [1, 2, 2.0, 0, 1, 20, 1, 10000], [1, 2, 1.0, 0, 1, 10, 1, 10000], [1, 2, 1.0, 0, 1, 15, 1, 10000], [1, 2, 1.0, 0, 1, 25, 1, 10000], \
             [1, 10, 10, 0, -1, 1, 1, 10000], [1, 10, 10, 0, -1, 25, 1, 10000], [0, 1, 1.0, 0, -1, 1, 1, 10000], [1, 1, 1.0, 0, -5, 2, 1, 10000], [1, 0, 1.0, 0, 1, 1, 0.3, 10000], [1, 0, 1.0, 0, 1, 1, 0.5, 10000], [1, 0, 1.0, 0, 1, 1, 0.7, 10000],\
           [1, 5, 1.0, 0, 1, 15, 0.3, 10000], [1, 5, 1.0, 0, 1, 20, 0.3, 10000], [1, 5, 1.0, 0, 1, 25, 0.3, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 1, 10, 1, 5000],\
               [1, 10, 1.0, 0, 1, 10, 1, 10000], [0.5, 10, 1.0, 0, 1, 10, 1, 10000], [0.1, 10, 1.0, 0, 1, 10, 1, 10000], [0, 10, 1.0, 0, 1, 10, 1, 5000], [0, 10, 1.0, 0, 0.1, 10, 1, 5000], [0, 10, 1.0, 0, 0.5, 10, 1, 5000], [0, 10, 1.0, 0, 1.5, 10, 1, 5000],\
                [0, 10, 0.1, 0, 1.0, 10, 1, 5000], [0, 1, 0.5, 0, 1.0, 10, 1, 5000], [0, 10, 2.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 15, 1, 5000], [0, 10, 1.0, 0, 1.0, 20, 1, 5000], [0, 5, 1.0, 0, 1.0, 25, 1, 10000],
            [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.1, 1.0, 10, 1, 10000], [1, 0, 1.0, 0.5, 1.0, 10, 1, 10000], [1, 0, 1.0, 1.0, 1.0, 10, 1, 5000], [1, 0, 1.0, 2.0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.01, 10, 1.0, 0, 1.0, 10, 1, 5000],
            [0.02, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.03, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.04, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.05, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.06, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.07, 10, 1.0, 0, 1.0, 10, 1, 5000], [0.08, 10, 1.0, 0, 1.0, 10, 1, 5000],
            [0.09, 10, 1.0, 0, 1.0, 10, 1, 5000],  [0.1, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 10, 1, 5000], [0, 10, 1.0, 0, 1.0, 11, 1, 5000], [0, 10, 1.0, 0, 1.0, 12, 1, 5000], [0, 10, 1.0, 0, 1.0, 13, 1, 5000], [0, 10, 1.0, 0, 1.0, 14, 1, 5000], [0, 10, 1.0, 0, 1.0, 15, 1, 5000], [0, 10, 1.0, 0, 1.0, 16, 1, 5000],
            [0, 10, 1.0, 0, 1.0, 17, 1, 5000], [0, 10, 1.0, 0, 1.0, 18, 1, 5000], [0, 10, 1.0, 0, 1.0, 19, 1, 5000], [0, 10, 1.0, 0, 1.0, 20, 1, 5000], [0, 10, 1.0, 0, 1.0, 21, 1, 5000], [0, 10, 1.0, 0, 1.0, 22, 1, 5000], [0, 10, 1.0, 0, 1.0, 23, 1, 5000], [0, 10, 1.0, 0, 1.0, 24, 1, 5000], [0, 10, 1.0, 0, 1.0, 25, 1, 5000], [0, 10, 1.0, 0, 1.0, 26, 1, 5000]]


# 40 is a duplicate of 41


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
    xs_to_plot, ys_to_plot, zs_to_plot = [xs[i] for i in to_plot], [
        ys[i] for i in to_plot], [zs[i] for i in to_plot]
    # neighbour number
    cs_to_plot = [len(
        list(filter(lambda i: (occupations[i] == 1), neighbours[n]))) for n in to_plot]
    return (xs_to_plot, ys_to_plot, zs_to_plot, cs_to_plot)


fullticks = [0, 65, 130]

# make plots showing the effect of absorption-dependent growth

for i in range(3):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dpi=1200)


# set hplots
for i in range(3, 6):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    # plot graph and set axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dpi=1200)


# set tau-plots
for i in range(6, 9):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    # plot graph and set axes
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dpi=1200)

# set cplots
for i in range(9, 12):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    # plot graph and set axes
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i)+".eps", format="eps", dpi=1200)


# now make anchorage dependent figure

for i in range(4):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+12]
    occupations, t = np.load("sim_"+str(i+12)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    ax.set_xticks(fullticks)
    ax.set_yticks(fullticks)
    ax.set_zticks(fullticks)
    ax.set_xlabel(r'$\mu$' + "m")
    ax.set_ylabel(r'$\mu$' + "m")
    ax.set_zlabel(r'$\mu$' + "m")
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
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+12)+".eps", format="eps", dpi=1200)


# make a plot showing the effect of differentiation
for i in range(3):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+16]
    occupations, t = np.load("sim_"+str(i+16)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+16)+".eps", format="eps", dpi=1200)


# make a plot showing differentiation with absorption
for i in range(3):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+19]
    occupations, t = np.load("sim_"+str(i+19)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+19)+".eps", format="eps", dpi=1200)

# make example plots for neighbour suppression
for i in range(4):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+22]
    occupations, t = np.load("sim_"+str(i+22)+".npy", allow_pickle=True)
    if i < 2:
        (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    else:
        (xs_i, ys_i, zs_i, cs_i) = plothalf(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+22)+".eps", format="eps", dpi=1200)


# make a plot that shows budding behaviour emerging from decreasing alpha
for i in range(4):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+26]
    occupations, t = np.load("sim_"+str(i+26)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+26)+".eps", format="eps", dpi=1200)

# make a comparison plot for neighbour suppression

# h plots
for i in range(3):
    # set parameters and retrieve data
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+30]
    occupations, t = np.load("sim_"+str(i+30)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+30)+".eps", format="eps", dpi=1200)

# set tau-plots
for i in range(3):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+33]
    occupations, t = np.load("sim_"+str(i+33)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    # plot graph and set axes
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+33)+".eps", format="eps", dpi=1200)
# set cplots
for i in range(3):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+36]
    occupations, t = np.load("sim_"+str(i+36)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    # plot graph and set axes
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+36)+".eps", format="eps", dpi=300)


# this saves as plot 21 for unclear reasons- every figure after that is pushed back a step
# generate comparison plot for 'light' neighbour suppression
for i in range(1):
    # set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+39]
    occupations, t = np.load("sim_"+str(i+39)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    ax = plt.axes(projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig("NS.eps", format="eps", dpi=1200)


# make senescence plots
for i in range(4):
    # set parameters and retrieve data- skip to 40 because 39 is an accidental repeat
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i+41]
    occupations, t = np.load("sim_"+str(i+41)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str(i+41)+".eps", format="eps", dpi=1200)


# make granular alpha plot
for i in range(45, 56):
    # set parameters and retrieve data-
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
        prob_diff, no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot in 3D
    p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
    # introduce labels
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
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
 #   fig.tight_layout()
    fig.savefig(str("alpha_"+str(i))+".eps", format="eps", dpi=400)

# make probability plots

font = {'family': 'normal',
        'weight': 'normal',
        'size': 6}

def p(n, alpha, beta, h, contact_inhib_thresh):
    # compute division probability
    return (alpha + beta/(1 + np.exp(-h*(contact_inhib_thresh-(26-n)))))*dT


pfig = plt.figure()
ax1 = pfig.add_subplot(211)

ns = [i for i in range(26)]
for h in [0.001, 0.01, 0.1, 1.0, 10]:
    ps = [p(n, 1, 2, h, 13) for n in ns]
    ax1.plot(ns, ps, label="h="+str(h))

ax1.set_xlabel("Number of empty neighbour lattice-points")
ax1.set_ylabel("Probability")

ax2 = pfig.add_subplot(212)

for c in [5, 10, 15, 20, 25]:
    ps = [p(n, 1, 2, 1, c) for n in ns]
    ax2.plot(ns, ps, label="c="+str(c))

ax2.set_xlabel("Number of empty neighbour lattice-points")
ax2.set_ylabel("Probability")

pfig.tight_layout()

pfig.savefig("pfig.eps", format="eps", dpi=300)

# make probability figures for Discussion figure

pabsfig = plt.figure()
ax = pabsfig.add_subplot(111)
[alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff,
    no_cells_max] = paramsets[0]  # we are using the first figure here
ps = [p(n, alpha, beta, h, contact_inhib_thresh) for n in ns]
ax.plot(ns, ps)
ax.set_xlabel("Number of empty neighbour lattice-points")
ax.set_ylabel("Division probability")

pabsfig.savefig("pabs.eps", format="eps", dpi=300)

#for anchorage probability

pancfig = plt.figure()
ax = pancfig.add_subplot(111)
[alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff,
    no_cells_max] = paramsets[12]  # we are using the first figure here
ps = [p(n, alpha, beta, -h, contact_inhib_thresh) for n in ns]
ax.plot(ns, ps)
ax.set_xlabel("Number of empty neighbour lattice-points")
ax.set_ylabel("Division probability")

pancfig.savefig("panc.eps", format="eps", dpi=300)


#now for differentiation
pdifffig = plt.figure()
ax = pdifffig.add_subplot(111)
[alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff,
    no_cells_max] = paramsets[15]  # we are using the first figure here
ps = [alpha*dT for n in ns]
ax.plot(ns, ps)
ax.set_xlabel("Number of empty neighbour lattice-points")
ax.set_ylabel("Division probability")

pdifffig.savefig("pdiff.eps", format="eps", dpi=300)

#now for neighbour suppression

pNSfig = plt.figure()
ax = pNSfig.add_subplot(111)
[alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff,
    no_cells_max] = paramsets[24]  # we are using the first figure here
ps = [p(n, alpha, beta, h, contact_inhib_thresh) for n in ns]
ax.plot(ns, ps)
ax.set_xlabel("Number of empty neighbour lattice-points")
ax.set_ylabel("Division probability")

pNSfig.savefig("pNS.eps", format="eps", dpi=300)

#now make figure for average number of neighbours
cs, av_neighbour_no = [], []
for i in range(56, 73):
    [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh, prob_diff,
        no_cells_max] = paramsets[i]
    occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
    no_cells, no_neighbours = sum(occupations), 0
    for n in range(N): #all of these should have occupation number 1 or 0
        if occupations[n] == 1:
            occupied_neighbour_no = len(list(filter(lambda i: (occupations[i]==1), neighbours[n]))) #how many neighbours does a cell have?
            no_neighbours += occupied_neighbour_no
    no_neighbours /= no_cells
    cs.append(contact_inhib_thresh)
    av_neighbour_no.append(no_neighbours)
neighfig = plt.figure()
ax = neighfig.add_subplot(111)
ax.scatter(cs, av_neighbour_no)
ax.set_xlabel("Threshold number of occupied cell neighbours")
ax.set_xticks([10, 15, 20, 25])
ax.set_ylabel("Average number of occupied cell neighbours")
neighfig.savefig("neighbour.eps", format="eps", dpi=300)

#make final figures
for i in [0, 12, 15, 24]:
   # set parameters and retrieve data-
   [alpha, beta, delta, senesc_rate, h, contact_inhib_thresh,
       prob_diff, no_cells_max] = paramsets[i]
   occupations, t = np.load("sim_"+str(i)+".npy", allow_pickle=True)
   (xs_i, ys_i, zs_i, cs_i) = plotfull(occupations)
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   # plot in 3D
   p = ax.scatter(xs_i, ys_i, zs_i, c=cs_i)
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
   fig.colorbar(p, ticks=[5, 10, 15, 20, 25], pad=0.2)
#   fig.tight_layout()
   fig.savefig(str("comp_"+str(i))+".eps", format="eps", dpi=400)
   