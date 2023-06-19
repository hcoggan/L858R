#!/usr/bin/env python
# coding: utf-8

# In[22]:


# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:14:35 2023
@author: 44749
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

t_as, t_ps, et_as, et_ps = [], [], [], []

with open('measurements-t-et.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in reader:
        if 0 < counter < 33: #it's a non-mutant cluster
            area, perim = float(row[2]), float(row[6])
            t_as.append(area)
            t_ps.append(perim)
        else:
            if counter >= 33: #it's a mutant cluster
                area, perim = float(row[2]), float(row[6])
                et_as.append(area)
                et_ps.append(perim)
        counter += 1

#calculate average 'stretching coefficient'
t_ratios, et_ratios = [p/np.sqrt(a) for (p, a) in zip(t_ps, t_as)], [p/np.sqrt(a) for (p, a) in zip(et_ps, et_as)]

print(t_ratios)
print(et_ratios)


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

#make a violin plot of their densities
plt.violinplot([t_ratios, et_ratios], showmeans=True, showextrema=True, showmedians=True, widths=0.7, )
plt.xticks([1.0, 2.0], ["Wild-type", "Mutant"])
plt.ylabel("Stretching coefficient")
plt.tight_layout()

violinfig = plt.gcf()

violinfig.savefig('violin.eps', format='eps', dpi=300)


et_curvatures = [0.506713586, 0.64995638, 0.387796164, 0.477550118, 0.288001231, 0.46369821, 0.374440223, 0.462171088, 0.414894981, 
                 0.319721397, 0.518051034, 0.317310297, 0.755916161]
t_curvatures = [0.288412423, 0.29791084, 0.376638707, 0.292339914, 0.435845143, 0.279141359, 0.354936945, 0.503770014, 0.305319397,
                0.382634357, 0.393427934, 0.462182964, 0.486880544, 0.296889476, 0.302899792]

plt.clf()
#make a violin plot of their densities
plt.violinplot([t_curvatures, et_curvatures], showmeans=True, showextrema=True, showmedians=True, widths=0.7, )
plt.xticks([1.0, 2.0], ["Wild-type", "Mutant"])
plt.ylabel("Average curvatures")
plt.tight_layout()

curve_figure = plt.gcf()
curve_figure.savefig("avcurvefig.eps", format='eps', dpi=300)
labels, max_neg_curvatures, max_pos_curvatures, std_curvatures = ["CURVE 1"], [0], [0], []

plt.clf()

#calculate maximum point curvatures and ranges
with open('newimage_for_ImageJ_analysis_neg.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in reader:
        if counter > 0: #skip first row
            if row[0] != labels[-1]:
                labels.append(row[0])
                max_neg_curvatures.append(0)
                max_pos_curvatures.append(0)
                std_curvatures.append(float(row[3]))
            else:
                if float(row[6]) < max_neg_curvatures[-1]:
                    max_neg_curvatures[-1] = float(row[6])
                if float(row[6]) > max_pos_curvatures[-1]:
                    max_pos_curvatures[-1] = float(row[6])
                    
        counter += 1

t_curv_mins, et_curv_mins = max_neg_curvatures[:11], max_neg_curvatures[11:]
t_curv_ranges, et_curv_ranges = [ma - mi for (ma, mi) in zip(max_pos_curvatures[:11], max_neg_curvatures[:11])],  [ma - mi for (ma, mi) in zip(max_pos_curvatures[11:], max_neg_curvatures[11:])]

print(max(t_curv_ranges))

#make a violin plot of curvature ranges
plt.violinplot([t_curv_ranges, et_curv_ranges], showmeans=True, showextrema=True, showmedians=True, widths=0.7, )
plt.xticks([1.0, 2.0], ["Wild-type", "Mutant"])
plt.ylabel("Range of curvature, "+ r'$\mu m^-1$')
plt.tight_layout()

maxcurv = plt.gcf()
maxcurv.savefig("rangecurvefig.eps", format='eps', dpi=300)

plt.clf()

#plot the relationship between perimeter and area

plt.scatter(t_as, t_ps, label="Wild-type")
plt.scatter(et_as, et_ps, label="Mutant")

#calculate 'circular' relationship
max_a = max(max(t_as), max(et_as))

trial_as = np.linspace(0, max_a, 100)
ideal_ps = [2*np.pi*np.sqrt(a/np.pi) for a in trial_as]

plt.plot(trial_as, ideal_ps, '--', color='green')

plt.xlabel("Cross-sectional area, " + r'$\mu m^2$')
plt.ylabel("Perimeter, " + r'$\mu$' + "m")
plt.legend()

plt.show()

perimfig = plt.gcf()
perimfig.savefig("perimfig.eps", format="eps", dpi=300)

"""
#scatterplot this in 2D
plt.scatter(np.sqrt(t_as), t_ps, label = "non-mutant")
plt.scatter(np.sqrt(et_as), et_ps, label = "mutant")
plt.xlabel("sqrt. area, um")
plt.ylabel("perimeter, um")
plt.legend()
plt.show()
"""


# In[24]:


#get curvature ranges for all simulations

curv_ranges = np.zeros(79)



indices_not_used = [25, 39, 40, 41, 46, 48, 50, 52, 54]
for i in range(79):
    if i not in indices_not_used:
        with open(str(i)+'.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            counter = 0
            max_curv, min_curv = 0, 0
            for row in reader:
                if counter > 0:
                    if float(row[6]) < min_curv:
                        min_curv = float(row[6])
                    if float(row[6]) > max_curv:
                        max_curv = float(row[6])
                counter += 1
            curv_ranges[i] = max_curv-min_curv

np.save("curvature_ranges_sim.npy", curv_ranges)


            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




