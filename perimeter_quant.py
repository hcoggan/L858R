# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:14:35 2023

@author: 44749
"""

import numpy as np
from matplotlib import pyplot as plt
import csv

t_as, t_ps, et_as, et_ps = [], [], [], []

with open('measurements-t-et.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in reader:
        if 0 < counter < 33: #it's a non-mutant cluster
            print(len(row))
            area, perim = float(row[3]), float(row[6])
            t_as.append(area)
            t_ps.append(perim)
            print(row[0])
        else:
            if counter >= 33: #it's a mutant cluster
                area, perim = float(row[3]), float(row[6])
                et_as.append(area)
                et_ps.append(perim)
        counter += 1

#calculate average 'stretching coefficient'
t_ratio, et_ratio = np.average([p/np.sqrt(a) for (p, a) in zip(t_ps, t_as)]), np.average([p/np.sqrt(a) for (p, a) in zip(et_ps, et_as)])

print(t_ratio, et_ratio, 2*np.sqrt(np.pi))

#plot this in 2D
plt.scatter(np.sqrt(t_as), t_ps, label = "non-mutant")
plt.scatter(np.sqrt(et_as), et_ps, label = "mutant")
plt.xlabel("sqrt. area, um")
plt.ylabel("perimeter, um")
plt.legend()
plt.show()