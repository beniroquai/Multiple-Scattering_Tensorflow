#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:57:00 2019

@author: bene
"""


#%% Check the monotonic function for graphical correctness
import numpy as np
import matplotlib.pyplot as plt


def PreMonotonicPos(tfin):
    monoPos=((np.sign(tfin-1) + 1)/2)*(tfin)+((1-np.sign(tfin-1))/2)*(1.0/(2-tfin))
    return monoPos

myinput = np.array(range(-100,100,1))/10.0+0.1

mycalc=PreMonotonicPos(myinput)

plt.plot(myinput, mycalc, 'b.', label='piecewise') # "bo" is for "blue dot"
plt.title('Piecewise')
plt.xlabel('myinput'); plt.ylabel('output'); plt.legend(); plt.show()

