#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:00:28 2019

@author: bene
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Python port of https://bl.ocks.org/mbostock/19168c663618b7f07158

from math import *
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

#---------------------------
class PoissonDiskSampler:
  #---------------------------
  def __init__(self,width, height, radius):

    self.k = 200 # max nb of candidate generation attempts before deactivation of a sample
    self.radius2 = radius**2
    self.R = 3 * self.radius2 # distance de sampling maximale ? Pourquoi *3 ? Pourquoi **2 ?
    self.cellSize = radius * sqrt(0.5)
    self.width = width
    self.height = height

    # This grid is a trick to faster reject points that are too close
    self.gridW = ceil(width / self.cellSize)+1
    self.gridH = ceil(height / self.cellSize)+1
    #print "grid creation",self.gridW,self.gridH,width, height, self.cellSize
    self.grid = np.ones((int(self.gridW),int(self.gridH),2)) * -1

    self.samples = [] # list of the generated samples
    self.activeSamples = []  # list of the active samples

  #---------------------------
  # tries to add a new point to the samples
  # returns False if it did not managed to do so
  def addOneSample(self):
    if len(self.samples) == 0:
      # if no sample yet, choose a point anywhere in the box
      print("No samples yet, I will create one")
      self.acceptCandidate(rnd.random() * self.width, rnd.random() * self.height)
    elif len(self.activeSamples) !=0 :
      # queue is not empty, activate a sample from the queue and try to generate a new sample from it
      # select one active sample
      s = rnd.choice(self.activeSamples)
      #print "Trying to generate a candidate close to",s

      self.generateCandidate(s,0)
    else:
    # there are samples, but the queue is empty => impossible to add one, sampling finished !
      print("No active samples anymore, I'm done.")
      return False

    return True

  #---------------------------
  # generates a candidate and tests whether it is ok
  def generateCandidate(self,s,j):
    nb = j+1
    # if we made more than k attempts to generate a candidate, 
    # remove the active sample from the queue
    if nb>self.k :
      #print "removing",s,"from queue"
      self.activeSamples.remove(s)
    else:
      # from the active sample, draw a new point using polar representation
      a = 2 * pi * rnd.random() # angle
      r = sqrt(rnd.random() * self.R + self.radius2) # radius: HERE, I do not understand the **2... A question of 2d uniformity?
      x = s[0] + r * cos(a)
      y = s[1] + r * sin(a)

      # if outside box, generate new candidate
      if (x<0 or x>=self.width or y<0 or y>=self.height):
        self.generateCandidate(s,nb)
      # otherwise: if acceptable, add sample ; if not, generate new candidate
      elif self.far(x,y):
        self.acceptCandidate(x,y)
      else:
        self.generateCandidate(s,nb)
  #---------------------------
  def acceptCandidate(self,x,y):
    #print "New sample!",x,y,int(ceil(x/self.cellSize)),int(ceil(y/self.cellSize))
    self.activeSamples.append([x,y])
    self.samples.append([x,y])
    self.grid[int(ceil(x/self.cellSize)),int(ceil(y/self.cellSize))] = [x,y]

  #---------------------------
  # returns False/True if the proposed coordinates are too close/not to existing samples
  def far(self,x,y):
    i = x / self.cellSize
    j = y / self.cellSize
    i0 = int(max(i-2, 0))
    j0 = int(max(j-2,0))
    i1 = int(min(i+3, self.gridW))
    j1 = int(min(j+3, self.gridH))

    # look in the neighboring cells if samples are already present
    # if so, check that they are not too close
    for j in range(j0,j1):
      for i in range(i0,i1):
        sx = self.grid[i,j,0]
        sy = self.grid[i,j,1]
        if [sx,sy] != [-1,-1]:
          dx = sx - x
          dy = sy - y
          if (dx**2 + dy**2) < self.radius2:
            return False
    return True

  #---------------------------
  def plot(self):
    fig = plt.figure(facecolor='white')
    ax = fig.add_axes([0.005,0.005,.99,.99], frameon=True, aspect=1)
    s = np.array(self.samples)
    scat = ax.scatter(s[:,0],s[:,1])
    plt.xlim(0,self.width)
    plt.ylim(0,self.height)
    plt.show()    

  #---------------------------
  def fullSampling(self):
    while self.addOneSample():
      pass


#--------------------------
def get_poisson_disk(Nx=64, Ny=64, radius=5):
    PDS = PoissonDiskSampler(Nx,Ny,radius)
    PDS_map = np.zeros((Nx,Ny))
    while PDS.addOneSample():
        #print len(PDS.samples)
        pass
    
    samples = np.array(PDS.samples)
    PDS_map[np.int32(samples[:,0]),np.int32(samples[:,1])]=1
    
    return PDS_map
