#!/usr/bin/env python3
# -*- coding = utf-8 -*-
"""
Created on Sat Apr 20 19 =05 =34 2019

@author = bene
"""
import numpy as np
from src import data as data
 

class MyParameter:
    # Parameter class for all the system's parameters
    lambda0 = 0.6500
    nEmbb = 1.3300
    NAc = 0.5200
    NAci = 0
    NAo = 0.9500
    shiftX = 0
    shiftY = 0
    dx = 0.2228
    dy = 0.2228
    dz = 0.7000    
    Nz = 34
    Nx = 240
    Ny = 240
    
    def loadmat(self, mymatpath='./Data/DROPLETS/S19_multiple/Parameter.mat', mymatname='myParameter'):
        # Cast the parameter-mat file into a python class
        self.matpath = mymatpath
        self.matname = mymatname
        
        myParamter = data.import_parameters_mat(filename = self.matpath, matname = self.matname)

        # ASsign variables from Experiment
        self.lambda0 = np.squeeze(np.array(myParamter.get('lambda0')))                 # free space wavelength (µm)
        self.NAo= np.squeeze(np.array(myParamter.get('NAo'))); # Numerical aperture objective
        self.NAc= np.squeeze(np.array(myParamter.get('NAc'))); # Numerical aperture condenser
        try:
            self.NAci = np.squeeze(np.array(myParamter.get('NAci'))); # Numerical aperture condenser
        except: 
            print('No inner NA has been defined!')
            self.NAci = 0
                
         
        # eventually decenter the illumination source - only integer!
        self.shiftIcX = 0#int(np.squeeze(np.array(myParamter.get('shiftIcX'))))
        self.shiftIcY = 0#int(np.squeeze(np.array(myParamter.get('shiftIcY'))))
         
        self.nEmbb = np.squeeze(np.array(myParamter.get('nEmbb'))) 
        self.dn=.1; # self.nImm - self.nEmbb
        print('Assigned some value for dn which is not good!')
         
        # calculate pixelsize
        self.dx = np.double(np.squeeze(np.array(myParamter.get('dx'))))
        self.dy = np.double(np.array(myParamter.get('dy')))
        self.dz = np.double(np.array(myParamter.get('dz')))
             
        # Sampling coordinates
        self.Rsim= 0.5*np.double(np.array(myParamter.get('Nx')))*self.dx; # Radius over which simulation is performed.
         
        self.Nz=int(np.double(np.array(myParamter.get('Nz'))))
        self.Nx=np.int(np.floor((2*self.Rsim)/self.dx)+1);
        self.Ny=np.int(np.floor((2*self.Rsim)/self.dy)+1)
         
