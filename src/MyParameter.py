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
    shiftIcX = 0
    shiftIcY = 0
    dx = 0.2228
    dy = 0.2228
    dz = 0.7000  
    dn = .1
    Nz = 50
    Nx = 120
    Ny = 120
    mysize = np.array((Nz, Nx, Ny))
    zernikefactors = np.array((0,0,0,0,0,0,0,0,0,0,0))
    zernikemask = np.zeros(zernikefactors.shape)
    mysubsamplingIC = 0
    
    # Reconstruciton parameters 
    cc_center = mysize[1:-1]//2 #np.array((0,0))
    cc_size = np.array((600,600))
    
    def __init__(self, lambda0 = 0.65, nEmbb = 1.33, NAc = 0.52, NAci = 0, 
                 NAo=.95, shiftIcX = 0, shiftIcY = 0,
                 dx = 0.2228, dy = 0.2228, dz = 0.2228, dn = .1,
                 Nx = 120, Ny = 120, Nz = 50, mysubsamplingIC=0):
        
        # Parameter class for all the system's parameters
        self.lambda0 = lambda0
        self.nEmbb = nEmbb
        self.NAc = NAc
        self.NAci = NAci
        self.NAo = NAo
        self.shiftIcX = shiftIcX
        self.shiftIcY = shiftIcY
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.Nz = Nz
        self.Nx = Nx
        self.Ny = Ny
        self.dn = dn
        self.mysize = np.array((Nz, Nx, Ny))
        self.zernikefactors = np.array((0,0,0,0,0,0,0,0,0,0,0))
        self.zernikemask = np.zeros(self.zernikefactors.shape)
        self.mysubsamplingIC = mysubsamplingIC
        
        # Reconstruciton parameters 
        self.cc_center = self.mysize[1:-1]//2 #np.array((0,0))
        self.cc_size = np.array((600,600))
        
        
    def print(self):
        '''This Function just plots all values'''
        print('Lambda0: '+str(self.lambda0))
        print('nEmbb: '+str(self.nEmbb))
        print('NAc: '+str(self.NAc))
        print('NAci: '+str(self.NAci))
        print('NAo: '+str(self.NAo))
        print('shiftIcX: '+str(self.shiftIcX))
        print('shiftIcY: '+str(self.shiftIcY))
        print('dx: '+str(self.dx))
        print('dy: '+str(self.dy))
        print('dz: '+str(self.dz))
        print('Nx: '+str(self.Nx))
        print('Ny: '+str(self.Ny))
        print('Nz: '+str(self.Nz))   
        print('mysize: ' +str(self.mysize))
        print('zernikefactors: '+str(self.zernikefactors))  
        print('zernikemask: '+str(self.zernikemask))  
    
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
        self.Nx=np.int(np.floor((2*self.Rsim)/self.dx));
        self.Ny=np.int(np.floor((2*self.Rsim)/self.dy))
         

    def preset_40x(self, dz = .3):
        ''' This presets data for the 40x '''

        print('We set the default parameters for the 40x lens from NIKON')
        # NA of the condenser lens
        self.NAc = .32
        self.NAo = .95

        # taken from the Q-PHase setup itself
        self.fov_x = 93.587; # mum
        self.fov_y = 93.587;
        self.Nx_orig = 2048; # corresponds to reconstructed hologram.
        self.Ny_orig = 2048;
        self.dx_orig = self.fov_x/self.Nx_orig
        self.dy_orig = self.fov_y/self.Ny_orig
        self.dx = self.dx_orig
        self.dy = self.dy_orig
        
        # FOV parameters
        mysubsample = 1;
        self.dz_orig = dz # % mum
        self.dz = self.dz_orig*mysubsample; # % mum
        
    
