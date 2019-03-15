#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:57:41 2019

@author: bene
"""
import tensorflow as tf
import numpy as np
import tf_helper as tf_helper
import matplotlib.pyplot as plt
import data as data

myobj = (tf_helper.rr((128,128))<5)*.2;
mydelta = myobj*0. + 0j;
mydelta[40,40] = 1j*1.;
#%mydelta = mydelta/sqrt(abssqr(prod(size(mydelta))));
tf_myres = tf.ifft2d(tf.fft2d(myobj)*tf.fft2d(mydelta));
sess = tf.Session()
myres = sess.run(tf_myres)



#myres = myres*np.sqrt(np.prod(myres.shape)) not necessary in tf
plt.imshow(np.real(myres)), plt.colorbar(), plt.show()
plt.imshow(np.imag(myres)), plt.colorbar(), plt.show()


#%%
mymatfile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/MATLAB/MuScat/mtfs.mat'
myatf = data.import_realdata_h5(filename = mymatfile, matname='myatf', is_complex=True)
myasf = data.import_realdata_h5(filename = mymatfile, matname='myasf', is_complex=True)

def myift3d(x):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))*np.prod(x.shape)

def myft3d(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))/np.prod(x.shape)




mysample = (tf_helper.rr(myasf.shape)<1)*.1
mysample = 1.33+mysample + 0j
k02 = (2*np.pi/.65)**2
myV = k02*(mysample**2-1.33**2)
#myres = myift3d(myft3d(myV)*myft3d(myasf))


TF_myV = tf.cast(myV,tf.complex64)
TF_myasf = tf.placeholder(dtype=tf.complex64, shape=myasf.shape)
myfftfac = np.prod(myres.shape)
TF_myres = tf_helper.my_ift3d(tf_helper.my_ft3d(TF_myV,myfftfac)*tf_helper.my_ft3d(TF_myasf,myfftfac),myfftfac)

myres = sess.run(TF_myres, feed_dict={TF_myasf:myasf})

plt.imshow(np.real(myV[:,:,16])), plt.colorbar(), plt.show()
plt.imshow(np.real(myres[:,:,16])), plt.colorbar(), plt.show()


#myobj = (rr<40)*1;
#mydelta = newim(myobj, 'scomplex');
#mydelta(40,40) = .1i;
#%mydelta = mydelta/sqrt(abssqr(prod(size(mydelta))));
#myres = ift(ft(myobj)*ft(mydelta));
#myres = myres*sqrt(prod(size(myres)));
#real(myres)
#imag(myres)