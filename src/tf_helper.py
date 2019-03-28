# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:53:08 2017

@author: Bene
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy.io
import time
import scipy as scipy
from scipy import ndimage
import h5py 
from tensorflow.python.client import device_lib
import scipy.misc
import numbers



defaultTFDataType="float32"
defaultTFCpxDataType="complex64"



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def MidPos2D(anImg):
    res=np.shape(anImg)[0:2];
    return np.floor(res)/2

def MidPos3D(anImg):
    res=np.shape(anImg)[0:3];
    return np.ceil(res)/2

# Some helpful MATLAB functions
def abssqr(inputar):
    return np.real(inputar*np.conj(inputar))
    #return tf.abs(inputar)**2

def tf_abssqr(inputar):
    return tf.real(inputar*tf.conj(inputar))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind # array_shape[1]
    return (rows, cols)

def binary_activation(x):

    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

# total variation denoising
def total_variation_regularization(x, beta=1):
    #assert isinstance(x, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: tf.nn.conv2d(x, wh, strides = [1, 1, 1, 1], padding='SAME')
    tvw = lambda x: tf.nn.conv2d(x, ww, strides = [1, 1, 1, 1], padding='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    return tv


def relative_error(orig, rec):
    return np.mean((orig - rec) ** 2)
    #return np.sum(np.square(np.abs(orig-rec))/np.square(np.abs(orig)))

def repmat4d(inputarr, n4dim):
    return np.tile(np.reshape(inputarr, [inputarr.shape[0], inputarr.shape[1], 1, 1]), [1, 1, 1, n4dim])
    


# %% FT

# fftshifts
def fftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1  # from 0 to shape-1
    top, bottom = tf.split(tensor, 2, last_dim)  # split into two along last axis
    tensor = tf.concat([bottom, top], last_dim)  # concatenates along last axis
    left, right = tf.split(tensor, 2, last_dim - 1)  # split into two along second last axis
    tensor = tf.concat([right, left], last_dim - 1)  # concatenate along second last axis
    return tensor

def ifftshift2d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 2 dims of tensor (on both for size-2-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    last_dim = len(tensor.get_shape()) - 1
    left, right = tf.split(tensor, 2, last_dim - 1)
    tensor = tf.concat([right, left], last_dim - 1)
    top, bottom = tf.split(tensor, 2, last_dim)
    tensor = tf.concat([bottom, top], last_dim)
    return tensor

def fftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor

def ifftshift3d(tensor):
    """
    Shifts high frequency elements into the center of the filter.
    Works on last 3 dims of tensor (on all for size-3-tensors)
    Works only for even number of elements along these dims.
    """
    #warn("Only implemented for even number of elements in each axis.")
    left, right = tf.split(tensor, 2, -2)
    tensor = tf.concat([right, left], -2)
    top, bottom = tf.split(tensor, 2, -1)
    tensor = tf.concat([bottom, top], -1)
    front, back = tf.split(tensor, 2, -3)
    tensor = tf.concat([back, front], -3)
    return tensor 

def midPos(tfin):
    """
    helper function to get the mid-point in integer coordinates of tensorflow arrays.

    It calculates the floor of the integer division of the shape vector. This is useful to get the zero coordinate in Fourier space

    Parameters
    ----------
    tfin : tensorflow array to be convolved with the PSF

    Returns
    -------
    vector to mid point

    """
    return tf.floordiv(tf.shape(tfin),2)

def fftshift(tfin):
    """
    shifts the coordinate space before an FFT.

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array

    """
    with tf.name_scope('preFFTShift'):
        return tf.manip.roll(tfin, shift=-midPos(tfin), axis=tf.range(0, tf.size(tf.shape(tfin))))  # makes a copy which is shifted

def ifftshift(tfin):
    """
    shifts the coordinate space after an FFT.

    performs an fftshift operation, cyclicly wrapping around by a shift vector corresponding to the middle.
    The middle will end up at the zero coordinate pixel.

    Parameters
    ----------
    tfin : tensorflow array to be shifted

    Returns
    -------
    shifted tensorflow array

    """
    with tf.name_scope('postFFTShift'):
        return tf.manip.roll(tfin, shift=midPos(tfin), axis=tf.range(0, tf.size(tf.shape(tfin))))  # makes a copy which is shifted



# I would recommend to use this
def my_ft2d(tensor, scaling=1.):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift(tf.fft2d(ifftshift(tensor)))/scaling

def my_ift2d(tensor, scaling=1.):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of ifft unlike dip_image.
    """
    return fftshift(tf.ifft2d(ifftshift(tensor)))*scaling

def my_ft3d(tensor, scaling=1.):
    """
    fftshift(fft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift(tf.fft3d(ifftshift(tensor)))/scaling

def my_ift3d(tensor, scaling=1.):
    """
    fftshift(ifft(ifftshift(a)))
    
    Applies shifts to work with arrays whose "zero" is in the middle 
    instead of the first element.
    
    Uses standard normalization of fft unlike dip_image.
    """
    return fftshift(tf.ifft3d(ifftshift(tensor)))*scaling

def angle(z):
    """
    Returns the elementwise arctan of z, choosing the quadrant correctly.

    Quadrant I: arctan(y/x)
    Qaudrant II: π + arctan(y/x) (phase of x<0, y=0 is π)
    Quadrant III: -π + arctan(y/x)
    Quadrant IV: arctan(y/x)

    Inputs:
        z: tf.complex64 or tf.complex128 tensor
    Retunrs:
        Angle of z
    """
    print('ATTENTION: We use unofficial Angle-fct here!')
    if z.dtype == tf.complex128:
        dtype = tf.float64
    else:
        dtype = tf.float32
    x = tf.real(z)
    y = tf.imag(z)
    xneg = tf.cast(x < 0.0, dtype)
    yneg = tf.cast(y < 0.0, dtype)
    ypos = tf.cast(y >= 0.0, dtype)

    offset = xneg * (ypos - yneg) * np.pi

    return tf.atan(y / x) + offset


###### CHRISTIANS STUFF
    # Copyright Christian Karras
    
def ramp(mysize=(256,256), ramp_dim=0, corner='center'):
    '''
    creates a ramp in the given direction direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
                         size_x = 101 -> goes from -50 to 50
        negative : goes from negative size_x to 0
        positvie : goes from 0 size_x to positive
        int number: is the index where the center is!
    '''
    
    if type(mysize)== np.ndarray:
        mysize = mysize.shape;
    
    res = np.ones(mysize);
   
    if corner == 'negative':
        miniramp = np.arange(-mysize[ramp_dim]+1,1,1);
    elif corner == 'positive':
        miniramp = np.arange(0,mysize[ramp_dim],1);
    elif corner == 'freq':
        miniramp = np.arange(-mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),1)/mysize[ramp_dim];
    elif corner == 'center':
        miniramp = np.arange(-mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),mysize[ramp_dim]//2+np.mod(mysize[ramp_dim],2),1);
    elif (type(corner) == int or type(corner) == float):
        miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
    else:
        try: 
            if np.issubdtype(corner.dtype, np.number):
                miniramp = np.arange(0,mysize[ramp_dim],1)-corner;
        except AttributeError:
           
            pass;
    minisize = list(np.ones(len(mysize)).astype(int));
    minisize[ramp_dim] = mysize[ramp_dim];
    #np.seterr(divide ='ignore');
    miniramp = np.reshape(miniramp,minisize)
    res*=miniramp;
    return(res);


def xx(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in x direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if x is even, it has one more value in the positve:
                    e.g. size_x = 100 -> goes from -50 to 49
        negative : goes from negative size_x to 0
        positvie : goes from 0 size_x to positive
    '''
    return(ramp(mysize,1,mode))

def yy(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in y direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,0,mode))
 
def zz(mysize = (256,256), mode = 'center'):
    '''
    creates a ramp in z direction 
    standart size is 256 X 256
    mode: 
        center: 0 is at cetner
                if y is even, it has one more value in the positve:
                    e.g. size_y = 100 -> goes from -50 to 49
        negative : goes from negative size_y to 0
        positvie : goes from 0 size_y to positive
    '''
    return(ramp(mysize,2,mode))

def rr(mysize=(256,256), offset = (0,0,0), scale = None, mode='center'):
    '''
    creates a ramp in r direction 
    standart size is 256 X 256
    mode is always "center"
    offset -> x/y offset in pixels (number, list or tuple)
    scale is tuple, list, none or number of axis scale
    '''
    import numbers;
    if offset is None:
        scale = [0,0,0];
    elif isinstance(offset, numbers.Number):
        offset = [offset];
    elif type(offset)  == list or type(offset) == tuple:
        offset = list(offset[0:3]);
    else:
        raise TypeError('Wrong data type for offset -> must be Number, list, tuple or none')
        
    if scale is None:
        scale = [1,1,1];
    elif isinstance(scale, numbers.Integral):
        scale = [scale, scale];
    elif type(scale)  == list or type(scale) == tuple:
        scale = list(scale[0:3]);
    else:
        raise TypeError('Wrong data type for scale -> must be integer, list, tuple or none')

    if(len(mysize)==2):
        return(np.sqrt(((ramp(mysize,0, mode)-offset[0])*scale[0])**2
                       +((ramp(mysize,1, mode)-offset[1])*scale[1])**2))      
    elif(len(mysize)==3):
        return(np.sqrt(((ramp(mysize,0, mode)-offset[0])*scale[0])**2
                       +((ramp(mysize,1, mode)-offset[1])*scale[1])**2
                       +((ramp(mysize,2, mode)-offset[2])*scale[2])**2))
   
def phiphi(mysize=(256,256), offset = 0, angle_range = 1):
    '''
    creates a ramp in phi direction 
    standart size is 256 X 256
    mode is always center
    offset: angle offset in rad
    angle_range:
            1:   0 - pi for positive y, 0 - -pi for negative y
            2:   0 - 2pi for around
        
    '''
    np.seterr(divide ='ignore', invalid = 'ignore');
    x = ramp(mysize,0,'center');
    y = ramp(mysize,1,'center');
    #phi = np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1);
    if angle_range == 1:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset)+np.pi, 2*np.pi) -np.pi;
    elif angle_range == 2:
        phi = np.mod((np.arctan(y/x)+(x<0)*np.pi*((y>0)*2-1)+offset), 2*np.pi);
    phi[phi.shape[0]//2,phi.shape[1]//2]=0;
    np.seterr(divide='warn', invalid = 'warn');
    return(phi)    


def totensor(img):
    if istensor(img):
        return img
    if (not isinstance(0.0,numbers.Number)) and ((img.dtype==defaultTFDataType) or (img.dtype==defaultTFCpxDataType)):
        img=tf.constant(img)
    else:
        if iscomplex(img):
            img=tf.constant(img,defaultTFCpxDataType)
        else:
            img=tf.constant(img,defaultTFDataType)
    return img

def iscomplex(mytype):
    mytype=str(datatype(mytype))
    return (mytype == "complex64") or (mytype == "complex128") or (mytype == "complex64_ref") or (mytype == "complex128_ref") or (mytype=="<dtype: 'complex64'>") or (mytype=="<dtype: 'complex128'>")

def datatype(tfin):
    if istensor(tfin):
        return tfin.dtype
    else:
        if isinstance(tfin,np.ndarray):
            return tfin.dtype.name
        return tfin # assuming this is already the type

def istensor(tfin):
    return isinstance(tfin,tf.Tensor)

def shapevec(tfin):
    """
        returns the shape of a tensor as a numpy ndarray
    """
    if istensor(tfin):
        return np.array(tfin.shape.as_list())
    else:
        return np.array(tfin.shape)

def expanddimvec(shape,ndims,othersizes=None,trailing=False):
    """
        expands an nd image shape tuple to the necessary number of dimension by inserting leading dimensions
        ----------
        img: input image to expand
        ndims: number of dimensions to expand to
        trailing (default:False) : append trailing dimensions rather than dimensions at the front of the size vector
        othersizes (defatul:None) : do not expand with ones, but rather use the provided sizes
    """
    if isinstance(shape,numbers.Number):
        shape=(shape,)
    else:
        shape=tuple(shape)
    missingdims=ndims-len(shape)
    if missingdims > 0:
        if othersizes is None:
            if trailing:
                return shape+(missingdims)*(1,)
            else:
                return (missingdims)*(1,)+shape
        else:
            if trailing:
                return shape+tuple(othersizes[-missingdims::])
            else:
                return tuple(othersizes[0:missingdims])+shape
    else:
        return shape[-ndims:]


def extract(tfin, newsize, mycenter=None, constant_values=0):
    """
    extracts (and pads) a subregion of a tf tensor.

    Parameters
    ----------
    tfin : tensorflow array to pad
    newsize: size of the array after extraction
    mycenter (default=None): The center of the ROI to extract. If None is supplied, the center (Fourier-convention) of the old array is assumed

    Returns
    -------
    resulting tensorflow array

    """
    tfin = totensor(tfin)
    oldsize = shapevec(tfin)
    newsize = np.array(expanddimvec(newsize, len(oldsize), othersizes=oldsize), dtype="int32")

    if mycenter is None:
        mycenter = oldsize // 2
    else:
        mycenter = np.array(mycenter)
    spos = mycenter - newsize // 2  # start position in old array
    epos = spos + newsize  # end position in old array
    ExtractSPos = np.maximum(spos, 0)
    ExtractSize = np.maximum(np.minimum(epos - ExtractSPos, oldsize - ExtractSPos), 0)
    if any(ExtractSize <= 0):
        raise ValueError("Trying to extract an empty region")
    #    print("ExtractSPos",ExtractSPos)
    #    print("ExtractSize",ExtractSize)
    Extracted = tf.slice(tfin, ExtractSPos, ExtractSize)
    padbefore = np.maximum(-spos, 0)
    padafter = np.maximum(newsize - ExtractSize - padbefore, 0)
    #    print(Extracted)
    #    print(padbefore)
    #    print(padafter)
    return pad(Extracted, padbefore, padafter, constant_values=constant_values)

def pad(tfin, padbefore, padafter, constant_values=0):
    """
    pads values outside the array

    Allows to account for borders.

    Parameters
    ----------
    tfin : tensorflow array to pad
    padbefore : vector of pad sized before the array
    padafter : vector of pad sized after the array
    constant_values : value to pad in (default:0)

    Returns
    -------
    resulting tensorflow array

    """
    with tf.name_scope('Pad'):
        paddings=np.stack((padbefore,padafter),1)
#        print(paddings)
        return tf.pad(tfin, paddings,constant_values=constant_values)  # [0, 2]

    
    
def DampEdge(im, width = None, rwidth=0.1, axes =None, func = None, method="damp", sigma=4.0):
    '''
        Dampedge function 
        
        im  image to damp edges 
        
        rwidth : relative width (default : 0.1 meaning 10%)
            width in relation to the image size along this dimenions. Can be a single number or a tuple
            
        width (None: rwidht is used, else width takes precedence)
            -> characteristic absolute width for damping
            -> can be integer, than every (given) axis is damped by the same size
            -> can be list or tupel -> than individual damping for given axis
            
        axes-> which axes to be damped (default is (0,1))

        func   - which function shall be used for damping -> some are stated in functions.py, first element should be x, second one the length (Damping length!)
                e.g. cossqr, coshalf, linear
                default: coshalf
        
        method -> which method should be used?
                -> "zero" : dims down to zero
                -> "damp" : blurs to an averaged mean (default)
                -> "moisan" : HF/LF split method according to Moisan, J Math Imaging Vis (2011) 39: 161–179, DOI 10.1007/s10851-010-0227-1
    
        return image with damped edges
        
        TODO in FUTURE: padding of the image before damping
    '''
    res = np.ones(im.shape);    
    if width==None:
        width=tuple(np.round(np.array(im.shape)*np.array(rwidth)).astype("int"))
        
    if axes==None:
        axes=np.arange(0,im.ndim).tolist()
    if type(width) == int:
        width = [width];
    if type(width) == tuple:
        width = list(width);
    if len(width) < len(axes):
        ext = np.ones(len(axes)-len(width))*width[-1];
        width.extend(list(ext.astype(int)));
        
    res=im
    mysum=im*0.0
    sz=im.shape;
    den=-2*len(set(axes)); # use only the counting dimensions
    for i in range(len(im.shape)):
        if i in axes:
            line = np.arange(0,im.shape[i],1);
            ramp = make_damp_ramp(width[i],func);            
            if method=="zero":
                line = cat((ramp[::-1],np.ones(im.shape[i]-2*width[i]),ramp),0);
                goal=0.0 # dim down to zero
            elif method=="moisan":
#                for d=1:ndims(img)
                top=subslice(im,i,0)
                bottom=subslice(im,i,-1)
                mysum=subsliceAsg(mysum,i,0,bottom-top + subslice(mysum,i,0));
                mysum=subsliceAsg(mysum,i,sz[i]-1,top-bottom + subslice(mysum,i,sz[i]-1));
                den=den+2*np.cos(2*np.pi*ramp(dimVec(i,sz[i],len(sz)),i,freq='freq'))
            elif method=="damp":
                line = cat((ramp[::-1],np.ones(im.shape[i]-2*width[i]+1),ramp[:-1]),0);  # to make it perfectly cyclic
                top=subslice(im,i,0)
                bottom=subslice(im,i,-1)
                goal = (top+bottom)/2.0
                kernel=gaussian(goal.shape,sigma)
                goal = convolve(goal,kernel,norm2nd=True)
            else:
                raise ValueError("DampEdge: Unknown method. Choose: damp, moisan or zero.")
            #res = res.swapaxes(0,i); # The broadcasting works only for Python versions >3.5
#            res = res.swapaxes(len(im.shape)-1,i); # The broadcasting works only for Python versions >3.5
    if method=="moisan":
        den=nip.MidValAsg(den,1);  # to avoid the division by zero error
        den=nip.ft(mysum)/den;
        den=nip.MidValAsg(den,0);  # kill the zero frequency
        den=nip.ift(den)
        res=im-den
    else:
        line = expanddim(line,im.ndim).swapaxes(0,i); # The broadcasting works only for Python versions >3.5
        try:
            res = res*line + (1.0-line)*goal
        except ValueError:
            print('Broadcasting failed! Maybe the Python version is too old ... - Now we have to use repmat and reshape :(')
            from numpy.matlib import repmat;
            res *= np.reshape(repmat(line, 1, np.prod(res.shape[1:])),res.shape, order = 'F');
        
    #return(res)
    return(res.view(image));


def make_damp_ramp(length, function):
    '''
        creates a damp ramp:
            length - length of the damp ramp in pixes
            function - function of the damp ramp
                        Generally implemented in nip.functions
                        Make sure that first element is x and second element is characteristica lengths of the function
    '''
    x = np.arange(0, length,1);
    return(function(x, length-1));
    
    

def cat(self, imlist, ax):
    if get_type(imlist) == 'list':
        im =cat(imlist+[self],ax=ax);
    elif get_type(imlist)[0] == 'array':
        im =cat([imlist,self],ax= ax);
        
    else:
        raise TypeError('Imlist is wrong data type')
    im = im.view(image);
    im.__array_finalize__(self);
    return(im);

def subslice(img,mydim,start):
    '''
        extracts an N-1 dimensional subslice at dimension dim and position start        
        It keeps empty slices as singleton dimensions
    '''
    if start!=-1:
        end=start+1
    else:
        end=None
    coords=(mydim)*[slice(None)]+[slice(start,end)]+(img.ndim-mydim-1)*[slice(None)]
    return img[tuple(coords)]

def dimVec(d,mysize,ndims):
    '''
        creates a vector of ndims dimensions with all entries equalt to one except the one at d which is mysize
        ----------
        d: dimension to specify entry for
        mysize : entry for res[d]        
        ndims: length of the result vector
    '''
    res=ndims*[1]
    res[d]=mysize
    return tuple(res)

def subsliceAsg(img,mydim,start,val):
    '''
        assigns val to an N-1 dimensional subslice at dimension dim and position start
        -----
        img : input image to assign to
        mydim : dimension into which the subslice is chosen
        start : offset along this dimension
        val : value(s) to assign into the array
        It keeps empty slices as singleton dimensions
    '''
    if start!=-1:
        end=start+1
    else:
        end=None
    coords=(mydim)*[slice(None)]+[slice(start,end)]+(img.ndim-mydim-1)*[slice(None)]
    img[tuple(coords)]=val
    return img

def gaussian(myshape,sigma):
    '''
        n-dimensional gaussian function
    '''
    if isinstance(sigma,numbers.Number):
        sigma=len(myshape)*[sigma]
    return np.exp(-rr2(myshape,scale=tuple(1/(np.sqrt(2)*np.array(sigma)))))


def rr2(mysize=(256,256), offset = (0,0),scale = None, freq=None):
    '''
    creates a square of a ramp in r direction 
    standart size is 256 X 256
    mode is always "center"
    offset -> x/y offset in pixels (number, list or tuple)
    scale is tuple, list, none or number of axis scale
    '''
    import numbers;
    if offset is None:
        scale = [0,0];
    elif isinstance(offset, numbers.Number):
        offset = [offset, 0];
    elif type(offset)  == list or type(offset) == tuple:
        offset = list(offset[0:2]);
    else:
        raise TypeError('Wrong data type for offset -> must be Number, list, tuple or none')
        
    if scale is None:
        scale = [1,1];
    elif isinstance(scale, numbers.Integral):
        scale = [scale, scale];
    elif type(scale)  == list or type(scale) == tuple:
        scale = list(scale[0:2]);
    else:
        raise TypeError('Wrong data type for scale -> must be integer, list, tuple or none')
    res=((ramp(mysize,0,'center',freq)-offset[0])*scale[0])**2
    for d in range(1,len(mysize)):
        res+=((ramp(mysize,d,'center',freq)-offset[d])*scale[d])**2
    return res

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
def insertPerfectAbsorber(mysample,SlabX,SlabW=1,direction=None,k0=None,N=4):
    '''
    % mysample=insertPerfectAbsorber(mysample,SlabX,SlabW) : inserts a slab of refractive index material into a dataset
    % mysample : dataset to insert into
    % SlabX : middle coordinate
    % SlabW : half the width of slab
    % direction : direction of absorber: 1= left to right, -1=right to left, 2:top to bottom, -2: bottom to top
    '''

    if k0==None:
        k0=0.25/np.max(np.real(mysample));

    k02 = k0**2;
    
    if mysample.ndim < 3:
        mysample = np.expand_dims(mysample, 2)
        
        
    myXX=xx((mysample.shape[0], mysample.shape[1], mysample.shape[2]))+mysample.shape[0]/2
    
    if np.abs(direction) <= 1:
        myXX=xx((mysample.shape[0], mysample.shape[1], mysample.shape[2]))+mysample.shape[0]/2
    elif np.abs(direction) <= 2:
        myXX=yy((mysample.shape[0], mysample.shape[1], mysample.shape[2]))+mysample.shape[1]/2
    elif np.abs(direction) <= 3:
        myXX=zz((mysample.shape[0], mysample.shape[1], mysample.shape[2]))+mysample.shape[2]/2


    alpha=0.035*100/SlabW #; % 100 slices
    if direction > 0:
        #% mysample(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp(xx(mysize,'corner')/mysize(1));  % increasing absorbtion
        myX=myXX-SlabX
    else:
        # %mysample(abs(myXX-SlabX)<=SlabW)=1.0+1i*alpha*exp((mysize(1)-1-xx(mysize,'corner'))/mysize(1));  % increasing absorbtion
        myX=SlabX+SlabW-myXX-1
        
        
    myMask= (myX>=0) * (myX<SlabW)

    alphaX=alpha*myX[myMask]
    
    PN=0;
    for n in range(0, N+1):
        PN = PN + np.power(alphaX,n)/factorial(n)

    k2mk02 = np.power(alphaX,N-1)*abssqr(alpha)*(N-alphaX + 2*1j*k0*myX[myMask])/(PN*factorial(N))
    
 
    mysample[myMask] = np.sqrt((k2mk02+k02)/k02)
            
    #np.array(mysample)[0]
    
    return mysample,k0
