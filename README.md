# MuSCAT

This is an attempt to invert multiple scattering based on Beam Propagation Method (BPM) implement in Tensorflow.
We're refering to this [publication.](http://www.focusonmicroscopy.org/2018/PDF/1094_Diederich.pdf)

In general this treats the following imaging geometry:

<p align="center">
<img src="./images/muscat_setup.png" height="400">
<href = "UC2_WORKSHOP_Lightsheet_Microscope_v0_english.pdf">
</p>



## Quick-Start
We provide a iPython notebook file to test the "Partially Coherent Imager" based on single and multiple scattering. Therefore, the following steps need to be done after installing all dependencies listed below.

- Download this repository by entering (e.g. in the Terminal):
``git clone https://github.com/beniroquai/Multiple-Scattering_Tensorflow``
- Start the iPython notebook by typing ```ipython notebook```
- Select the file ```Listings_0_FWD_Born_BPM.ipynb``` [here](Listings_0_FWD_Born_BPM.ipynb)
- Start the script by running it inside the Webbrowser
- Get familar and post a bug (very likely)


## Installation of all dependencies

- Install Anacoda3 with Python version >3.6.
- Install Tensorflow-GPU following this [guide](https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/).

The following packages are needed in order to make the toolbox work.


```
pip install tifffile
pip install matplotlib
pip install pyyaml
pip install scipy
pip install scikit
pip install git+https://NanoImagingUser:NanoImagingUser@gitlab.com/bionanoimaging/nanoimagingpack
pip install pyaml
pip install scikit-image
```

## Data
Data can be provided upon request.

## Examples

### Simulated cheek-cell 'imaged' using single and multiple scattering

This one is the result after a convolution of the `ASF` with the scattering potential `V`:
![Cheek_abs_born](./images/Cheek_abs_born.png)

This one is the result after full BPM simulation of the partially-coherent image-formation using the same sample:
![Cheek_abs_born](./images/Cheek_abs_bpm.png)

# Theory

## Getting started

### Simulation only

To simulate a 3D measuremnt (Born/BPM) of the QPhase you can start using the following file:

***Listings_10_fwd_born_ms_simple.py***

All parameters of the microscope are in [here](./src/Simulations.py). (**import src.simulations as experiments)**

Major parameters (all parameters are in µm!):

```
dx, dy, dz = the sampling in spatial dimension (e.g. pixelsize/z-stepsize)
NAo = NA of objective lens
NAc = NA of illumination lens/condensor
NAci = NA of illumination lens/condensor (if annular, inner diameter)
shiftIcX/shiftIcY = shift in X/Y of the effective illuminatoin aperture in normalized pupil coordinates
mysize = number of voxels in XYZ - order: ZXY
is_mictype = type of microscope you want to use (DIC, BF, DF, PC) - > computes the proper amplitudes for the pupil planes to simulate the imaging effect of those methods

```


Modes for propagating the results:

* 'Born'
* 'BPM'
* '3DQDPC' - follows this code: https://github.com/Waller-Lab/3DQuantitativeDPC/tree/master/python_code


Not really important - don't touch it
```
''' Define parameters '''
is_padding = False # better don't do it, some normalization is probably incorrect
is_display = True
is_optimization = False
is_optimization_psf = False
is_flip = False
is_measurement = False
```

The following creates a model:
```
''' Create the Model'''
muscat = mus.MuScatModel(myparams, is_optimization=is_optimization)
```

It creates spatial frequency grid for the normalized pupil function for example. Parameter-handling is one here - no computation done here! Basically it creates the class-model of the microscope

Next you create an object (you can also side-load it using a matlab-MAT file). Basically provide a 3D numpy array with dimensions depicted in ```mysize```. Assign proper values for RI for real and imaginary part with
```obj = obj_real + 1j*obj_absorption```.
#### Initializing Tensorflow

print('Start the TF-session')
sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
```


mysubsamplingIC - if larger than zero (integer values) ,  the illumination source gets sub-sampled, meaning that not every point in the effective pupil plane gets propagated through the system.

Shifting the pupil function in frequency space by applying a phaseramp in normalized coordinates
nip.view(nip.ift(nip.ft(nip.rr((100,100))<25) * np.exp(1j * 2 * np.pi * 50 * nip.xx((100,100), freq='ftfreq'))))


## MATLAB-Side

### 1st create a struct called myParamter with experimentation parameters like so:

```
myParameter = struct()
myParamter.dx = .1;
myParamter.dz = .1;
myParamter.dy = .1;
...
```

### 2nd take your measuremnt stack (XYZ-data), cast it to double and call it allAmpsimu like so:
```
allAmpSimu = double((allAmp_clean_extract));
```

### 3rd create a binary reference object which localizes the refractive index in XYZ and cast it to ```
double too; name it mysphere_mat:
mysphere_mat = double(mysphere);
```


### 4th finally save it as hdf5-files:

```
allAmpSimu = double((allAmp_clean_extract));
myfolder_save = './data/';

save([myfolder_save, myFile,'myParameter.mat'], 'myParameter','-v7.3')
save([myfolder_save, myFile, '_allAmp.mat'], 'allAmpSimu','-v7.3')
save([myfolder_save, myFile, '_mysphere.mat'], 'mysphere_mat','-v7.3')
%
%allAmpSimu_ang = angle(allAmpSimu);
%([myfolder_save, myFile, '_allAmp_angle.mat'], 'allAmpSimu_ang','-v7.3')
[myfolder_save, myFile,'myParameter.mat']
[myfolder_save, myFile, '_allAmp.mat']
[myfolder_save, myFile, '_mysphere.mat']
```


## PYTHON-Side

Create a new case in the file './src/experiments.py' like the one below, where you enter the filenames for matlab_par_filename, matlab_obj_file, obj_meas_filename

```
if(1):

    '''Droplets recent from Dresden! '''

    ''' Hologram Reconstruction Parameters '''
    myFolder = 'W:\\diederichbenedict\\Q-PHASE\\PATRICK\\';
    myFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif';
    myDarkFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_dark.tif';
    myBrightfieldFile = 'S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment_bright.tif';
    roi_center = (229,295)
    cc_center = np.array((637, 1418))
    cc_size = np.array((600, 600))

    ''' Reconstruction Parameters '''
    # data files for parameters and measuremets
    obj_meas_filename = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_allAmp.mat'
    matlab_par_filename = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tifmyParameter.mat'
    matlab_obj_file = './Data/cells/S0105k_zstack_dz0-2um_25C_40x_Ap0-52_Int63_sameAlignment.tif_mysphere.mat'
    matlab_par_name = 'myParameter'
    matlab_val_name = 'allAmpSimu'   
    mybackgroundval = 0.
    dn = 0.1
    NAc = .3

    is_dampic= 0.1#.051 # smaller => more damping!
    mysubsamplingIC = 0


    ''' Control-Parameters - Optimization '''
    #my_learningrate = 1e3  # learning rate



    zernikefactors = np.zeros((11,))

    #zernikemask=1.*(np.abs(zernikefactors)>0)
    zernikemask = np.ones(zernikemask.shape)
    zernikemask[0:4] = 0 # don't care about defocus, tip, tilt

    # worked with tv= 1e1, 1e-12, lr: 1e-2 - also in the git!
    zernikefactors = np.array((3.4140131e-01, -3.2123593e-03,  9.3390346e-03, -2.5132412e-01, -1.7191889e-04, -3.0539016e-04, -1.1636049e-02,  3.3056294e-03, -1.2694970e-05, -7.7548197e-05,  1.2201133e-01))

    shiftIcX = 0.01009951
    shiftIcY =-0.009845202




    #zernikefactors = 0*np.array((0.,0.,0.,0.,0.49924508,-1.162684,-0.09952152,-0.4380897,-0.6640752,0.16908956,0.860051))

    #for BORN
    if(1):
        regularizer = 'TV'
        lambda_reg = 1e-0
        my_learningrate = 1e0  # learning rate
        lambda_zernike = 1.
        lambda_icshift = 1.
        lambda_neg = 0*100.
        myepstvval = 1e-11 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky
        dz = .3
        #zernikefactors *= 0
        NAci = 0
        NAc = .4
        is_mictype = 'BF'
    else:
        ''' Control-Parameters - Optimization '''
        is_mictype = 'DF'

        regularizer = 'TV'
        lambda_reg = 1e-3
        myepstvval = 1e-11 ##, 1e-12, 1e-8, 1e-6)) # - 1e-1 # smaller == more blocky

        NAci = .1
        NAc = .2
        my_learningrate = 1e1  # learning rate
        mysubsamplingIC = 0
        dz = .3
```

## License
This software is licensed under the GNU General Public License v3.0.

## Disclaimer
The software is full of errors. Please file an issue if you find any suspicious piece of code. We are not responsible for any error

## Acknoledgements
Thanks to R. Heintzmann, L. Tian, P.
