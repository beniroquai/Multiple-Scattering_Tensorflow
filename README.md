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

## License
This software is licensed under the GNU General Public License v3.0. 

## Disclaimer
The software is full of errors. Please file an issue if you find any suspicious piece of code. We are not responsible for any error

## Acknoledgements
Thanks to R. Heintzmann, L. Tian, P. 