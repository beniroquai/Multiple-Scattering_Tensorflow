# MuSCAT

This is an attempt to invert multiple scattering based on Beam Propagation Method (BPM) implement in Tensorflow. 
We're refering to this [publication.](http://www.focusonmicroscopy.org/2018/PDF/1094_Diederich.pdf)


## Examples
### 3D Sphere immersed in Water
This is the sliced 3D object :

![3D GT](./images/sphere_gt.gif)

This is the output of the partially coherent fwd model (Q-PHASE):

![3D Meas](./images/sphere_meas.gif)

After minimizing the L2-norm and applying the TV-Regularizer with lambda=1e-3 after 1000 iterations:

![3D Meas](./images/sphere_rec.gif)

### Simulated cheek-cell 'imaged' using single and multiple scattering

This one is the result after a convolution of the `ASF` with the scattering potential `V`:
![Cheek_abs_born](./images/Cheek_abs_born.png)

This one is the result after full BPM simulation of the partially-coherent image-formation using the same sample:
![Cheek_abs_born](./images/Cheek_abs_bpm.png)