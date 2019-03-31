% This is the MATLAB reference implementation for the Wiener Filtering 
load('./ExperimentAsfObj.mat')

myasf = (asf);
myatf = ft(myasf);
myobj = (obj);
myalpha = 1e-1;

myres = fftshift(dip_image(ifftn((conj(fftn(myasf)).*fftn(myobj))./(abssqr(fftn(myasf))+myalpha))))
