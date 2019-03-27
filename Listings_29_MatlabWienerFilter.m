% This is the MATLAB reference implementation for the Wiener Filtering 
load('./ExperimentAsfObj.mat')

myasf = dip_image(asf);
myatf = ft(myasf);
myobj = dip_image(obj);
myalpha = .1;

myres = ift((conj(myatf)*ft(myobj))/(abssqr(myatf)+myalpha))
