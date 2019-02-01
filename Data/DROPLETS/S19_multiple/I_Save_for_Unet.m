% This file simply converts the .mat files for the Unet 
myfolder = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/BOSTON/MUSCAT/PYTHON/muScat/Data/DROPLETS/S19_multiple/';
objfolder = 'NN_GT/';
measfolder = 'NN_MEAS/';
objfiles = dir(fullfile([myfolder objfolder], '*.mat'));
measfiles = dir(fullfile([myfolder measfolder], '*.mat'));

savefolder = './qphase_unet/'

for i = 1:size(objfiles,1)
    myfolder = [savefolder 'image' num2str(i)]
    mkdir(myfolder)
    load([objfiles(i).folder '/' objfiles(i).name]);
    load([measfiles(i).folder '/' measfiles(i).name]);
    f_gt = mysphere;
    BPM = cat(4, real(allAmp_red), imag(allAmp_red));
    
    save([myfolder '/f_gt.mat'], 'f_gt')
    save([myfolder '/BPM.mat'], 'BPM')
end


