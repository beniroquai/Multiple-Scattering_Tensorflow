% This file should create a set of experimental data of droplets
ampfolder = 'K:\Droplets\S19\Amplitude\';
anglefolder = 'K:\Droplets\S19\PHase\';
anglefiles = dir(fullfile(anglefolder, '*.tiff'));
ampfiles = dir(fullfile(ampfolder, '*.tiff'));


%% This is the file holding the parameters for the EM experiment provided by
% the Droplets
myParameter = struct();

% necessary for the script to work
myParameter.nEmbb = 1.33; %
myParameter.lambda0 = .650; % nm
myParameter.lambdaM = myParameter.lambda0*myParameter.nEmbb; % nm
myParameter.NAc = 0.52; % outer radius of the condensers NA
myParameter.NAo = .95;
myParameter.FOV = 94.09;
myParameter.Nx_raw = 600;
myParameter.Ny_raw = 600;
myParameter.Nz_raw = 701;
myParameter.dx = myParameter.FOV/myParameter.Nx_raw; % nm
myParameter.dy = myParameter.dx; % nm
myParameter.dz_raw = .020; % nm

myParameter.shiftIcX=-1;
myParameter.shiftIcY=-1

%% adjust number of z-samples
subsampling_z = 10;
myParameter.dz = subsampling_z*myParameter.dz_raw; % nm
myParameter.Nz = int16(floor(myParameter.Nz_raw/subsampling_z));

% adjust number of x/y pixels
myParameter.Nx = 32;
myParameter.Ny = 32;

% Display the names
nfiles = size(ampfiles,1);
myamp = readim([ampfolder ampfiles(round(nfiles/2)).name]);
myangle =  readim([anglefolder anglefiles(round(nfiles/2)).name]);
%
figure 
subplot(121)
imagesc(double(myamp))
colorbar
axis image, colormap gray
title 'Amplitude of Droplets'
subplot(122)
imagesc(double(myangle))
colorbar
axis image, colormap gray
title 'Angle of Droplets'

AH=dipshow(myamp) % find edges of CC signal
diptruesize(AH,150)
%fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
fprintf('Get the ROI coordinates')
fprintf('Select center of droplets.');
nroi = 30;
roi_coordinates = zeros(2,nroi);
for i =1:nroi
    roi_coordinates(:,i) = dipgetcoords(AH,1);
    fprintf([num2str(nroi-i) ' selections left \n']);
end
myParameter.roi_coordinates = roi_coordinates;

% Save everything to disk
save('Parameter', 'myParameter', '-v7.3')

%% here we extract the subrois
myroistack = dip_image(1i*ones(myParameter.Nx,myParameter.Ny,myParameter.Nz,nroi));
iz = 0;
for ifile = floor(linspace(1,nfiles,myParameter.Nz))
    myamp = readim([ampfolder ampfiles(ifile).name]);
    myphase = readim([anglefolder anglefiles(ifile).name]);
    for iroi = 1:nroi
        amptmp = extract(myamp, [myParameter.Nx myParameter.Ny], [ roi_coordinates(1,iroi)  roi_coordinates(2,iroi)]);
        angletmp = extract(myphase, [myParameter.Nx myParameter.Ny], [ roi_coordinates(1,iroi)  roi_coordinates(2,iroi)]);
        myroistack(:,:,iz,iroi-1) = amptmp*exp(1i*angletmp);
    end
    iz = iz+1;
    fprintf(['-' num2str(ifile)])
end

%% Clean for stripes along Z and save to disk
figure
for iroi = 1:nroi
    myroi = squeeze(myroistack(:,:,:,iroi-1));
    mystripes = mean(myroi,[],[1,2]);
    allAmp_red = double(myroi/mystripes);
    %save(['S19_subroi' num2str(iroi+20)], 'allAmp_red', '-v7.3')
    
    subplot(4,nroi/2,iroi)
    imagesc(transpose(squeeze(abs(allAmp_red(:,16,:)))))
    axis image, colormap gray
    title(['ABS#' num2str(iroi)])
    subplot(2*2,nroi/2,iroi+nroi)
    imagesc(transpose(squeeze(angle(allAmp_red(:,16,:)))))
    axis image, colormap gray
    title(['Ang#' num2str(iroi)])
end


