% This file should create a set of experimental data of droplets
ampfolder = 'K:\Droplets\S14a_TIF\Amplitude\';
anglefolder = 'K:\Droplets\S14a_TIF\PHase\';
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
myParameter.NAo = .5;
myParameter.FOV = 188.185;
myParameter.Nx_raw = 600;
myParameter.Ny_raw = 600;
myParameter.Nz_raw = 752;
myParameter.dx = myParameter.FOV/myParameter.Nx_raw; % nm
myParameter.dy = myParameter.dx; % nm
myParameter.dz_raw = .040; % nm

myParameter.shiftIcX=-1;
myParameter.shiftIcY=-1

%% adjust number of z-samples
subsampling_z = 5;
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
nroi = 50;
if(0)
roi_coordinates = zeros(2,nroi);
for i =1:nroi
    roi_coordinates(:,i) = dipgetcoords(AH,1);
    fprintf([num2str(nroi-i) ' selections left \n']);
end
myParameter.roi_coordinates = roi_coordinates;
end
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
nroi = 50
for iroi = 1:nroi
    if(1)
        myroi = squeeze(myroistack(:,:,:,iroi-1));
        mystripes = mean(myroi,[],[1,2]);
        allAmp_red = double(myroi/mystripes);
        save(['S14a_subroi' num2str(iroi)], 'allAmp_red', '-v7.3')
        if(1)
            subplot(4,nroi/2,iroi)
            imagesc(transpose(squeeze(abs(allAmp_red(:,16,:)))))
            axis image, colormap gray
            title(['ABS#' num2str(iroi)])
            subplot(2*2,nroi/2,iroi+nroi)
            imagesc(transpose(squeeze(angle(allAmp_red(:,16,:)))))
            axis image, colormap gray
            title(['Ang#' num2str(iroi)])
        end
        % now annotate the data
    else
        load(['S14a_subroi' num2str(iroi)], 'allAmp_red')
        proj_z = sum(dip_image(allAmp_red), [], 3);
        proj_x = sum(dip_image(allAmp_red), [], 1);
        proj_y = sum(dip_image(allAmp_red), [], 2);
        
        % Get position and size of the sphere
        fprintf('Select boundaries of the "sphere" (upper, right, lower, left)')
        proj_z_ang = double(angle(proj_z));
        proj_z_ang = proj_z_ang - min(proj_z_ang(:));
        proj_z_ang = proj_z_ang./max(proj_z_ang(:));
        proj_z_ang = double(1.*threshold(proj_z_ang).*angle(proj_z_ang));
        %proj_z_ang = proj_z_ang>.2;
        try
            
            fprintf('Select center of the "sphere"')
            AH=dipshow(angle(proj_x)^2) % find edges of CC signal
            diptruesize(AH,1200)
            sphere_coordinates_xz = dipgetcoords(AH,1);
            fprintf('Select center of the "sphere"')
            AH=dipshow(angle(proj_y)^2) % find edges of CC signal
            diptruesize(AH,1200)
            sphere_coordinates_yz = dipgetcoords(AH,1);
            posz = round(0.5*(sphere_coordinates_xz(2)+sphere_coordinates_yz(2)));
            
            
            %imagesc(proj_z_ang), colormap gray, axis image
            proj_z_ang = (angle(allAmp_red(:,:,posz)));
            %proj_z_ang = double(1.*threshold(proj_z_ang ).*angle(proj_z_ang ));
            [centers,radius] = imfindcircles(proj_z_ang,[5 50]);
            posx = round(centers(1));
            posy = round(centers(2));
            
            
            mymidpos = MidPos(allAmp_red);
            mysphere = double(circshift(rr(size(allAmp_red))<radius, -round([mymidpos(1)-posx mymidpos(2)-posy mymidpos(3)-posz])));
            
            %%
            %close all
            if(0)
                %%
                proj_z = sum(dip_image(allAmp_red), [], 3);
                proj_x = sum(dip_image(allAmp_red), [], 1);
                proj_y = sum(dip_image(allAmp_red), [], 2);

                figure(1000)
                subplot(231)
                imagesc(squeeze(double(angle(proj_z)))), axis image, colormap gray, title 'Proj Z'
                hold on
                %h = viscircles(centers,radius);
                hold off
                subplot(234)
                imagesc(double(sum(1*dip_image(mysphere),[],3))), axis image, colormap gray, title 'Proj Z'
                subplot(232)
                imagesc(squeeze(double(angle(proj_x)))), axis image, colormap gray, title 'Proj X'
                subplot(235)
                imagesc(squeeze(double(sum(1*dip_image(mysphere),[],1)))), axis image, colormap gray, title 'Proj X'
                subplot(233)
                imagesc(squeeze(double(angle(proj_y)))), axis image, colormap gray, title 'Proj Y'
                subplot(236)
                imagesc(squeeze(double(sum(1*dip_image(mysphere),[],2)))), axis image, colormap gray, title 'Proj Y'
            end
            save(['S14a_sphere' num2str(iroi)], 'mysphere', '-v7.3')
            
        catch
            disp('No sphere was found!')
        end
        
    end
    
end