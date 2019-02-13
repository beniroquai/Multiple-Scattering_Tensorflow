% This file should generate a shepp-logan phantom with less
% complicated/dense structrues
mysize3D = [50 50 50];
phantom = permute(double(abs(phantom3D(mysize3D(2)))),[2,3,1]);
phantom(phantom==1) = .15 ; % outer shell
phantom(phantom>.25 & phantom<.35) = .3;
phantom(phantom>.0 & phantom<.1) = 0; % inner wobbles 
phantom(phantom>.18 & phantom<.22) = .01; % inner mass 
save(['phantom_' num2str(size(phantom,1)) '_' num2str(size(phantom,2)) '_' num2str(size(phantom,3)) '.mat'], 'phantom', '-v7.3');
