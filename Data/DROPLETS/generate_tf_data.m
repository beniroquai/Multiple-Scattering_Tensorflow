% load the parameters
load('S0014a_zstack_dz0-04um_751planes_20x_myParameter.mat')
myParameter.lambda0 = 0.65;
myParameterNew.NAo = 0.95;
myParameter.dz = 0.02;
myParameter.NAo = 0.95;
myParameter.dx = 0.1568;
myParameter.dy = 0.1568
% read the dataset 
myamps = readtimeseries('S0019-2a_zstack_dz0-02um_751planes_40x_Amplitudes_256_x280_y260.tif');
myangs = readtimeseries('S0019-2a_zstack_dz0-02um_751planes_40x_Phases_256_x280_y260.tif');

%% extract the data
mycenter = [103 89];
mysize = [32 32 size(myamps,3)];
myamps = extract(myamps, mysize, mycenter);
myangs = extract(myangs, mysize, mycenter);

allAmp = myamps*exp(1i*myangs);

% reduce stack in Z-direction
zreduce_fac = 6;
allAmp_red = allAmp(:,:,1:zreduce_fac:end);
% cut out part around the sphere so that sphere is in center
if(1)
mycenter = [mysize(1)/2 mysize(2)/2 35];
mysize = [mysize(1) mysize(2) 70];
allAmp_red = extract(allAmp_red, mysize, mycenter);
end
% normalize amplitudes and get rid of the stripes
allAmp_red = allAmp_red/mean(allAmp_red, [], [1,2]);

% normalize phase
allAmpft = ft(allAmp_red);
midpos3D = MidPos(allAmpft);
allAmpft = allAmpft/exp(1i*angle(allAmpft(midpos3D(1),midpos3D(2),midpos3D(3))));
allAmp_red = double(ift(allAmpft));

% Save the data for tensorflow 
save('allAmp_red', 'allAmp_red', '-v7.3')


% update the parameter struct and save it 
myParameterNew = struct();
myParameterNew.lambda0 = myParameter.lambda0;
myParameterNew.NAo = myParameter.NAo;
myParameterNew.NAc = myParameter.NAc;
myParameterNew.NAci = 0;
myParameterNew.nEmbb = myParameter.nEmbb;
myParameterNew.dx = myParameter.dx;
myParameterNew.dy = myParameter.dy;
myParameterNew.dz = myParameter.dz*zreduce_fac;
myParameterNew.Nx = size(allAmp_red,1);
myParameterNew.Ny = size(allAmp_red,2);
myParameterNew.Nz = size(allAmp_red,3);
myParameterNew.shiftIcX = 0; 
myParameterNew.shiftIcY = 1;

save('myParameterNew', 'myParameterNew', '-v7.3')
