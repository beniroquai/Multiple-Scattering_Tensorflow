%% This is the file holding the parameters for the EM experiment provided by
% JAmes McNally
myParameter = struct()

% necessary for the script to work
myParameter.lambda0= 2.431; % nm
myParameter.nEmbb = 1.0; %
myParameter.NAc = 0.0206; % outer radius of the condensers NA
myParameter.NAci = 0.0145; % inner radius of the condensers NA
myParameter.NAo = 0.0483;
myParameter.dx = 5.48; % nm
myParameter.dy = 5.48; % nm
myParameter.dz = 40; % nm
myParameter.Nx = 187;
myParameter.Ny = 187;
myParameter.Nz = 401;



myParameter.filename= '/media/useradmin/Data/Benedict/QPHASE/BEAD/Beads_40x_100a.cdf'

save('Parameter', 'myParameter', '-v7.3')