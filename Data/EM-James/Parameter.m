%% This is the file holding the parameters for the EM experiment provided by
% JAmes McNally
myParameter = struct()

% necessary for the script to work
myParameter.lambda0= 2.431; % nm
myParameter.nEmbb = 1.0; %
myParameter.NAc = 0.0206; % outer radius of the condensers NA
myParameter.NAci = 0.0145; % inner radius of the condensers NA
myParameter.NAo = 0.0486;
myParameter.dx = 5.48; % nm
myParameter.dy = 5.48; % nm
myParameter.dz = 40; % nm
myParameter.Nx = 187;
myParameter.Ny = 187;
myParameter.Nz = 401;


myParameter.filename= '/media/useradmin/Data/Benedict/QPHASE/BEAD/Beads_40x_100a.cdf'

save('Parameter', 'myParameter', '-v7.3')

% 270nmBead510eV_focus20171025-02-a.mrc
% Experimental data set size is 187x187x401
%  
% Imaging Conditions
% lambda = 2.431 nm (510 eV)
% xy pixel size (object space) = 5.48 nm
% z step size (object space) = 40 nm
% Magnification = 3652.1
%  
% NAobj = 0.0486
% NAcond = 0.0206
% NAcond =0.0145
%  
% Gold Shell Parameters
% Inner core is silica (SiO2), Ri = 114 nm, delta_silica = 1.564e-3, beta_silica = 0.262e-3 (at 510 eV)
% Outer shell is gold (Au), Ro = 135 nm, delta_gold = 4.761e-3, beta_gold = 4.590e-3 (at 510 eV)
% (so gold outer shell is 21 nm thick).
%  
% 60nmBead510eV_focus20171026-02-a.mrc
% Experimental data set size is 200x200x301
%  
% Imaging Conditions
% lambda = 2.431 nm (510 eV)
% xy pixel size (object space) = 5.48 nm
% z step size (object space) = 40 nm
% Magnification = 3652.1
%  
% NAobj = 0.0486
% NAcond = 0.0206
% NAcond =0.0145
%  
% Gold Shell Parameters
% 60 nm diameter gold bead (Au), Ro = 30 nm, delta_gold = 4.761e-3, beta_gold = 4.590e-3 (at 510 eV)
% 
