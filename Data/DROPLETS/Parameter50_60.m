% update the parameter struct and save it 
myParameterNew = struct();
myParameterNew.lambda0 = .65;
myParameterNew.NAo = .95;
myParameterNew.NAc = .52;
myParameterNew.NAci = 0;
myParameterNew.nEmbb = 1.33;
myParameterNew.dx = 0.1568;
myParameterNew.dy = 0.1568;
myParameterNew.dz = .12;
myParameterNew.Nx = 64;
myParameterNew.Ny = 64;
myParameterNew.Nz = 60;
myParameterNew.shiftIcX = 0; 
myParameterNew.shiftIcY = 0;

save('myParameter50_60', 'myParameterNew', '-v7.3')