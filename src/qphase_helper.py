# -*- coding: utf-8 -*-
import NanoImagingPack as nip
import numpy as np



def extractQPhaseCCterm(allholo, cc_center=None, cc_size=None):
    # UNTITLED5 Summary of this function goes here
    #   Detailed explanation goes here
    if cc_center is None:
        cc_center = allholo.shape[1:-1]//2
        
    if cc_size is None:
        cc_center = np.array((600,600))

    # Extract CC-TERM
    if (0):
        print('I have to impolement this some time''')
        '''
        AH=dipshow(ft(allholo(:,:,round(size(allholo,3)/2))), 'log'); #  find edges of CC signal
        diptruesize(AH,400)
        # fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) #  find edges of CC signal
        fprintf('Get the center of the CC TERM (upper right maximum)\n')
        fprintf('1. Click the center of the CC Term\n');
        # fprintf('2. Click the outer rim of the roi CC Term you wanna extract ')
        
        #  crop the freq space? Better ?! k-sampling is then varied!
        fprintf('2. Click the outer rim of the roi CC Term you wanna extract\n ')
        roi_coordinates = dipgetcoords(AH,2);
        cc_center = [roi_coordinates(1, 1) roi_coordinates(1,2)]; #  StartX, StartY
    end
    if(cc_size==0)
        difX = abs(roi_coordinates(1)-roi_coordinates(2));
        difY = abs(roi_coordinates(3)-roi_coordinates(4));
        cc_size = max(difX, difY); # size of the Box
        cc_size = 2*[cc_size cc_size];
    end
    '''
    
    # #  damp edges
    print('Damp the Edges in 3D - also takes some time')
    de_flat = nip.DampEdge(nip.image(np.squeeze(allholo)),rwidth = .25, axes=(0,1,2)) #0.25,2,1,3);
    print('Now fouriertransformin and extracting the Hologram. This can take a while!\n')
    f2_flat=nip.ft2d(de_flat);
    fe_flat=nip.extractFt(f2_flat,(f2_flat.shape[0], cc_size[0], cc_size[1]), (f2_flat.shape[0]//2, cc_center[0], cc_center[1]))
    mymask = nip.rr(cc_size, placement='center',freq='ftfreq')<.45
    fe_flat = mymask*fe_flat
    
    print('We are masking the CC-term with a circular pupil function!')
    allAmp = nip.ift2d(fe_flat)
    #  allAmp_ft = extract(f2_flat, [cc_size(1) cc_size(2) size(f2_flat,3)], [cc_center(1) cc_center(2) round(size(f2_flat,3)/2)]);
    # allAmp = ift2d(allAmp_ft);# /exp(1i*angle(allAmp(floor(size(allAmp,1)/2), floor(size(allAmp,1)/2), :))))
    print('ROI has been extracted')
    
    return allAmp



def QPhaseNormBackground(allAmp, roi_center=None, roi_size=None):
    ''' Extract empty ROI to remove its background phase and amplitude '''
    if roi_center is None:
        print('we set the ROI-center to default:')
        roi_coordinates = allAmp.shape//2
        print(str(roi_coordinates))
    if roi_size is None:
        print('we set the ROI-size to default:')
        roi_size =(allAmp.shape[0],50,50)
        print(str(roi_size))
    #if len(roi_center)<3:
    #    roi_center = np.array((allAmp.shape[0]//2, roi_center[0], roi_center[1]))
    #if len(roi_size)<3:
    #    roi_size = np.array((allAmp.shape[0], roi_size[0], roi_size[1]))
        
    # 1.) first get rid of the intensity scaling of the source - correct for
    # intensity fluctuations along Z
    if(0):
        print('1.) normalize the phases') 
        allAmp_emptyroi = nip.extract(allAmp, roi_size, roi_center, checkComplex=False)
        tmp_mean = np.exp(-1j*np.angle((np.mean(allAmp_emptyroi,(1, 2)))))
        allAmp *= np.expand_dims(np.expand_dims(tmp_mean,-1),-1)
        #allAmp_emptyroi = extract(allAmp, roi_size, (roi_center));
        tmp_mean = np.mean(np.abs(allAmp_emptyroi),(1,2))
        allAmp /= np.expand_dims(np.expand_dims(tmp_mean,-1),-1)
        #allAmp = allAmp*exp(-1i*pi);
        # This should already be enough. Intensity is scaled to 1, phase is 0
        # bring ref-
    # 2.) Then subtract the mean - we want to have zero field in background
    # regions
    
    print('2.) normalize the real/imaginary to be one')
    print(roi_size)
    print(roi_center)
    print(allAmp.shape)
    allAmp_emptyroi = nip.extract(allAmp, roi_size, roi_center, checkComplex=False) 
    myBackground = np.mean(allAmp_emptyroi,(1,2))
    allAmp = allAmp-np.expand_dims(np.expand_dims(myBackground,-1),-1)
    
    # remove global phase and mangitude from sub-roi mean
    #; (exp(1i*(mean(angle(allAmp_emptyroi2),[],[1 2])));
    #allAmp = allAmp/(mean(abs(allAmp_emptyroi2),[],[1 2]));
    
    print('Done!\n');
    
    return allAmp
            
