###############
### IMPORTS ###
###############
import os
import sys
import time
import subprocess
#import tables as tb
#import pandas as pd
import h5py
import numpy as np
from astropy.table import Table   #astropy routine for reading tables
import matplotlib.pyplot as plt   #plotting routines
from matplotlib.backends.backend_pdf import PdfPages

# Random forest routine from scikit-learn:
from sklearn.ensemble import RandomForestRegressor

# Cross-Validation routines:
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict

# Useful Delight routines:
#sys.path.append(os.path.realpath(os.path.normpath(os.path.join('./', delight_dir))))
#from delight.io import *
#from delight.utils import *

#################
### FONCTIONS ###
#################
def load_raw_hdf5_data(infile, groupname='None'):
    """
    read in h5py hdf5 data, return a dictionary of all of the keys
    """
    data = {}
    infp = h5py.File(infile, "r")
    if groupname != 'None':
        f = infp[groupname]
    else:
        f = infp
    for key in f.keys():
        data[key] = np.array(f[key])
    infp.close()
    return data

def group_entries(f):
    """
    group entries in single numpy array
    
    """
    galid = f['id'][()][:,np.newaxis]
    redshift = f['redshift'][()][:,np.newaxis]
    mag_err_g_lsst = f['mag_err_g_lsst'][()][:,np.newaxis]
    mag_err_i_lsst = f['mag_err_i_lsst'][()][:,np.newaxis]
    mag_err_r_lsst = f['mag_err_r_lsst'][()][:,np.newaxis]
    mag_err_u_lsst = f['mag_err_u_lsst'][()][:,np.newaxis]
    mag_err_y_lsst = f['mag_err_y_lsst'][()][:,np.newaxis]
    mag_err_z_lsst = f['mag_err_z_lsst'][()][:,np.newaxis]
    mag_g_lsst = f['mag_g_lsst'][()][:,np.newaxis]
    mag_i_lsst = f['mag_i_lsst'][()][:,np.newaxis]
    mag_r_lsst = f['mag_r_lsst'][()][:,np.newaxis]
    mag_u_lsst = f['mag_u_lsst'][()][:,np.newaxis]
    mag_y_lsst = f['mag_y_lsst'][()][:,np.newaxis]
    mag_z_lsst = f['mag_z_lsst'][()][:,np.newaxis]
    context = np.full(galid.shape, 255)
    
    full_arr=np.hstack( (galid, mag_u_lsst, mag_err_u_lsst,\
                                mag_g_lsst, mag_err_g_lsst,\
                                mag_r_lsst, mag_err_r_lsst,\
                                mag_i_lsst, mag_err_i_lsst,\
                                mag_z_lsst, mag_err_z_lsst,\
                                mag_y_lsst, mag_err_y_lsst,\
                                context, redshift) )
    return full_arr

def filter_mag_entries(d, mag_filt):
    """
    Filter accoding magnitudes
    
    Only remove creazy u values
    """
    
    u=d[:,1]
    
    idx_u= np.where(u>mag_filt)[0]  
    #d_del=np.delete(d,idx_u,axis=0)
    
    return np.array(idx_u)

def filter_zmin_entries(d, zmin=None):
    """
    Filter accoding to redshifts
    
    Remove data below zmin
    """
    
    z=d[:,14]
    
    if not zmin is None:
        idx_z = np.where(z<zmin)[0]  
        #d_del=np.delete(d,idx_u,axis=0)
        return np.array(idx_z)
    else:
        return np.empty_like([], dtype=int)
    
def filter_zmax_entries(d, zmax=None):
    """
    Filter accoding to redshifts
    
    Remove data above zmax
    """
    
    z=d[:,14]
    
    if not zmax is None:
        idx_z = np.where(z>zmax)[0]  
        #d_del=np.delete(d,idx_u,axis=0)
        return np.array(idx_z)
    else:
        return np.empty_like([], dtype=int)

def mag_to_flux(d, nbFilt=6):
    """  
    Convert magnitudes to fluxes

    :param d:
    :return:
    """

    fluxes=np.zeros_like(d)
   
    fluxes[:,0]=d[:,0]
    fluxes[:,13]=d[:,13]
    fluxes[:,14]=d[:,14]
    
    for idx in np.arange(nbFilt):
        fluxes[:,1+2*idx]=np.power(10,-0.4*d[:,1+2*idx])
        fluxes[:,2+2*idx]=fluxes[:,1+2*idx]*d[:,2+2*idx]
    return fluxes

def filter_fluxes_entries(d, nsig=5, nbFilt=6):
    """
    """ 
    
    indexes=[]
    #indexes=np.array(indexes,dtype=np.int)
    indexes=np.array(indexes,dtype=int)
    
    for idx in np.arange(nbFilt):
        ratio=d[:,1+2*idx]/d[:,2+2*idx]  # flux divided by sigma-flux
        bad_indexes=np.where(ratio<nsig)[0]
        indexes=np.concatenate((indexes,bad_indexes))
        
    indexes=np.unique(indexes)
    return np.sort(indexes)

def filter_sigtonoise_entries(d, nsig=5, nbFilt=6):
    """
    """ 
    
    indexes=[]
    #indexes=np.array(indexes,dtype=np.int)
    indexes=np.array(indexes,dtype=int)
    
    for idx in np.arange(nbFilt):
        errMag=d[:,2+2*idx]  # error in M_AB
        bad_indexes=np.where(errMag > (1/nsig))[0]
        indexes=np.concatenate((indexes,bad_indexes))
        
    indexes=np.unique(indexes)
    return np.sort(indexes)


def create_all_inputs(h5file, mag=31.8, snr=5, zMin=None, zMax=None, returnErrors=False, fileout_lephare='test_DC2_VALID_CAT_IN.in', fileout_delight='test_gal_fluxredshifts.txt'):
    h5_file = load_raw_hdf5_data(h5file, groupname='photometry')

    ## produce a numpy array
    dataArray=group_entries(h5_file)
    #print(dataArray.shape)

    # Filter mag entries
    indexes=filter_mag_entries(dataArray, mag_filt=mag)
    #print(indexes.shape)
    data_f0=dataArray
    data_f=np.delete(dataArray,indexes,axis=0)
    data_f_removed=dataArray[indexes,:]
    print("U-Magnitude filter: {} original, {} removed, {} left ({} total for check).".format(data_f0.shape,\
                                                                                              data_f_removed.shape,\
                                                                                              data_f.shape,\
                                                                                              data_f_removed.shape[0]+data_f.shape[0]))

    # Get data better than SNR
    indexes_bad=filter_sigtonoise_entries(data_f,nsig=snr)
    data_f=np.delete(data_f,indexes_bad,axis=0)
    print("SNR filter: {} bad indexes, {} left ({} total for check).".format(indexes_bad.shape,\
                                                                             data_f.shape,\
                                                                             indexes_bad.shape[0]+data_f.shape[0]))

    # Get data higher than zMin
    indexes_zlow=filter_zmin_entries(data_f,zmin=zMin)
    data_f=np.delete(data_f,indexes_zlow,axis=0)
    print("Zmin filter: {} bad indexes, {} left ({} total for check).".format(indexes_zlow.shape,\
                                                                             data_f.shape,\
                                                                             indexes_zlow.shape[0]+data_f.shape[0]))
    
    # Get data lower than zMax
    indexes_zhigh=filter_zmax_entries(data_f,zmax=zMax)
    data_f=np.delete(data_f,indexes_zhigh,axis=0)
    print("Zmax filter: {} bad indexes, {} left ({} total for check).".format(indexes_zhigh.shape,\
                                                                             data_f.shape,\
                                                                             indexes_zlow.shape[0]+data_f.shape[0]))

    # Generate file for LEPHARE++
    np.savetxt(fileout_lephare, data_f, fmt=['%1i', '%1.6g', '%1.6g',\
                                                             '%1.6g', '%1.6g',\
                                                             '%1.6g', '%1.6g',\
                                                             '%1.6g', '%1.6g',\
                                                             '%1.6g', '%1.6g',\
                                                             '%1.6g', '%1.6g',\
                                                             '%1i', '%1.3f'])
    
    # Generate flux-redshift file for Delight
    data_flux = mag_to_flux(data_f)
    gal_id, u_flux, uf_err,\
            g_flux, gf_err,\
            r_flux, rf_err,\
            i_flux, if_err,\
            z_flux, zf_err,\
            y_flux, yf_err,\
            context, z = np.hsplit(data_flux, data_flux.shape[1])
    print(gal_id.shape)
    end_col=np.zeros_like(z)
    delightUntFac=2.22e10
    data_flux = np.column_stack((u_flux*delightUntFac, (uf_err*delightUntFac)**2,\
                                 g_flux*delightUntFac, (gf_err*delightUntFac)**2,\
                                 r_flux*delightUntFac, (rf_err*delightUntFac)**2,\
                                 i_flux*delightUntFac, (if_err*delightUntFac)**2,\
                                 z_flux*delightUntFac, (zf_err*delightUntFac)**2,\
                                 y_flux*delightUntFac, (yf_err*delightUntFac)**2,\
                                 z, end_col))
    
    np.savetxt(fileout_delight, data_flux) #, fmt=['%1i', '%1.6g', '%1.6g',\
                                           #                  '%1.6g', '%1.6g',\
                                           #                  '%1.6g', '%1.6g',\
                                           #                  '%1.6g', '%1.6g',\
                                           #                  '%1.6g', '%1.6g',\
                                           #                  '%1.6g', '%1.6g',\
                                           #                  '%1i', '%1.3f'])
    
    # Generate datasets for ML regressor (typ. random forrest)
    gal_id, u_mag, u_err,\
            g_mag, g_err,\
            r_mag, r_err,\
            i_mag, i_err,\
            z_mag, z_err,\
            y_mag, y_err,\
            context, z = np.hsplit(data_f, data_f.shape[1])
    
    ##Photometry perturbed: doubling sizes of all errors
    ##--------------------------------------------------
    #u_magn = u_mag + np.sqrt(1)*u_err*np.random.randn(len(u_mag))
    #g_magn = g_mag + np.sqrt(1)*g_err*np.random.randn(len(g_mag))
    #r_magn = r_mag + np.sqrt(1)*r_err*np.random.randn(len(r_mag))
    #i_magn = i_mag + np.sqrt(1)*i_err*np.random.randn(len(i_mag))
    #z_magn = z_mag + np.sqrt(1)*z_err*np.random.randn(len(z_mag))
    #y_magn = y_mag + np.sqrt(1)*y_err*np.random.randn(len(y_mag))

  
    # First: magnitudes only
    data_mags = np.column_stack((u_mag, g_mag, r_mag, i_mag, z_mag, y_mag))
    err_mags = np.column_stack((u_err, g_err, r_err, i_err, z_err, y_err))

    # Next: colors only
    data_colors = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag))

    # Next: colors and one magnitude
    data_colmag = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag, i_mag))
    #perturbed_colmag = np.column_stack((u_magn-g_magn, g_magn-r_magn, r_magn-i_magn, i_magn-z_magn, z_magn-y_magn, i_magn))

    # Finally: colors, magnitude, and size
    #data_colmagsize = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag, i_mag, rad))
    
    data_z = z
    #print('Magnitudes, colors, Colors+mag, perturbed colors+mag, Spectro-Z for ML ; input CAT file for LEPHARE ; input flux-redshift file for Delight :')
    #return data_mags, data_colors, data_colmag, perturbed_colmag, data_z, fileout_lephare, fileout_delight
    if returnErrors:
        print('Magnitudes, magnitude errors, colors, Colors+mag, Spectro-Z for ML ; input CAT file for LEPHARE ; input flux-redshift file for Delight :')
        return data_mags, err_mags, data_colors, data_colmag, data_z, fileout_lephare, fileout_delight
    else:
        print('Magnitudes, colors, Colors+mag, Spectro-Z for ML ; input CAT file for LEPHARE ; input flux-redshift file for Delight :')
        return data_mags, data_colors, data_colmag, data_z, fileout_lephare, fileout_delight

# A function that we will call a lot: makes the zphot/zspec plot and calculates key statistics
# Returns a figure that can then be shown or saved using pyplot methods.
def plot_and_stats(z_spec, z_phot, i_mag=[], title=''):
    x = np.arange(0,5.4,0.05)
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)
    
    print('DEBUG: zphot size = {}, zspec size = {}'.format(z_phot.shape, z_spec.shape))
    deltaZ = (z_phot - z_spec)/(1 + z_spec)
    
    mask = np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15
    notmask = ~mask 
    
    # Standard Deviation of the predicted redshifts compared to the data:
    #-----------------------------------------------------------------
    std_result = np.std((z_phot - z_spec)/(1 + z_spec), ddof=1)
    std_res_uncert = std_result / np.sqrt(2*len(z_spec))
    print(title+' :')
    print('Standard Deviation: %6.4f' % std_result)
    print('Standard Deviation Uncertainty: %6.4f' % std_res_uncert)

    # Normalized MAD (Median Absolute Deviation):
    #------------------------------------------
    nmad = 1.48 * np.median(np.abs((z_phot - z_spec)/(1 + z_spec)))
    print('Normalized MAD: %6.4f' % nmad)

    # Percentage of delta-z > 0.15(1+z) outliers:
    #-------------------------------------------
    eta = np.sum(np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15)/len(z_spec)
    print('Delta z >0.15(1+z) outliers: %6.3f percent' % (100.*eta))
    
    # Median offset (normalized by (1+z); i.e., bias:
    #-----------------------------------------------
    bias = np.median(((z_phot - z_spec)/(1 + z_spec)))
    sigbias=std_result/np.sqrt(0.64*len(z_phot))
    print('Median offset: %6.3f +/- %6.3f' % (bias,sigbias))
    print('\n')
    
    # make photo-z/spec-z plot
    #------------------------
    fig, axs = plt.subplots(1,3,figsize=(28, 8))
    
    #add lines to indicate outliers
    axs[0].plot(x, outlier_upper, 'k--')
    axs[0].plot(x, outlier_lower, 'k--')
    axs[0].plot(z_spec[mask], z_phot[mask], 'r.', markersize=6,  alpha=0.5)
    axs[0].plot(z_spec[notmask], z_phot[notmask], 'b.',  markersize=6, alpha=0.5)
    axs[0].plot(x, x, linewidth=1.5, color = 'red')
    axs[0].set_xlim([0.0, 3])
    axs[0].set_ylim([0.0, 3])
    axs[0].set_xlabel('$z_{\mathrm{spec}}$', fontsize = 27)
    axs[0].set_ylabel('$z_{\mathrm{photo}}$', fontsize = 27)
    axs[0].grid(alpha = 0.8)
    axs[0].tick_params(labelsize=15)
    
    axs[1].plot(z_spec[mask], deltaZ[mask], 'r.', markersize=6,  alpha=0.5)
    axs[1].plot(z_spec[notmask], deltaZ[notmask], 'b.', markersize=6,  alpha=0.5)
    axs[1].hlines([-0.15, 0.15], xmin=0.0, xmax=3.01, colors='k', linestyles='--')
    axs[1].set_xlim([0.0, 3])
    axs[1].set_xlabel('$z_{\mathrm{spec}}$', fontsize = 27)
    axs[1].set_ylabel('$(z_{\mathrm{photo}} - z_{\mathrm{spec}}) / (1 + z_{\mathrm{spec}})$', fontsize = 27)
    axs[1].grid(alpha = 0.8)
    axs[1].tick_params(labelsize=15)
    
    if len(i_mag) > 0:
        xmin, xmax = np.min(i_mag), np.max(i_mag)
        axs[2].plot(i_mag[mask], deltaZ[mask], 'r.', markersize=6,  alpha=0.5)
        axs[2].plot(i_mag[notmask], deltaZ[notmask], 'b.', markersize=6,  alpha=0.5)
        axs[2].hlines([-0.15, 0.15], xmin=xmin, xmax=xmax, colors='k', linestyles='--')
        axs[2].set_xlim([xmin, xmax])
        axs[2].set_xlabel('$mag_{\mathrm{i band}}$', fontsize = 27)
        axs[2].set_ylabel('$(z_{\mathrm{photo}} - z_{\mathrm{spec}}) / (1 + z_{\mathrm{spec}})$', fontsize = 27)
        axs[2].grid(alpha = 0.8)
        axs[2].tick_params(labelsize=15)
    
    fig.suptitle(title+' : $\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad+'$(\Delta z)>0.15(1+z) $ outliers = %6.3f'%(eta*100)+'%', fontsize=18)
    return fig

def plot_random_pdz(z1, pdz1, z2, pdz2, z3, pdz3, z4, pdz4, label1='', label2='', label3='', label4=''):
    ncol = 4
    fig, axs = plt.subplots(5, ncol, figsize=(12, 12), sharex=True, sharey=False)
    axs = axs.ravel()
    z = fluxredshifts[:, redshiftColumn]
    sel = np.random.choice(nobj, axs.size, replace=False)
    lw = 2
    for ik in range(axs.size):
        k = sel[ik]
        zspec=fluxredshifts[k, redshiftColumn]
        #print(zspec, np.where(zs == zspec))
        dummy=np.array([])
        for z in zs:
            dummy=np.append(dummy, (z-zspec))
        dummy=np.absolute(dummy)
        galId=Id[np.argmin(dummy)]
        axs[ik].plot(z1, pdz1[k,:], lw=lw, label=label1)
        axs[ik].plot(z2, pdz2[k,:], lw=lw, label=label2)
        axs[ik].plot(z3, pdz3[np.argmin(dummy), 1:], lw=lw, label=label3)
        axs[ik].plot(z4, pdz4, lw=lw, label=label4)
        axs[ik].axvline(zspec, c="k", lw=1, label='Spec-z')
        ymax = np.max(np.concatenate((pdz1[k,:], pdz2[k,:], pdz3[np.argmin(dummy), 1:], pdz4)))
        axs[ik].set_ylim([0, ymax*1.2])
        axs[ik].set_xlim([0, 3.1])
        axs[ik].set_yticks([])
        axs[ik].set_xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
    for i in range(ncol):
        axs[-i-1].set_xlabel('Redshift', fontsize=10)
    axs[0].legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.4))
    return fig

def transposeSEDtoZ(wavelength, intensity, ztarget, zphot=0): ## Flux = integrale{ intensity(lambda)*transmitivity(lambda)*dlambda }
    lambda0 = wavelength / (1+zphot)
    int0 = intensity ## * (1+zphot)
    lambdaZ = lambda0 * (1+ztarget)
    intensityZ = int0 ## / (1+ztarget)
    return lambdaZ, intensityZ, lambda0, int0

def SEDtoFlux_obs(wavelength, intensity, filters, ref=None): ## compute the flux of each template in each filter
    ## interpolate / extrapolate filters on the wavelength array
    interpFilt = np.zeros((len(wavelength), len(filters)))
    fluxes = np.zeros(len(filters))
    refFact = 1.0
    if not ref is None:
        refFact = np.interp(ref, wavelength, intensity, left=0.0, right=0.0) * ref
    for indFilt in np.arange(0, len(filters)):
        ## interpolate / extrapolate filters on the wavelength array
        lambdaF = np.loadtxt(filters[indFilt])[:, 0]
        transmitF = np.loadtxt(filters[indFilt])[:, 1]
        norm = np.trapz(transmitF/lambdaF, x=lambdaF) ## norm as computed in Delight processSEDs.py
        #print('Curiosity: norm = {}'.format(norm))
        interpFilt[:, indFilt] = np.interp(wavelength, lambdaF, transmitF, left=0.0, right=0.0)
        ## for each filter : multiply intensity * interpolated filter
        interpFilt[:, indFilt] *= (intensity * wavelength / refFact)
        ## compute the flux for each filter by integration over wavelength
        fluxes[indFilt] = np.trapz(interpFilt[:, indFilt], wavelength) / norm
    return fluxes

def buildTemplates(SEDs, redshifts, filters, output='magnitude'): ## create a set of templates of size nb of SEDs * nb of redshifts
    nbSED = len(SEDs)
    nbZ = len(redshifts)
    nbFilters = len(filters)
    #magOrFlux = np.zeros(nbZ*nbSED, nbFilters)
    magOrFlux = np.empty((0, nbFilters))
    #sedCol = np.empty_like([])
    sedCol = []
    allZCol = np.empty_like([])
    loc=-1
    for indZ in np.arange(nbZ):
        ztarget = redshifts[indZ]
        approxDL = np.exp( 30.5 * ztarget**0.04 - 21.7)
        for indSed in np.arange(nbSED):
            loc+=1
            wl = np.loadtxt(SEDs[indSed])[:, 0]
            intensity = np.loadtxt(SEDs[indSed])[:, 1]
            wlZ, intZ, wl0, int0 = transposeSEDtoZ(wl, intensity, ztarget, zphot=0)
            #sedCol = np.append(sedCol, [(SEDs[indSed], ztarget)])
            #sedCol.append( (SEDs[indSed], ztarget) )
            sedCol.append( (indSed, ztarget) )
            allZCol = np.append(allZCol, ztarget)
            fluxesSed = SEDtoFlux_obs(wlZ, intZ, filters)
            fluxesSed /= 4 * np.pi * approxDL**2 # * (1+ztarget)**2
            #magOrFlux[indZ * nbSED + indSed, :] = fluxesSed
            magOrFlux = np.row_stack((magOrFlux, fluxesSed))
    if output == 'magnitude':
        magOrFlux = - 2.5 * np.log10(magOrFlux)
    sedTupArr = np.empty(len(sedCol), dtype=object)
    sedTupArr[:] = sedCol
    return magOrFlux, allZCol, sedTupArr

def sedColors(SEDs, filters):
    nbSED = len(SEDs)
    nbFilters = len(filters)
    nbColors = nbFilters-1
    colorsArray = np.empty((nbSED, nbColors))
    for indSed in np.arange(nbSED):
        wavelength = np.loadtxt(SEDs[indSed])[:, 0]
        luminosity = np.loadtxt(SEDs[indSed])[:, 1]
        fluxesSed = SEDtoFlux_obs(wavelength, luminosity, filters)
        for indCol in np.arange(nbColors):
            color = - 2.5 * ( np.log10(fluxesSed[indCol])\
                             - np.log10(fluxesSed[indCol+1]) )
            colorsArray[indSed, indCol] = color
    return colorsArray

### ensuite :
#seds = ['sed0.txt', 'sed1.txt', ..., 'sedN.txt']
#filters = ['filtU.pb', 'filtG.pb', ..., 'filtY.pb']
#redshifts = np.arange(0, 3.01, 0.01)
#magOrFlux, allZcol, sedCol=buildTemplates(seds, redshifts, filters)
#magOrFlux = np.column_stack(magOrFlux, allZcol)
#regrn = RandomForestRegressor(n_estimators = nb_est, max_depth = reg_depth, max_features = 'auto')
#regrn.fit(magOrFlux, sedCol)
#templatePhot = regrn.predict(test_data_mags)