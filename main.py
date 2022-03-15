import os
#import tables as tb
#import pandas as pd
import h5py
import numpy as np


from astropy.table import Table   #astropy routine for reading tables
import matplotlib.pyplot as plt   #plotting routines

# Random forest routine from scikit-learn:
from sklearn.ensemble import RandomForestRegressor

# Cross-Validation routines:
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict

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


def create_all_inputs(h5file, mag=31.8, snr=5, fileout_lephare='test_DC2_VALID_CAT_IN.in', fileout_delight='test_gal_fluxredshifts.txt'):
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
    print("U-Magnitude filter: {} original, {} removed, {} left ({} total for check).".format(data_f0.shape, data_f_removed.shape, data_f.shape, data_f_removed.shape[0]+data_f.shape[0]))

    # Get data better than SNR
    indexes_bad=filter_sigtonoise_entries(data_f,nsig=snr)
    data_f=np.delete(data_f,indexes_bad,axis=0)
    print("SNR filter: {} bad indexes, {} left ({} total for check).".format(indexes_bad.shape, data_f.shape, indexes_bad.shape[0]+data_f.shape[0]))

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
    
    #Photometry perturbed: doubling sizes of all errors
    #--------------------------------------------------
    u_magn = u_mag + np.sqrt(1)*u_err*np.random.randn(len(u_mag))
    g_magn = g_mag + np.sqrt(1)*g_err*np.random.randn(len(g_mag))
    r_magn = r_mag + np.sqrt(1)*r_err*np.random.randn(len(r_mag))
    i_magn = i_mag + np.sqrt(1)*i_err*np.random.randn(len(i_mag))
    z_magn = z_mag + np.sqrt(1)*z_err*np.random.randn(len(z_mag))
    y_magn = y_mag + np.sqrt(1)*y_err*np.random.randn(len(y_mag))

  
    # First: magnitudes only
    data_mags = np.column_stack((u_mag,g_mag,r_mag,i_mag,z_mag,y_mag))

    # Next: colors only
    data_colors = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag))

    # Next: colors and one magnitude
    data_colmag = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag, i_mag))
    perturbed_colmag=np.column_stack((u_magn-g_magn, g_magn-r_magn, r_magn-i_magn, i_magn-z_magn, z_magn-y_magn, i_magn))

    # Finally: colors, magnitude, and size
    #data_colmagsize = np.column_stack((u_mag-g_mag, g_mag-r_mag, r_mag-i_mag, i_mag-z_mag, z_mag-y_mag, i_mag, rad))
    
    data_z = z
    print('Magnitudes, colors, Colors+mag, perturbed colors+mag, Spectro-Z for ML ; input CAT file for LEPHARE ; input flux-redshift file for Delight :')
    return data_mags, data_colors, data_colmag, perturbed_colmag, data_z, fileout_lephare, fileout_delight
    

filename = 'test_dc2_validation_9816.hdf5'
test_data_mags, test_data_colors, test_data_colmag, test_perturbed_colmag, test_z, test_fileout_lephare, test_fileout_delight = create_all_inputs(filename)

print(test_data_mags.shape, test_data_colors.shape, test_data_colmag.shape, test_perturbed_colmag.shape, test_z.shape, test_fileout_lephare, test_fileout_delight)
#h5_file.close()
