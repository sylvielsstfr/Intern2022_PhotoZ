###########################################
## Fonction to run comparisons between   ##
## several photoZ estimation methods:    ##
##  - pure template fitting: LEPHARE++   ##
##  - pure Machine Learning with sklearn ##
##  - TF & ML hybrid: Delight.           ##
## Joseph Chevalier, IJCLab - IN2P3      ##
## 16/03/2022                            ##
###########################################

###################
### USER INPUTS ###
###################
delight_dir                = 'Delight'                        # directory where Delight is installed, relative should work.
desc_dir                   = 'desc-dc2'                       # subdir of Delight dir, for specific use. If none, use an empty string ""
delight_paramDir           = 'tmp'                            # subdir of descDir
delight_runFile            = 'run_delight_descdc2.py'         # usually in descDir
delight_paramFile          = 'parameters_DESC-DC2.cfg'        # usually in paramDir
test_filename              = 'test_dc2_validation_9816.hdf5'  # relative path
train_filename             = 'train_dc2_validation_9816.hdf5' # relative path
test_fileout_delight       = 'test_gal_fluxredshifts.txt'     # file name only - will be created in the appropriate directory, until this is automated
train_fileout_delight      = 'train_gal_fluxredshifts.txt'    # file name only - will be created in the appropriate directory, until this is automated
lephare_dir                = 'LEPHARELSST'                    # relative path - should be the directory where LSST.para and runLePhareLSST.sh are located and run.
test_fileout_lephare       = 'test_DC2_VALID_CAT_IN.in'       # file name only - will be created in the appropriate directory, until this is automated
train_fileout_lephare      = 'train_DC2_VALID_CAT_IN.in'      # file name only - will be created in the appropriate directory, until this is automated
mag_filt                   = 31.8
snr_filt                   = 5.0
vc, vl, alC, alL           = 0.1, 0.1, 1e3, 1e2
ellPriorSigma, zPriorSigma = 0.5, 0.2
nb_est                     = 50
reg_depth                  = 30

################################
### END OF USER INPUTS       ###
### DO NOT MODIFY CODE BELOW ###
################################


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
sys.path.append(os.path.realpath(os.path.normpath(os.path.join('./', delight_dir))))
from delight.io import *
from delight.utils import *

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
    data_mags = np.column_stack((u_mag,g_mag,r_mag,i_mag,z_mag,y_mag))

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
    print('Magnitudes, colors, Colors+mag, Spectro-Z for ML ; input CAT file for LEPHARE ; input flux-redshift file for Delight :')
    return data_mags, data_colors, data_colmag, data_z, fileout_lephare, fileout_delight

# A function that we will call a lot: makes the zphot/zspec plot and calculates key statistics
# Returns a figure that can then be shown or saved using pyplot methods.
def plot_and_stats(z_spec, z_phot, i_mag=[], title=''):
    x = np.arange(0,5.4,0.05)
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)
    
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

def plot_random_pdz(z1, pdz1, label1='', z2, pdz2, label2='', z3, pdz3, label3='', z4, pdz4, label4=''):
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
        axs[ik].plot(z1, pdz1, lw=lw, label=label1)
        axs[ik].plot(z2, pdz2, lw=lw, label=label2)
        axs[ik].plot(z3, pdz3, lw=lw, label=label3)
        axs[ik].plot(z4, pdz4, lw=lw, label=label4)
        axs[ik].axvline(zspec, c="k", lw=1, label='Spec-z')
        ymax = np.max(np.concatenate((pdz1, pdz2, pdz3, pdz4)))
        axs[ik].set_ylim([0, ymax*1.2])
        axs[ik].set_xlim([0, 3.1])
        axs[ik].set_yticks([])
        axs[ik].set_xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
    for i in range(ncol):
        axs[-i-1].set_xlabel('Redshift', fontsize=10)
    axs[0].legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.4))
    return fig

#####################################################
### CODE, DO NOT MODIFY UNLESS STRICTLY NECESSARY ###
#####################################################

# Generate the inputs
t_start=time.time()
delight_absRunFile=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, delight_runFile)))
delight_absParamFile=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, delight_paramDir, delight_paramFile)))
params = parseParamFile(delight_absParamFile, verbose=False)
test_fileout_delight = params['target_catFile']
train_fileout_delight = params['training_catFile']
testFile_absPath=os.path.realpath(os.path.normpath(os.path.join("./", test_filename)))
trainFile_absPath=os.path.realpath(os.path.normpath(os.path.join("./", train_filename)))
delight_testFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, test_fileout_delight)))
delight_trainFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, train_fileout_delight)))
lephare_testFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir, test_fileout_lephare)))
lephare_trainFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir, train_fileout_lephare)))

test_data_mags, test_data_colors, test_data_colmag, test_perturbed_colmag, test_z, dummy1, dummy2 =\
    create_all_inputs(testFile_absPath,\
                      mag=mag_filt,\
                      snr=snr_filt,\
                      fileout_lephare=lephare_testFileoutAbs,\
                      fileout_delight=delight_testFileoutAbs)

train_data_mags, train_data_colors, train_data_colmag, train_perturbed_colmag, train_z, dummy3, dummy4 =\
    create_all_inputs(trainFile_absPath,\
                      mag=mag_filt,\
                      snr=snr_filt,\
                      fileout_lephare=lephare_trainFileoutAbs,\
                      fileout_delight=delight_trainFileoutAbs)

# DEBUG #
print(test_data_mags.shape, test_data_colors.shape, test_data_colmag.shape, test_perturbed_colmag.shape, test_z.shape, test_fileout_lephare, test_fileout_delight)
print(train_data_mags.shape, train_data_colors.shape, train_data_colmag.shape, train_perturbed_colmag.shape, train_z.shape, train_fileout_lephare, train_fileout_delight)
# END DEBUG #

t_init=time.time()
print('Input creation duration: {} s'.format(t_init-t_start))

# Run the ML - call the appropriate python fonctions
## We need to set up an implementation of the scikit-learn RandomForestRegressor in an object called 'regrn'. 
regrn = RandomForestRegressor(n_estimators = nb_est, max_depth = reg_depth, max_features = 'auto')

## Train the regressor using the training data
regrn.fit(train_data_mags, train_z)
t_MLtrain=time.time()

## Apply the regressor to predict values for the test data
z_phot = regrn.predict(test_data_mags)
t_MLpredict=time.time()

# Run Delight - call the appropriate python fonctions
execDir = os.path.realpath(os.getcwd())
os.chdir(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir))))
from run_delight_descdc2 import run_full_delight_confFile
run_full_delight_confFile(delight_absParamFile)
os.chdir(execDir)

# Run LePhare - call the appropriate shell fonctions
os.chdir(os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir))))
os.environ['LEPHAREDIR'] = os.path.realpath(os.path.normpath(os.path.join("./", 'LEPHARE')))
os.environ['LEPHAREWORK'] = os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir)))
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['CAT_FILE_IN'] = lephare_testFileoutAbs
subprocess.run('runLePhareLSST.sh')
os.chdir(execDir)


# Plots - same as notebook, save as PDF: see LEPHARE python plot fonction.

## Load delight data
### First read a bunch of useful stuff from the parameter file.
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
bandNames = params['bandNames']
numBands, numCoefs = bandCoefAmplitudes.shape
fluxredshifts = np.loadtxt(lephare_testFileoutAbs)
fluxredshifts_train = np.loadtxt(lephare_trainFileoutAbs)
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn, refBandColumn = readColumnPositions(params, prefix='target_')
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
dir_seds = os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['templates_directory'])))
dir_filters = os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['bands_directory'])))
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
nt = len(sed_names)
f_mod = np.zeros((redshiftGrid.size, nt, len(params['bandNames'])))
for t, sed_name in enumerate(sed_names):
    f_mod[:, t, :] = np.loadtxt(os.path.join(dir_seds, sed_name + '_fluxredshiftmod.txt'))
    
### Load the PDF files
metrics = np.loadtxt(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['metricsFile']))))
metricscww = np.loadtxt(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['metricsFileTemp']))))
# Those of the indices of the true, mean, stdev, map, and map_std redshifts.
i_zt, i_zm, i_std_zm, i_zmap, i_std_zmap = 0, 1, 2, 3, 4
i_ze = i_zm
i_std_ze = i_std_zm
pdfs = np.loadtxt(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['redshiftpdfFile']))))
pdfs_cww = np.loadtxt(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['redshiftpdfFileTemp']))))
pdfatZ_cww = metricscww[:, 5] / pdfs_cww.max(axis=1)
pdfatZ = metrics[:, 5] / pdfs.max(axis=1)
nobj = pdfatZ.size
pdfs /= np.trapz(pdfs, x=redshiftGrid, axis=1)[:, None]
pdfs_cww /= np.trapz(pdfs_cww, x=redshiftGrid, axis=1)[:, None]

## Load the results from LEPHARE
fileIn=os.path.realpath(os.path.normpath(os.path.join('./', lephare_dir, 'zphot_long.out')))

### Number of the filter start at 0
selFilt=0   # filter for the selection in mag
uFilt=0
gFilt=1
rFilt=2
iFilt=3
zFilt=4
yFilt=5

### Array in redshift and mag, isolate extreme values
range_z = [0,1,2,3,6]
z_min, z_max = np.amin(range_z), np.amax(range_z)
range_mag = [15.,22.5,23.5,25,28]
mag_min, mag_max = np.amin(range_mag), np.amax(range_mag)

### READ THE INPUT FILE
### Read the first argument with the name of the photo-z output catalogue
catIn=open(fileIn,'r')
print("Name of the photo-z catalogue : ", fileIn)

### Loop over the filters
nbFilt=6
magst=""
idmagst=""
### create the string to read the mag
for i in range(nbFilt) :
    magst=magst+",mag"+str(i)
    idmagst=idmagst+","+str(i+20)
### create the string to read the error mag
for i in range(nbFilt) :
    magst=magst+",emag"+str(i)
    idmagst=idmagst+","+str(i+20+nbFilt)
### create the string to read the kcor
for i in range(nbFilt) :
    magst=magst+",kcor"+str(i)
    idmagst=idmagst+","+str(i+20+2*nbFilt)
### create the string to read the absolute mag
for i in range(nbFilt) :
    magst=magst+",absmag"+str(i)
    idmagst=idmagst+","+str(i+20+3*nbFilt)
### create the string to read the uncertainties on absolute mag
for i in range(nbFilt) :
    magst=magst+",eabsmag"+str(i)
    idmagst=idmagst+","+str(i+20+4*nbFilt)
### create the string for absolute mag filters
for i in range(nbFilt) :
    magst=magst+",mabsfilt"+str(i)
    idmagst=idmagst+","+str(i+20+5*nbFilt)

lastIdMagst=int(idmagst.split(',')[-1])
lastString=",{},{},{},{}".format(lastIdMagst+1, lastIdMagst+2, lastIdMagst+3, lastIdMagst+4)

### Extract from the ascii file
commandst = "Id,zp,zl68,zu68,zml,zmll68,zmlu68,chi,mod,law,ebv,zp2,chi2,mod2,ebv2,zq,chiq,modq,mods,chis"+magst+",scale,nbFilt,context,zs= np.loadtxt(catIn, dtype='float', usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"+idmagst+lastString+"), unpack=True )"

### transform the string into a command
print(commandst)
exec(commandst)

### CONDITION FOR SELECTION
### Mag use to select the sample
mag=eval("mag"+str(selFilt))
### General condition to select the galaxies in the expected z/mag range
cond = (zp>z_min) & (zp<z_max) & (mag>mag_min) & (mag<mag_max) 
### condition to select stars
condstar = (chis<chi)
### condition to select galaxies
condgal =  (~condstar)
### condition to select spectroscopic redshifts
condspec = (zs>0) & (zs<9)

### Load the PDZ from LEPHARE
pdzOut = os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir, 'pdzOut.pdz')))
pdzRange = np.arange(0.01, 3.02, 0.01)
pdzRange = np.concatenate((np.array([0.0]), pdzRange, np.array([3.01])))
pdzPhare = np.loadtxt(pdzOut)


figZsZp_ML = plot_and_stats(test_z, z_phot, test_data_mags[:, 3], title='Random Forrest Regressor')
figZsZp_lephare = plot_and_stats(zs, zp, mag3, title='LEPHARE++')
figZsZp_delightTF = plot_and_stats(metricscww[:, i_zt], metricscww[:, i_zmap], title='Delight TF')
figZsZp_delightGP = plot_and_stats(metrics[:, i_zt], metrics[:, i_zmap], title='Delight TF+GP')
figRandPdz = plot_random_pdz(redshiftGrid, pdfs_cww[k, :], label1='Delight TF',\
                             redshiftGrid, pdfs[k, :], label2='Delight TF+GP',\
                             pdzRange, pdzPhare[np.argmin(dummy), 1:], label3='LePhare++ TF',\
                             [], [], label4="No PDF for ML method")

figZsZp_ML.show()
figZsZp_lephare.show()
figZsZp_delightTF.show()
figZsZp_delightGP.show()
figRandPdz.show()