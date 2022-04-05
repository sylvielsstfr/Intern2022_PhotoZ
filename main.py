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
test_filename              = 'PhotoZML/data/test_dc2_validation_9816.hdf5'   # relative path
train_filename             = 'PhotoZML/data/test_dc2_training_9816.hdf5' # relative path
#test_fileout_delight       = 'test_gal_fluxredshifts.txt'     # file name only - will be created in the appropriate directory, until this is automated
#train_fileout_delight      = 'train_gal_fluxredshifts.txt'    # file name only - will be created in the appropriate directory, until this is automated
lephare_dir                = 'LEPHARELSST'                    # relative path - should be the directory where LSST.para and runLePhareLSST.sh are located and run.
test_fileout_lephare       = 'test_DC2_VALID_CAT_IN.in'       # file name only - will be created in the appropriate directory, until this is automated
train_fileout_lephare      = 'train_DC2_VALID_CAT_IN.in'      # file name only - will be created in the appropriate directory, until this is automated
mag_filt                   = 31.8
snr_filt                   = 5.0
vc, vl, alC, alL           = 0.1, 0.1, 1e3, 1e2
ellPriorSigma, zPriorSigma = 0.5, 0.2
nb_est                     = 50
reg_depth                  = 30


runML=True
runDelight=False
runLePhare=True
photoZML_name='regrnPhotoZ.txt'
pdfOutput_name='plotFileOut.pdf'

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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Useful Delight routines:
sys.path.append(os.path.realpath(os.path.normpath(os.path.join('./', delight_dir))))
from delight.io import *
from delight.utils import *
from utils import *

#####################################################
### CODE, DO NOT MODIFY UNLESS STRICTLY NECESSARY ###
#####################################################

# Generate the inputs
print("Generating the inputs")
t_start=time.time()
delight_absRunFile=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, delight_runFile)))
delight_absParamFile=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, delight_paramDir, delight_paramFile)))
execDir = os.path.realpath(os.getcwd())
os.chdir(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir))))
params = parseParamFile(delight_absParamFile, verbose=False, catFilesNeeded=False)
os.chdir(execDir)
test_fileout_delight = params['target_catFile']
train_fileout_delight = params['training_catFile']
testFile_absPath=os.path.realpath(os.path.normpath(os.path.join("./", test_filename)))
trainFile_absPath=os.path.realpath(os.path.normpath(os.path.join("./", train_filename)))
delight_testFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, test_fileout_delight)))
#print('Delight test file:\n\t{}'.format(delight_testFileoutAbs))
delight_trainFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, train_fileout_delight)))
lephare_testFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir, test_fileout_lephare)))
lephare_trainFileoutAbs=os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir, train_fileout_lephare)))

test_data_mags, test_err_mags, test_data_colors, test_data_colmag, test_z, dummy1, dummy2 =\
    create_all_inputs(testFile_absPath,\
                      mag=mag_filt,\
                      snr=snr_filt,\
                      returnErrors=True,\
                      fileout_lephare=lephare_testFileoutAbs,\
                      fileout_delight=delight_testFileoutAbs)

train_data_mags, train_err_mags, train_data_colors, train_data_colmag, train_z, dummy3, dummy4 =\
    create_all_inputs(trainFile_absPath,\
                      mag=mag_filt,\
                      snr=snr_filt,\
                      returnErrors=True,\
                      fileout_lephare=lephare_trainFileoutAbs,\
                      fileout_delight=delight_trainFileoutAbs)

# DEBUG #
#print(test_data_mags.shape, test_data_colors.shape, test_data_colmag.shape, test_z.shape, lephare_testFileoutAbs, delight_testFileoutAbs)
#print(train_data_mags.shape, train_data_colors.shape, train_data_colmag.shape, train_z.shape, lephare_trainFileoutAbs, delight_trainFileoutAbs)
# END DEBUG #

t_init=time.time()
print('Input creation duration: {} s'.format(t_init-t_start))

# Run the ML - call the appropriate python fonctions
## We need to set up an implementation of the scikit-learn RandomForestRegressor in an object called 'regrn'.
if runML:
    print('Beginning of ML estimation')
    regrn = RandomForestRegressor(n_estimators = nb_est, max_depth = reg_depth, max_features = 'auto')

    ## Train the regressor using the training data
    print('Training estimator')
    regrn.fit(train_data_mags, train_z.ravel())
    t_MLtrain=time.time()

    ## Apply the regressor to predict values for the test data
    print('Applying estimator')
    z_phot = regrn.predict(test_data_mags)
    
    np.savetxt(photoZML_name, np.column_stack((test_z, test_data_mags, z_phot)))
else:
    print('Skip ML')
t_MLpredict=time.time()
print('ML estimation done. Training time: {} s, Estimation time: {} s'.format(t_MLtrain-t_init, t_MLpredict-t_MLtrain))

# Run Delight - call the appropriate python fonctions
if runDelight:
    os.chdir(os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir))))
    print('DEBUG - current working directory:\n\t {}'.format(os.getcwd()))
    sys.path.append(os.getcwd())
    from run_full_delight_confFile import run_full_delight_confFile
    print('Running Delight - please be patient.')
    run_full_delight_confFile(delight_absParamFile)
    os.chdir(execDir)
else:
    print('Skip Delight')
t_endDelight=time.time()
print('Delight complete. Duration: {} s'.format(t_endDelight-t_MLpredict))

# Run LePhare - call the appropriate shell fonctions
if runLePhare:
    print('Running LEPHARE++')
    os.chdir(os.path.realpath(os.path.normpath(os.path.join("./", lephare_dir))))
    os.environ['LEPHAREDIR'] = os.path.realpath(os.path.normpath(os.path.join("..", 'LEPHARE')))
    os.environ['LEPHAREWORK'] = os.getcwd()
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['CAT_FILE_IN'] = lephare_testFileoutAbs
    subprocess.run('runLePhareLSST.sh')
    os.chdir(execDir)
else:
    print('Skip LEPHARE++')
t_endLePhare=time.time()
print('LEPHARE++ complete. Duration: {} s'.format(t_endLePhare-t_endDelight))

# Plots - same as notebook, save as PDF: see LEPHARE python plot fonction.

## Load delight data
### First read a bunch of useful stuff from the parameter file.
dir_seds = os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['templates_directory'])))
dir_filters = os.path.realpath(os.path.normpath(os.path.join("./", delight_dir, desc_dir, params['bands_directory'])))
params['bands_directory'] = dir_filters
params['templates_directory'] = dir_seds
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
bandNames = params['bandNames']
numBands, numCoefs = bandCoefAmplitudes.shape
fluxredshifts = np.loadtxt(delight_testFileoutAbs)
fluxredshifts_train = np.loadtxt(delight_trainFileoutAbs)
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn, refBandColumn = readColumnPositions(params, prefix='target_')
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
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
print('nobj = {}'.format(nobj))
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

test_z=np.loadtxt(photoZML_name)[:, 0]
z_phot=np.loadtxt(photoZML_name)[:, -1]
test_data_mags=np.loadtxt(photoZML_name)[:, 1:-1]
figZsZp_ML = plot_and_stats(test_z, z_phot, test_data_mags[:, 3], title='Random Forrest Regressor')
figZsZp_lephare = plot_and_stats(zs, zp, mag3, title='LEPHARE++')
figZsZp_delightTF = plot_and_stats(metricscww[:, i_zt], metricscww[:, i_zmap], test_data_mags[:, 3], title='Delight TF')
figZsZp_delightGP = plot_and_stats(metrics[:, i_zt], metrics[:, i_zmap], test_data_mags[:, 3], title='Delight TF+GP')
#figRandPdz = plot_random_pdz(redshiftGrid, pdfs_cww,\
#                             redshiftGrid, pdfs,\
#                             pdzRange, pdzPhare,\
#                             [], [],\
#                             label1='Delight TF',\
#                             label2='Delight TF+GP',\
#                             label3='LePhare++ TF',\
#                             label4="No PDF for ML method")

ncol = 4
figRandPdz, axsRandPdz = plt.subplots(5, ncol, figsize=(12, 12), sharex=True, sharey=False)
axsRandPdz = axsRandPdz.ravel()
sel = np.random.choice(nobj, axsRandPdz.size, replace=False)
lw = 2
for ik in range(axsRandPdz.size):
    k = sel[ik]
    zspec=fluxredshifts[k, redshiftColumn]
    #print(zspec, np.where(zs == zspec))
    dummy=np.array([])
    for z in zs:
        dummy=np.append(dummy, (z-zspec))
    dummy=np.absolute(dummy)
    galId=Id[np.argmin(dummy)]
    axsRandPdz[ik].plot(redshiftGrid, pdfs_cww[k,:], lw=lw, label='Delight TF')
    axsRandPdz[ik].plot(redshiftGrid, pdfs[k,:], lw=lw, label='Delight TF+GP')
    axsRandPdz[ik].plot(pdzRange, pdzPhare[k, 1:], lw=lw, label='LePhare++ TF')
    axsRandPdz[ik].plot([], [], lw=lw, label="No PDF for ML method")
    axsRandPdz[ik].axvline(zspec, c="k", lw=1, label='Spectro-Z')
    ymax = np.max(np.concatenate((pdfs_cww[k,:], pdfs[k,:], pdzPhare[np.argmin(dummy), 1:], [])))
    axsRandPdz[ik].set_ylim([0, ymax*1.2])
    axsRandPdz[ik].set_xlim([0, 3.1])
    axsRandPdz[ik].set_yticks([])
    axsRandPdz[ik].set_xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
for i in range(ncol):
    axsRandPdz[-i-1].set_xlabel('Redshift', fontsize=10)
axsRandPdz[0].legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.4))

figZsZp_ML.savefig('figZsZp_ML.png')
figZsZp_lephare.savefig('figZsZp_lephare.png')
figZsZp_delightTF.savefig('figZsZp_delightTF.png')
figZsZp_delightGP.savefig('figZsZp_delightGP.png')
figRandPdz.savefig('figRandPdz.png')

# store the figure in a PDF
# All the figures will be collected in a single pdf file 
pdfOut = PdfPages(pdfOutput_name)
figZsZp_ML.savefig(pdfOut,format='pdf')
figZsZp_lephare.savefig(pdfOut,format='pdf')
figZsZp_delightTF.savefig(pdfOut,format='pdf')
figZsZp_delightGP.savefig(pdfOut,format='pdf')
figRandPdz.savefig(pdfOut,format='pdf')
pdfOut.close()
