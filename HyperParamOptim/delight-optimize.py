import sys
#from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils_cy import approx_flux_likelihood_cy
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import emcee
#import corner
import logging
import coloredlogs
import os

# Plot formatting
plt.rcParams["figure.figsize"] = (8,6)
#plt.rcParams["figure.tight_layout"] = True
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['xtick.labelsize']= 'xx-large'
plt.rcParams['ytick.labelsize']= 'xx-large'

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s %(name)s[%(process)d] %(levelname)s %(message)s')

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
params = parseParamFile(sys.argv[1], verbose=False)

# Read filter coefficients, compute normalization of filters
bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms\
    = readBandCoefficients(params)
numBands = bandCoefAmplitudes.shape[0]

redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
f_mod = readSEDs(params)
DL = approx_DL()

dir_seds = params['templates_directory']
dir_filters = params['bands_directory']
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
f_mod_grid = np.zeros((redshiftGrid.size, len(sed_names),
                       len(params['bandNames'])))
for t, sed_name in enumerate(sed_names):
    f_mod_grid[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +
                                     '_fluxredshiftmod.txt')

numZbins = redshiftDistGrid.size - 1
numZ = redshiftGrid.size
numConfLevels = len(params['confidenceLevels'])
numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
print('Number of Training Objects', numObjectsTraining)
print('Number of Target Objects', numObjectsTarget)

ellPriorSigmaList = np.logspace( 0, 6, 7 )
lenSigList = len(ellPriorSigmaList)
ellInd = -1

for ellPriorSigma in [1000]: #ellPriorSigmaList:
    print("ellPriorSigma = {}".format(ellPriorSigma))
    ellInd += 1
    alpha_C = 1e3
    alpha_L = 1e2
    V_C, V_L = 1.0, 1.0
    allVc = []
    allZ = []
    allMargLike = []
    allAlphaC = []
    allAlphaL = []
    allVl = []
    gp = PhotozGP(
        f_mod,
        bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
        params['lines_pos'], params['lines_width'],
        V_C, V_L, alpha_C, alpha_L,
        redshiftGridGP, use_interpolators=True)
    print("Initialisation du GP pour var_C = {}, alpha_C = {}".format(gp.kernel.var_C, gp.kernel.alpha_C))

    for extraFracFluxError in [1e-2]:
        redshifts = np.zeros((numObjectsTraining, ))
        bestTypes = np.zeros((numObjectsTraining, ), dtype=int)
        ellMLs = np.zeros((numObjectsTraining, ))
        model_mean = np.zeros((numZ, numObjectsTraining, numBands))
        model_covar = np.zeros((numZ, numObjectsTraining, numBands))
        # params['training_extraFracFluxError'] = extraFracFluxError
        params['target_extraFracFluxError'] = extraFracFluxError

        loc = -1
        trainingDataIter = getDataFromFile(params, 0, numObjectsTraining, prefix="training_", getXY=True)
        for z, normedRefFlux, bands, fluxes, fluxesVar, bCV, fCV, fvCV, X, Y, Yvar in trainingDataIter:
            loc += 1
            redshifts[loc] = z
            # print( "z = {},\nbands = {},\nfluxes = {}".format(z, bands, fluxes) )
            
            themod = np.zeros((1, f_mod_grid.shape[1], bands.size))
            for it in range(f_mod_grid.shape[1]):
                for ib, band in enumerate(bands):
                    themod[0, it, ib] = np.interp(z, redshiftGrid, f_mod_grid[:, it, band])
            
            chi2_grid, theellMLs = scalefree_flux_likelihood(fluxes, fluxesVar, themod, returnChi2=True)
            
            bestTypes[loc] = np.argmin(chi2_grid)
            #distribué uniformément?
            ellMLs[loc] = theellMLs[0, bestTypes[loc]]
            #autour de 1e6 car facteur dans la génération des flux
            
            X[:, 2] = ellMLs[loc]

            for V_C in np.linspace( 0.1, 1e4, 100 ):
                #print("Création du GP pour z = {}, V_C = {}".format(z, V_C))
                #gp.mean_fct = Photoz_linear_sed_basis(f_mod)
                #gp.kernel = Photoz_kernel(bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGrid=redshiftGridGP, use_interpolators=True)
                #gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGridGP, use_interpolators=True)
                gp.kernel.use_interpolators=False
                gp.kernel.var_C = V_C
                #gp.kernel.update_kernelparts(X)
                gp.setData(X, Y, Yvar, bestTypes[loc])
                marginalLikelihood = gp.margLike()
                #print("GP créé. Marginal Likelihood = {}".format(marginalLikelihood))
                allZ.append(z)
                allMargLike.append(marginalLikelihood)
                allVc.append(V_C)

            # ~ for V_L in np.linspace( 0.1, 1e4, 100 ):
                # ~ #print("Création du GP pour z = {}, V_L = {}".format(z, V_L))
                # ~ #gp.mean_fct = Photoz_linear_sed_basis(f_mod)
                # ~ #gp.kernel = Photoz_kernel(bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGrid=redshiftGridGP, use_interpolators=True)
                # ~ #gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGridGP, use_interpolators=True)
                # ~ gp.kernel.use_interpolators=False
                # ~ gp.kernel.var_L = V_L
                # ~ gp.kernel.update_kernelparts(X)
                # ~ gp.setData(X, Y, Yvar, bestTypes[loc])
                # ~ marginalLikelihood = gp.margLike()
                # ~ #print("GP créé. Marginal Likelihood = {}".format(marginalLikelihood))
                # ~ allZ.append(z)
                # ~ allMargLike.append(marginalLikelihood)
                # ~ allVl.append(V_L)

            # ~ for alpha_C in np.linspace( 0.1, 1e4, 100 ):
                # ~ #print("Création du GP pour z = {}, alpha_C = {}".format(z, alpha_C))
                # ~ #gp.mean_fct = Photoz_linear_sed_basis(f_mod)
                # ~ #gp.kernel = Photoz_kernel(bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGrid=redshiftGridGP, use_interpolators=True)
                # ~ #gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGridGP, use_interpolators=True)
                # ~ gp.kernel.use_interpolators=False
                # ~ gp.kernel.alpha_C = alpha_C
                # ~ #gp.kernel.update_kernelparts(X)
                # ~ gp.setData(X, Y, Yvar, bestTypes[loc])
                # ~ marginalLikelihood = gp.margLike()
                # ~ #print("GP créé. Marginal Likelihood = {}".format(marginalLikelihood))
                # ~ allZ.append(z)
                # ~ allMargLike.append(marginalLikelihood)
                # ~ allAlphaC.append(alpha_C)

            # ~ for alpha_L in np.linspace( 0.1, 1e4, 100 ):
                # ~ #print("Création du GP pour z = {}, alpha_L = {}".format(z, alpha_L))
                # ~ #gp.mean_fct = Photoz_linear_sed_basis(f_mod)
                # ~ #gp.kernel = Photoz_kernel(bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGrid=redshiftGridGP, use_interpolators=True)
                # ~ #gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, params['lines_pos'], params['lines_width'], V_C, V_L, alpha_C, alpha_L, redshiftGridGP, use_interpolators=True)
                # ~ gp.kernel.use_interpolators=False
                # ~ gp.kernel.alpha_L = alpha_L
                # ~ gp.kernel.update_kernelparts(X)
                # ~ gp.setData(X, Y, Yvar, bestTypes[loc])
                # ~ marginalLikelihood = gp.margLike()
                # ~ #print("GP créé. Marginal Likelihood = {}".format(marginalLikelihood))
                # ~ allZ.append(z)
                # ~ allMargLike.append(marginalLikelihood)
                # ~ allAlphaL.append(alpha_L)
                
        ## Plot for this iteration on ellPriorSigma:
        print("Création des graphes pour ellPriorSigma = {}".format(ellPriorSigma))
        cmap = "coolwarm_r"
        vmin = 0.0
        alpha = 0.9
        s = 5
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        # ~ vs0 = axs[0].scatter(allZ, allMargLike, s=s, c=allAlphaL, cmap=cmap, linewidth=0, alpha=alpha)
        # ~ vs1 = axs[1].scatter(allAlphaL, allMargLike, s=s, c=allZ, cmap=cmap, linewidth=0, alpha=alpha)
        # ~ clb0 = plt.colorbar(vs0, ax=axs[0], location = 'right')
        # ~ clb1 = plt.colorbar(vs1, ax=axs[1], location = 'right')
        # ~ clb0.set_label('$alpha_L$')
        # ~ clb1.set_label('MAP photo-$z$')
        # ~ axs[0].set_xlim([0, np.max(allZ)])
        # ~ axs[0].set_ylim([np.min(allMargLike), np.max(allMargLike)])
        # ~ axs[1].set_xlim([np.min(allAlphaL), np.max(allAlphaL)])
        # ~ axs[1].set_ylim([np.min(allMargLike), np.max(allMargLike)])
        # ~ axs[0].set_xlabel('MAP photo-$z$')
        # ~ axs[0].set_ylabel('GP Marginal likelihood')
        # ~ axs[1].set_xlabel('$alpha_L$')
        # ~ axs[1].set_ylabel('GP Marginal likelihood')

        # ~ axs[0].set_title('$\ell_{prior \sigma} = $'+'{}'.format(ellPriorSigma))
        # ~ axs[1].set_title('$\ell_{prior \sigma} = $'+'{}'.format(ellPriorSigma))
        
        # ~ fig.savefig( "alphaL_MarginalLikelihood_z_ellSigma-{}.png".format(ellPriorSigma) )
        
        
        vs0 = axs[0].hist2d(allZ, allMargLike, bins=[100, 100], range=[[np.min(allZ), np.max(allZ)], [-10, 100]], density=True, cmap="Blues", alpha=alpha)
        vs1 = axs[1].hist2d(allVc, allMargLike, bins=[50, 100], range=[[np.min(allVc), np.max(allVc)], [-10, 100]], density=True, cmap="Reds", alpha=alpha)
        # ~ clb0 = plt.colorbar(vs0, ax=axs[0], location = 'right')
        # ~ clb1 = plt.colorbar(vs1, ax=axs[1], location = 'right')
        # ~ clb0.set_label('$alpha_C$')
        # ~ clb1.set_label('MAP photo-$z$')
        # ~ axs[0].set_xlim([0, np.max(allZ)])
        # ~ axs[0].set_ylim([np.min(allMargLike), np.max(allMargLike)])
        # ~ axs[1].set_xlim([np.min(allAlphaC), np.max(allAlphaC)])
        # ~ axs[1].set_ylim([np.min(allMargLike), np.max(allMargLike)])
        axs[0].set_xlabel('MAP photo-$z$')
        axs[0].set_ylabel('GP Marginal likelihood')
        axs[1].set_xlabel('$V_C$')
        axs[1].set_ylabel('GP Marginal likelihood')

        axs[0].set_title('$\ell_{prior \sigma} = $'+'{}'.format(ellPriorSigma))
        axs[1].set_title('$\ell_{prior \sigma} = $'+'{}'.format(ellPriorSigma))
        
        fig.savefig( "VC_MargLike_z_v3.1_DESC_ellSigma-{}.png".format(ellPriorSigma) )




#             for redshiftSigma in [0.1, 1.0]:
# 
#                 loc = - 1
#                 targetDataIter = getDataFromFile(params, 0, numObjectsTarget,
#                                                  prefix="target_", getXY=False)
# 
#                 bias_zmap = np.zeros((redshiftDistGrid.size, ))
#                 bias_zmean = np.zeros((redshiftDistGrid.size, ))
#                 confFractions = np.zeros((numConfLevels,
#                                           redshiftDistGrid.size))
#                 binnobj = np.zeros((redshiftDistGrid.size, ))
#                 bias_nz = np.zeros((redshiftDistGrid.size, ))
#                 stackedPdfs = np.zeros((redshiftGrid.size,
#                                         redshiftDistGrid.size))
#                 cis = np.zeros((numObjectsTarget, ))
#                 zmeanBinLocs = np.zeros((numObjectsTarget, ), dtype=int)
#                 for z, normedRefFlux, bands, fluxes, fluxesVar, bCV, fCV, fvCV\
#                         in targetDataIter:
#                     loc += 1
#                     like_grid = np.zeros((model_mean.shape[0],
#                                           model_mean.shape[1]))
#                     ell_hat_z = normedRefFlux * 4 * np.pi\
#                         * params['fluxLuminosityNorm'] \
#                         * (DL(redshiftGrid)**2. * (1+redshiftGrid))
#                     ell_hat_z[:] = 1
#                     approx_flux_likelihood_cy(
#                         like_grid,
#                         model_mean.shape[0], model_mean.shape[1], bands.size,
#                         fluxes, fluxesVar,
#                         model_mean[:, :, bands],
#                         V_C*model_covar[:, :, bands],
#                         ell_hat_z, (ell_hat_z*ellPriorSigma)**2)
#                     like_grid *= np.exp(-0.5*((redshiftGrid[:, None] -
#                                                redshifts[None, :]) /
#                                               redshiftSigma)**2)
#                     pdf = like_grid.sum(axis=1)
#                     if pdf.sum() == 0:
#                         print("NULL PDF with galaxy", loc)
#                     if pdf.sum() > 0:
#                         metrics\
#                             = computeMetrics(z, redshiftGrid, pdf,
#                                              params['confidenceLevels'])
#                         ztrue, zmean, zstdzmean, zmap, zstdzmean,\
#                             pdfAtZ, cumPdfAtZ = metrics[0:7]
#                         confidencelevels = metrics[7:]
#                         zmeanBinLoc = -1
#                         for i in range(numZbins):
#                             if zmean >= redshiftDistGrid[i]\
#                                     and zmean < redshiftDistGrid[i+1]:
#                                 zmeanBinLoc = i
#                                 bias_zmap[i] += ztrue - zmap
#                                 bias_zmean[i] += ztrue - zmean
#                                 binnobj[i] += 1
#                                 bias_nz[i] += ztrue
#                         zmeanBinLocs[loc] = zmeanBinLoc
#                         for i in range(numConfLevels):
#                             if pdfAtZ >= confidencelevels[i]:
#                                 confFractions[i, zmeanBinLoc] += 1
#                         stackedPdfs[:, zmeanBinLoc]\
#                             += pdf / numObjectsTraining
#                         ind = pdf >= pdfAtZ
#                         pdf /= np.trapz(pdf, x=redshiftGrid)
#                         cis[loc] = np.trapz(pdf[ind], x=redshiftGrid[ind])
# 
#                 confFractions /= binnobj[None, :]
#                 bias_nz /= binnobj
#                 for i in range(numZbins):
#                     if stackedPdfs[:, i].sum():
#                         bias_nz[i] -= np.average(redshiftGrid,
#                                                  weights=stackedPdfs[:, i])
#                 ind = binnobj > 0
#                 bias_zmap /= binnobj
#                 bias_zmean /= binnobj
#                 print("")
#                 print(' =======================================')
#                 print("  ellSTD", ellPriorSigma,
#                       "fluxError", extraFracFluxError,
#                       "V_C", V_C, "zSTD", redshiftSigma)
#                 cis_pdf, e = np.histogram(cis, 50, range=[0, 1])
#                 cis_pdfcum = np.cumsum(cis_pdf) / np.sum(cis_pdf)
#                 print("-------------------------------->>  %.3g"
#                       % (np.max(np.abs(np.abs(e[1:] - cis_pdfcum)))))
#                 print(">>", end="")
#                 for i in range(numZbins):
#                     ind2 = zmeanBinLocs == i
#                     if ind2.sum() > 2:
#                         cis_pdf, e = np.histogram(cis[ind2], 50, range=[0, 1])
#                         cis_pdfcum = np.cumsum(cis_pdf) / np.sum(cis_pdf)
#                         print("  %.3g" % (np.max(np.abs(e[1:] - cis_pdfcum))),
#                               end=" ")


                # print("")
                # print(' >>>> mean z bias %.3g'
                # % np.abs(bias_zmean[ind]).mean(),
                # 'mean N(z) bias %.3g' % np.abs(bias_nz[ind]).mean(), ' <<<<')
                # print(' >>>> max z bias %.3g'
                # % np.abs(bias_zmean[ind]).max(),
                # 'max N(z) bias %.3g' % np.abs(bias_nz[ind]).max(), ' <<<<')
                # print(' > bias_zmap : ',
                # '  '.join(['%.3g' % x for x in bias_zmap]))
                # print(' > z bias : ',
                # '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in bias_zmean]))
                # print(' > nzbias : ',
                # '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in bias_nz]))
                # print(' --------------------------------')
                # for i in range(numConfLevels):
                #     print(' >', params['confidenceLevels'][i],
                # ' :: ', '  '.join([('%.3g' % x) if np.isfinite(x)
                # else '.' for x in confFractions[i, :]]))
                # print(' =======================================')
