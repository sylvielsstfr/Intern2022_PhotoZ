
import sys
from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
sys.path.append('/Users/bl/Dropbox/repos/Delight/')
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
import scipy.stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import emcee
import corner

def approx_flux_likelihood_multiobj(
        f_obs,  # no, nf
        f_obs_var,  # no, nf
        f_mod,  # no, nt, nf
        ell_hat,  # 1
        ell_var,  # 1
        returnChi2=False,
        normalized=True):

    assert len(f_obs.shape) == 2
    assert len(f_obs_var.shape) == 2
    assert len(f_mod.shape) == 3
    no, nt, nf = f_mod.shape
    f_obs_r = f_obs[:, None, :]
    var = f_obs_var[:, None, :]
    invvar = np.where(f_obs_r/var < 1e-6, 0.0, var**-1.0)  # nz * nt * nf
    FOT = np.sum(f_mod * f_obs_r * invvar, axis=2)\
        + ell_hat / ell_var  # no * nt
    FTT = np.sum(f_mod**2 * invvar, axis=2)\
        + 1. / ell_var  # no * nt
    FOO = np.sum(f_obs_r**2 * invvar, axis=2)\
        + ell_hat**2 / ell_var  # no * nt
    sigma_det = np.prod(var, axis=2)
    chi2 = FOO - FOT**2.0 / FTT  # no * nt
    denom = np.sqrt(FTT)
    if normalized:
        denom *= np.sqrt(sigma_det * (2*np.pi)**nf * ell_var)
    like = np.exp(-0.5*chi2) / denom  # no * nt
    if returnChi2:
        return chi2
    else:
        return like


comm = MPI.COMM_WORLD
threadNum = comm.Get_rank()
numThreads = comm.Get_size()

# Parse parameters file
if len(sys.argv) < 2:
    raise Exception('Please provide a parameter file')
paramFileName = sys.argv[1]
params = parseParamFile(paramFileName, verbose=False)

DL = approx_DL()
redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
numZ = redshiftGrid.size

# Locate which columns of the catalog correspond to which bands.
bandIndices, bandNames, bandColumns, bandVarColumns, redshiftColumn,\
    refBandColumn = readColumnPositions(params, prefix="training_")

dir_seds = params['templates_directory']
dir_filters = params['bands_directory']
lambdaRef = params['lambdaRef']
sed_names = params['templates_names']
numBands = bandIndices.size
nt = len(sed_names)
f_mod = np.zeros((numZ, nt, len(params['bandNames'])))
for t, sed_name in enumerate(sed_names):
    f_mod[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +
                                '_fluxredshiftmod.txt')

numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
print('Number of Training Objects', numObjectsTraining)
numMetrics = 7 + len(params['confidenceLevels'])
allFluxes = np.zeros((numObjectsTraining, numBands))
allFluxesVar = np.zeros((numObjectsTraining, numBands))
redshifts = np.zeros((numObjectsTraining, 1))
fmod_atZ = np.zeros((numObjectsTraining, nt, numBands))

# Now loop over training set to compute likelihood function
loc = - 1
trainingDataIter = getDataFromFile(params, 0, numObjectsTraining,
                                   prefix="training_", getXY=False)
for z, ell, bands, fluxes, fluxesVar, bCV, fCV, fvCV in trainingDataIter:
    loc += 1
    allFluxes[loc, :] = fluxes
    allFluxesVar[loc, :] = fluxesVar
    redshifts[loc, 0] = z
    for t, sed_name in enumerate(sed_names):
        for ib, b in enumerate(bands):
            fmod_atZ[loc, t, ib] = ell * np.interp(z, redshiftGrid,
                                                   f_mod[:, t, b])

def scalefree_flux_lnlikelihood_multiobj(
        f_obs,  # no, ..., nf
        f_obs_var,  # no, ..., nf
        f_mod,  # ..., nf
        normalized=True):

    assert len(f_obs.shape) == len(f_mod.shape)
    assert len(f_obs_var.shape) == len(f_mod.shape)
    assert len(f_mod.shape) >= 2
    nf = f_mod.shape[-1]
    assert f_obs.shape[-1] == nf
    assert f_obs_var.shape[-1] == nf
    # nz * nt * nf
    invvar = np.where(f_obs/f_obs_var < 1e-6, 0.0, f_obs_var**-1.0)
    FOT = np.sum(f_mod * f_obs * invvar, axis=-1)  # no * nt
    FTT = np.sum(f_mod**2 * invvar, axis=-1)  # no * nt
    FOO = np.sum(f_obs**2 * invvar, axis=-1)  # no * nt
    logsigma_det = np.sum(np.log(f_obs_var), axis=-1)
    chi2 = FOO - FOT**2.0 / FTT  # no * nt
    logdenom = 0.5 * np.log(FTT)
    ellML = FOT / FTT
    if normalized:
        logdenom += 0.5* logsigma_det
    lnlike = -0.5*chi2 - logdenom  # no * nt
    return lnlike, ellML

def lnprob(params, nt, allFluxes, allFluxesVar, fmod_atZ, pmin, pmax):
    if np.any(params > pmax) or np.any(params < pmin):
            return - np.inf
    alphas = params[0:nt]
    betas = params[nt:2*nt][None, :]
    lnlike_grid = scalefree_flux_lnlikelihood_multiobj(
        allFluxes[:, None, :], allFluxesVar[:, None, :], fmod_atZ)  # no, nt
    p_t = dirichlet(alphas)
    p_z = redshifts * np.exp(-0.5 * redshifts**2 / betas) / betas  # p(z|t)
    p_z_t = p_z * p_t  # no, nt
    lnlike_lt = logsumexp(lnlike_grid + np.log(p_z_t), axis=1)
    return - np.sum(lnlike_lt)


def plot_params(params):
    alphas = params[0:nt]
    betas = params[nt:2*nt]
    fig, axs = plt.subplots(4, 4, figsize=(16, 8))
    axs = axs.ravel()
    alpha0 = np.sum(alphas)
    dirsamples = dirichlet(alphas, 1000)
    for i in range(nt):
        mean = alphas[i]/alpha0
        std = np.sqrt(alphas[i] * (alpha0-alphas[i]) / alpha0**2 / (alpha0+1))
        axs[i].axvspan(mean-std, mean+std, color='gray', alpha=0.5)
        axs[i].axvline(mean, c='k', lw=2)
        axs[i].axvline(1/nt, c='k', lw=1, ls='dashed')
        axs[i].set_title('alpha = '+str(alphas[i]))
        axs[i].set_xlim([0, 1])
        axs[i].hist(dirsamples[:, i], 50, color="k", histtype="step")
    for i in range(nt):
        pz = redshiftGrid*np.exp(-0.5*redshiftGrid**2/betas[i])/betas[i]
        axs[nt+i].plot(redshiftGrid, pz, c='k', lw=2)
        axs[nt+i].axvline(betas[i], lw=2, c='k', ls='dashed')
        axs[nt+i].set_title('beta = '+str(betas[i]))
    fig.tight_layout()
    return fig


pmin = np.concatenate((np.repeat(0., nt), np.repeat(0., nt)))
pmax = np.concatenate((np.repeat(200., nt), np.repeat(redshiftGrid[-1], nt)))
ndim, nwalkers = 2*nt, 1000
p0 = [pmin + (pmax-pmin)*np.random.uniform(0, 1, size=ndim)
      for i in range(nwalkers)]

bounds = [[pmin[i], pmax[i]] for i in range(ndim)]

vals = np.zeros((len(p0), ))
for i in range(len(p0)):
    vals[i] = lnprob(p0[i], nt, allFluxes, allFluxesVar, fmod_atZ, pmin, pmax)

loc = np.argmin(vals)
print('Minimum value:', vals[loc])

res = minimize(lnprob, p0[loc], args=(nt, allFluxes, allFluxesVar, fmod_atZ, pmin, pmax), bounds=bounds)

print(res)

params_mean = res.x

alphas = params_mean[0:nt]
betas = params_mean[nt:2*nt]

alpha0 = np.sum(alphas)
print("p_t:", ' '.join(['%.2g' % x for x in alphas / alpha0]))
print("p_z_t:", ' '.join(['%.2g' % x for x in betas]))
print("p_t err:", ' '.join(['%.2g' % x
      for x in np.sqrt(alphas*(alpha0-alphas)/alpha0**2/(alpha0+1))]))
