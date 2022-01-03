
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
import matplotlib.pyplot as plt
import emcee
import corner


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


def lnprob(params, nt, allFluxes, allFluxesVar, fmod_atZ, pmin, pmax):
    if np.any(params > pmax) or np.any(params < pmin):
            return - np.inf
    alphas = params[0:nt]
    betas = params[nt:2*nt][None, :]
    sigma_ell = 1e16
    like_grid = approx_flux_likelihood_multiobj(
        allFluxes, allFluxesVar, fmod_atZ, 1, sigma_ell**2.)  # no, nt
    p_t = dirichlet(alphas)
    p_z = redshifts * np.exp(-0.5 * redshifts**2 / betas) / betas  # p(z|t)
    p_z_t = p_z * p_t  # no, nt
    like_lt = (like_grid * p_z_t).sum(axis=1)
    eps = 1e-305
    ind = like_lt > eps
    theprob = np.log(like_lt[ind]).sum()
    return theprob


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

ndim, nwalkers = 2*nt, 100
p0 = [pmin + (pmax-pmin)*np.random.uniform(0, 1, size=ndim)
      for i in range(nwalkers)]

for i in range(10):
    print(lnprob(p0[i], nt, allFluxes, allFluxesVar, fmod_atZ, pmin, pmax))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                threads=4,
                                args=[nt, allFluxes, allFluxesVar, fmod_atZ,
                                      pmin, pmax])
pos, prob, state = sampler.run_mcmc(p0, 2)
sampler.reset()
sampler.run_mcmc(pos, 10)
print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))
samples = sampler.chain.reshape((-1, ndim))
lnprob = sampler.lnprobability.reshape((-1, 1))

params_mean = samples.mean(axis=0)
params_std = samples.std(axis=0)

fig, axs = plt.subplots(4, 5, figsize=(16, 8))
axs = axs.ravel()
for i in range(ndim):
    axs[i].hist(samples[:, i], 50, color="k", histtype="step")
    axs[i].axvspan(params_mean[i]-params_std[i],
                   params_mean[i]+params_std[i], color='gray', alpha=0.5)
    axs[i].axvline(params_mean[i], c='k', lw=2)
fig.tight_layout()
fig.savefig('prior_parameters.pdf')

fig = plot_params(params_mean)
fig.savefig('prior_meanparameters.pdf')

print("params_mean", params_mean)
print("params_std", params_std)

alphas = params_mean[0:nt]
betas = params_mean[nt:2*nt]

alpha0 = np.sum(alphas)
print("p_t:", ' '.join(['%.2g' % x for x in alphas / alpha0]))
print("p_t err:", ' '.join(['%.2g' % x
      for x in np.sqrt(alphas*(alpha0-alphas)/alpha0**2/(alpha0+1))]))
print("p_z_t:", ' '.join(['%.2g' % x for x in betas]))

fig = corner.corner(samples)
fig.savefig("triangle.pdf")
