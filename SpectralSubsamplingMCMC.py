"""
Spectral Subsampling MCMC for stationary time series. 

Runs a data subsampling block pseudo-marginal algorithm to sample from a posterior density based on a spectral likelihood (the Whittle likelihood),
see the submission "Spectral Subsampling MCMC for stationary time series."

The code constructs grouped Taylor series control variates and coreset control variates (coreset code based on the bc-package: https://github.com/trevorcampbell/bayesian-coresets).
The coreset control variates are implemented using the multiprocessing module which can sometimes be problematic, especially on Windows machines.

The code carries out posterior sampling and gives several output files and figures.

The user needs to specify a data_set_name and other outputs, see the code. Many assertions are done on these inputs, so that the resulting examples correspond to those in the submission.
It is straightforward to modify the code for running on a new dataset, just create a new dataset and specify the settings for it. 
"""
from __future__ import division
from operator import itemgetter
import autograd.numpy as np
from autograd import grad, hessian, jacobian
import progressbar
import numpy.random as npr
import scipy.stats as sps
import autograd.scipy.stats as sps_autograd
import autograd.scipy.special as sc_autograd
from autograd.numpy.fft import fft
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime
import time
import seaborn as sns
import sys
import os
import copy
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize, Bounds
import multiprocessing as mp
import bayesiancoresets as bc
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from numdifftools import Hessian as Hess_finite_diff
import pickle


def rds2dict(filename):
    '''
    Creates a Python Dictionary with Numpy Array elements

    from an R .RDS file
    '''
    pandas2ri.activate()

    readRDS = robjects.r['readRDS']

    rds = readRDS(filename)

    data = {}

    for i in range(len(rds.names)):

        data[rds.names[i]] = np.array(rds[i])

    return data


def nparray2rmatrix(x):
    """
    Converts a nparray to an r matrix.
    """
    try:
        nr, nc = x.shape
    except ValueError:
        nr = x.shape[0]
        nc = 1
    xvec = robjects.FloatVector(x.transpose().reshape((x.size)))
    xr = robjects.r.matrix(xvec, nrow=nr, ncol=nc)
    return xr


wrkDir = sys.path[0]
path = wrkDir + '/Results/Runs/'

# For some insane reason, we can't pass these as arguments to the function if we want to avoid pickling error with multiprocessing.
M, projection_dim, singleTheta = 200, 500, True
use_Hessians = True  # False # Hessians for optimizations performed by scipy

sns.set(style="ticks")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 22

plt.rc('axes', labelsize=22)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True

gtol = 1e-4
max_iter_optim = 500

# ARFIMA reparametrized
np.random.seed(123)
seed = 1
np.random.seed(seed)  # This seed gives hard data if frequencies are grouped.


def simulate_truncated_normal(mode, CovMatrix, FI_term, TFI_term):
    def subfunc():
        while True:
            prop = np.random.multivariate_normal(mode, CovMatrix)
            if FI_term:
                if TFI_term:
                    # includes lamda_, sigma2 and d (transformed to unrestricted.)
                    if not any(abs(prop[:-3]) > 1):
                        break

                else:
                    # includes sigma2 and d (transformed to unrestricted.)
                    if not any(abs(prop[:-2]) > 1):
                        break
            else:
                # Includes only sigma 2 (transformed to unrestricted)
                if not any(abs(prop[:-1]) > 1):
                    break
        return prop
    return subfunc


def log_density(ind, params, I_pg, p, q):
    # Returns the whittle log-density for ind
    return whittle_likelihood(ind, params, I_pg, p, q, FI_term, TFI_term, SV_model, w=[],
                              use_w=False, return_sum=False)


def log_density_groups(y, I, params, I_pg, p, q):
    # Log-density summed within a group for all groups in I.
    # u is a list of length G, where each element in the list contains an array of size groupSize (with indices)
    return np.array([log_density_group(y, I[ind], params, I_pg, p, q) for ind in range(len(I))])


def log_density_group(y, I_single, params, I_pg, p, q):
    # Log-density summed within a single group indicated by u_single.
    return np.sum(log_density(y[I_single], params, I_pg, p, q))


# Example model: The Whittle likelihood. Has additional functions for reparameterization to induce stationarity
def reparam(params, MA=False):
    """
    Transforms params to induce stationarity/invertability.
    Takes as input parameters in the partial auto-correlation parameterization and returns parameters that are on the ordinary parameterization.
    """
    newparams = np.array(params, copy=True)
    tmp = np.array(params, copy=True)

    for j in range(1, len(params)):
        if not MA:
            tmp_new = tmp[:j] - \
                np.array([(newparams[j]*newparams[j-k-1]) for k in range(j)])
        else:
            tmp_new = tmp[:j] + \
                np.array([(newparams[j]*newparams[j-k-1]) for k in range(j)])

        tmp = np.hstack([tmp_new, newparams[j:]])
        newparams = np.hstack((tmp[:j], newparams[j:]))

    return newparams


def p_gram(x):  # Construct Periodogram
    id = int(np.floor((len(x)-1)/2))
    return np.square(np.abs(x[0:(id+1)]))/(2 * np.pi * len(x))


def f_ARTFIMA(id, phi, theta, var, d, lambda_, var2, n):
    # General spectral density that includes all the ones in the paper as special cases.
    # Note: var2 only exist if there is an ARTFIMA process on an SV model. Otherwise it is zero. This is for the Stochastic volatility ARTFIMA - see paper for details
    omega = (2*np.pi*id/n)

    FI_term = np.abs(1 - np.exp(-(lambda_ + 1j*omega))
                     )**(-2*d)  # Fractional Integration Term

    if phi.any():
        log_arg_phi = np.outer(-1j*omega, np.arange(1, len(phi)+1))
        vv1 = 1/(1 - np.sum(phi * np.exp(log_arg_phi), 1))
    else:
        vv1 = 1

    if theta.any():
        log_arg_theta = np.outer(-1j*omega, np.arange(1, len(theta)+1))
        vv2 = (1 + np.sum(theta * np.exp(log_arg_theta), 1))
    else:
        vv2 = 1

    f = FI_term * (var/(2*np.pi)) * (np.real(vv1)**2 + np.imag(vv1)**2) \
                * (np.real(vv2)**2 + np.imag(vv2)**2) + (var2/(2*np.pi))
    return f


def whittle_likelihood(ind, params, I, p, q, FI_term, TFI_term, SV_model,  w=[],
                       use_w=False, return_sum=False):
    # We parameterize the prior in terms of partial autocorrelations and for the rest of the parameters in the unrestricted space.
    # ind : list of indices used to evaluate the whittle likelihood
    # params : parameter vector for model
    # prm_info: list of information telling the function how the parameter vector is
    #               decomposed, first element is #AR terms, second is #MA terms
    # I : COMPLETE periodogram for all Fourier frequencies
    # w : weights, w[j] corresponds to ind[j]
    # return_sum : returns array of log-likelihood values if false, otherise
    #               returns the sum of these values (complete log-likelihood)

    if FI_term:
        var = np.exp(params[-2])
        if TFI_term:
            # Tempered fractional difference
            # ARTFIMA model - d is unrestricted, may take any values (except integers, occurs only of sets with measure zero)
            d = params[-1]
            lambda_ = np.exp(params[-3])
            if SV_model:
                var2 = np.exp(params[-4])
            else:
                var2 = 0
        else:
            # ARFIMA model, restrict d to [-0.5, 0.5]
            d = 0.5*np.tanh(params[-1])  # 0.5*sc_autograd.expit(2*params[-1])
            lambda_ = 0
            var2 = 0
    else:
        var = np.exp(params[-1])
        d = 0  # ARMA process
        lambda_ = 0  # ARMA process
        var2 = 0

    if q > 0:
        phi = np.array(reparam(params[:q]))
    else:
        phi = np.array([])

    if p > 0:
        # FI_term
        if FI_term:
            if TFI_term:
                if SV_model:
                    # Also estimating var2, i.e. in total 4 parameters extra to estimate (beside the partial autocorrelations)
                    last_MA_term = -4
                else:
                    # Estimating lambda_, d and var
                    last_MA_term = -3
            else:
                # Estimating d and var only
                last_MA_term = -2
        else:
            last_MA_term = -1

        theta = np.array(reparam(params[q: last_MA_term], MA=True))
    else:
        theta = np.array([])

    if not use_w:
        w = np.ones(len(ind))
    fj = f_ARTFIMA(np.array(ind), phi, theta, var, d, lambda_, var2, len(x))

    if not return_sum:
        log_like = -w * (np.log(fj) + I[ind]/fj)
    else:
        log_like = np.sum(-w * (np.log(fj) + I[ind]/fj))

    return log_like


def exact_log_likelihood_arma(x, params, p, q):
    """
    Computes the exact likelihood of an ARMA process. Input is partial correlation parameterization and the variance term in its log.
    p is the #lags in MA
    q is the #lags in AR
    """
    # Transform to standard parameterization
    phi = np.array(reparam(params[:q]))
    theta = np.array(reparam(params[q: -1], MA=True))
    var = np.exp(params[-1])
    ans = sm.tsa.innovations.arma_loglike(
        x, ar_params=phi, ma_params=theta, sigma2=var)
    return(ans)


Nfeval = 1


def callbackF(Xi, nn):
    global Nfeval
    print('{0:4d} {1: 3.6f} {2:.2e}'.format(
        Nfeval, obj(Xi), np.linalg.norm(gr_logp(Xi))))
    Nfeval += 1


def log_likelihood(y, params):
    # This one is only for MCMC so no need for grouping.
    return np.sum(log_density(y, params))


def Taylor_proxies(params, paramsStar, dens_at_Star, grad_at_Star, Hess_at_Star, order=2):
    # Taylor proxies of order "order". Expands around paramsStar
    const_term = dens_at_Star
    if order == 0:
        aa = 1
    if order == 0:
        q_k = const_term
    elif order == 1:
        first_term = np.sum(grad_at_Star*(params - paramsStar), axis=1)
        q_k = const_term + first_term
    elif order == 2:
        first_term = np.sum(grad_at_Star*(params - paramsStar), axis=1)
        second_term = 0.5 * \
            np.sum(np.sum(Hess_at_Star*np.outer(params - paramsStar,
                   params - paramsStar), axis=1), axis=1)
        q_k = const_term + first_term + second_term
    else:
        raise ValueError("order must be 0<=order<=2")
    return q_k


def optimize_subset(obs, I, ind, args, bnds, x0, gtol, return_gradient=False):
    def log_likelihood(x): return -log_density_group(obs, I[ind], x, *args)
    if not bnds:
        bnds = ()
    # Get gradient for optimization
    gr_log_likelihood = grad(log_likelihood)
    if use_Hessians:
        hess_log_likelihood = hessian(log_likelihood)
    else:
        hess_log_likelihood = None

    if not return_gradient:
        print("Constructing control variates for group %s" % ind)
        res = minimize(log_likelihood, bounds=bnds, jac=gr_log_likelihood, hess=hess_log_likelihood,
                       method='trust-constr', x0=x0, options={'gtol': gtol, 'maxiter': max_iter_optim})
        assert(res['success'])
        return res
    else:
        # Only return gradient
        return gr_log_likelihood


def optimize_subset_unpack(args):
    # Needed for multiprocessing to not throw errors.
    return optimize_subset(*args)


def coreset_subset(obs, I, ind, args, bnds, x0, gtol, mode_and_Cov=None, return_gradient=False):
    # This construct a coreset for the subset indicated by ind.
    # Runs an optimization to find thetaStar to construct the weighting function for the coreset
    # Constructs a coreset for the ind group of the data
    #
    # NOTE: For some insane reason, M and project_dim cannot be inputs here (yields a Pickling Errror)
    #print("This function is called for ind = %s" % ind)
    if not singleTheta:
        def log_likelihood(x): return -log_density_group(obs, I[ind], x, *args)
        def log_dens(obs, x, idx=None): return log_density(obs, x, *args)

        # First find MAP and Covariance matrix to construct the posterior approximation.
        if not bnds:
            bnds = ()

        # Get gradient for optimization
        gr_log_likelihood = grad(log_likelihood)
        if use_Hessians:
            Hess_log_likelihood = hessian(log_likelihood)
        else:
            Hess_log_likelihood = None

        if not return_gradient:
            print("Constructing coreset control variates for group %s" % ind)
            res = minimize(log_likelihood, bounds=bnds, jac=gr_log_likelihood, hess=Hess_log_likelihood,
                           method='trust-constr', x0=x0, options={'gtol': gtol, 'maxiter': max_iter_optim})
            assert(res['success'])
            mode = res.x
            CovMatrix = np.linalg.inv(Hess_log_likelihood(mode))

        else:
            # Only return gradient
            return gr_log_likelihood
    else:
        def log_dens(obs, x, idx=None): return log_density(obs, x, *args)

        mode, CovMatrix = mode_and_Cov[0], mode_and_Cov[1]

    post_approx = simulate_truncated_normal(mode, CovMatrix, FI_term, TFI_term)

    ind_range = obs[I[ind]]

    proj = bc.ProjectionF(ind_range, log_dens, projection_dim, post_approx)

    # construct the N x K discretized log-likelihood matrix; each row represents the discretized
    # LL func for one datapoint
    vecs = proj.get()

    ############################
    # Step 4: Build the Coreset
    ############################

    # do coreset construction using the discretized log-likelihood functions
    giga = bc.GIGA(vecs)
    # build the coreset

    print("\n\nRunning GIGA Optimization for group %s..." % ind)

    for i in range(M):
        giga._step(True)
        # print(giga.error())

    wts = giga.weights()  # get the output weights

    # Begin debug:
    # idcs = wts > 0  # pull out the indices of datapoints that were included in the coreset
    #ind_range_sub = ind_range
    # ind_range_sub = # ind_range[I[ind]]
    #coreset_n = np.sum(wts != 0)
    #w_cs, ind_cs = wts[wts != 0], ind_range_sub[wts != 0]
    # coreset_n

    # Whittle group vs coreset
    #
    # whittle_group = np.sum(log_density(ind_range_sub, x0, *args_whittle))#, idx = None : log_density(obs, x, *args)
    #whittle_coreset = whittle_likelihood(ind_cs, x0, *args_whittle, use_w = True, w=w_cs, return_sum=True)
    # End debug
    return wts


def coreset_subset_unpack(args):
    # Needed for multiprocessing to not throw errors.
    return coreset_subset(*args)


def prepare_control_variates_Taylor(obs, I, Taylor_order, log_density, args, x0, singleTheta, MAP_singleTheta=None, bnds=False, file_control_variates=None):
    # Compute control variates for observations in obs.
    # If thetaStar is not None, runs an optimization to determine thetaStar, then constructs quantities to make a Taylor of order Taylor order  approximation
    #
    # obs: The observations to construct control variates for
    # I: The indices for the groups to construct the control variate for
    # Taylor order: Order of the approximation
    # log_density: The density function (we approximate the sum of log-densities of all observations)
    # args: extra argument required for log_density
    # x0: start value for optimization
    # bnds: A scipy optimize object constructed with Bounds

    if not os.path.isfile(file_control_variates + 'thetaStar.npy'):
        G = len(I)
        p = len(x0)
        thetaStar = np.zeros([G, p])
        sum_dens_at_Star = np.zeros(G)
        sum_grad_at_Star = np.zeros([G, p])
        sum_Hess_at_Star = np.zeros([G, p, p])

        success = np.zeros([G])

        if not singleTheta:
            # Construct stuff in parallell
            # For map (used for debugging) the input is different than Pool.map()
            #obs_list = [obs for item in range(G)]
            #I_list = [I for item in range(G)]
            #ind_list = [item for item in range(G)]
            #args_list = [args for item in range(G)]
            #bnds_list = [bnds for item in range(G)]
            #x0_list =[x0 for item in range(G)]
            #gtol_list =[gtol for item in range(G)]
            #res = list(map(optimize_subset, obs_list, I_list, ind_list, args_list, bnds_list, x0_list, gtol_list))
            # End map for debugging
            nWorkers = 8
            chunksize = None

            arguments = [[obs, I, ind, args, bnds, x0, gtol]
                         for ind in range(G)]
            # Initialize workers
            pool = mp.Pool(processes=nWorkers)
            # Evaluate function
            result = pool.map_async(
                optimize_subset_unpack, arguments, chunksize=chunksize)
            pool.close()
            pool.join()
            res = result.get()

            for ind in range(G):
                thetaStar[ind] = res[ind].x
                gr_log_likelihood = optimize_subset(
                    obs, I, ind, args, bnds, x0, gtol, return_gradient=True)
                assert(res[ind]['success'])

        else:
            assert(MAP_singleTheta is not None)

            print("Constructing control variates based on a single thetaStar")
            thetaStar = np.array([MAP_singleTheta for item in range(G)])

        for ind in range(G):
            if ind % 10 == 0:
                print(ind)

            grad_group, Hess_group = grad(
                log_density_group, 2), hessian(log_density_group, 2)
            sum_dens_at_Star[ind], sum_grad_at_Star[ind, :], sum_Hess_at_Star[ind, :, :] = log_density_group(obs, I[ind], thetaStar[ind], *args), \
                grad_group(obs, I[ind], thetaStar[ind], *args), \
                Hess_group(obs, I[ind], thetaStar[ind], *args)

        np.save(file_control_variates + 'thetaStar.npy', thetaStar)
        np.save(file_control_variates +
                'sum_dens_at_Star.npy', sum_dens_at_Star)
        np.save(file_control_variates +
                'sum_grad_at_Star.npy', sum_grad_at_Star)
        np.save(file_control_variates +
                'sum_Hess_at_Star.npy', sum_Hess_at_Star)
        np.save(file_control_variates + 'success.npy', success)

    else:
        thetaStar = np.load(file_control_variates + 'thetaStar.npy')
        sum_dens_at_Star = np.load(
            file_control_variates + 'sum_dens_at_Star.npy')
        sum_grad_at_Star = np.load(
            file_control_variates + 'sum_grad_at_Star.npy')
        sum_Hess_at_Star = np.load(
            file_control_variates + 'sum_Hess_at_Star.npy')
        success = np.load(file_control_variates + 'success.npy')

    return thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star, success


def prepare_control_variates_Taylor_no_grouping(obs, Taylor_order, log_density, args, x0, singleTheta, MAP_singleTheta=None, bnds=False, file_control_variates=None):
    # NOTE: We write a separate function here to make it faster. If we run the grouping code with G = n it is painfully slow.
    # Compute control variates for observations in obs.
    # If thetaStar is not None, runs an optimization to determine thetaStar, then constructs quantities to make a Taylor of order Taylor order  approximation
    #
    # obs: The observations to construct control variates for
    # I: The indices for the groups to construct the control variate for
    # Taylor order: Order of the approximation
    # log_density: The density function (we approximate the sum of log-densities of all observations)
    # args: extra argument required for log_density
    # x0: start value for optimization
    # bnds: A scipy optimize object constructed with Bounds
    if not os.path.isfile(file_control_variates + 'thetaStar.npy'):
        if singleTheta is False:
            raise("Function not implemented with this option.")

        p = len(x0)
        grad_at_Star = np.zeros([n, p])
        Hess_at_Star = np.zeros([n, p, p])
        thetaStar = np.array([MAP_singleTheta for item in range(n)])
        dens_at_Star = whittle_likelihood(
            obs, MAP_singleTheta, *args, FI_term, TFI_term, SV_model)
        grad_, Hess_ = grad(whittle_likelihood, 1), hessian(
            whittle_likelihood, 1)

        for ind in range(n):
            # print(ind)
            if ind % 10 == 0:
                print(ind)
            grad_at_Star[ind, :], Hess_at_Star[ind, :, :] = grad_(obs[ind], thetaStar[ind], *args, FI_term, TFI_term, SV_model, w=1, use_w=True), Hess_(
                obs[ind], thetaStar[ind], *args, FI_term, TFI_term, SV_model, w=1, use_w=True)

        np.save(file_control_variates + 'thetaStar.npy', thetaStar)
        np.save(file_control_variates + 'sum_dens_at_Star.npy', dens_at_Star)
        np.save(file_control_variates + 'sum_grad_at_Star.npy', grad_at_Star)
        np.save(file_control_variates + 'sum_Hess_at_Star.npy', Hess_at_Star)
    else:
        thetaStar = np.load(file_control_variates + 'thetaStar.npy')
        dens_at_Star = np.load(file_control_variates + 'sum_dens_at_Star.npy')
        grad_at_Star = np.load(file_control_variates + 'sum_grad_at_Star.npy')
        Hess_at_Star = np.load(file_control_variates + 'sum_Hess_at_Star.npy')

    return thetaStar, dens_at_Star, grad_at_Star, Hess_at_Star


def control_variates_Taylor(theta, u, thetaStar, dens_at_Star, grad_at_Star, Hess_at_Star,  singleTheta, sum_quants):
    # Computes the q_k for the sample in u evaluated at theta. In addition, returns the sum over all q_k in the population.
    # sum_quants = (dens_at_Star, grad_at_Star, Hess_at_Star) is the sum of all quantities. Only used if singleTheta is true (to no repeatedly compute the sum)
    theta_minus_thetaStar = theta - thetaStar
    if not singleTheta:
        theta_minus_thetaStar_outer_prod = np.matmul(
            theta_minus_thetaStar[:, :, np.newaxis], theta_minus_thetaStar[:, np.newaxis, :])
        qsum = np.sum(dens_at_Star) + np.sum(grad_at_Star*theta_minus_thetaStar) + \
            0.5*np.sum(Hess_at_Star*theta_minus_thetaStar_outer_prod)
        q_k_u = dens_at_Star[u] + np.sum(grad_at_Star[u]*theta_minus_thetaStar[u], axis=1) + \
            0.5 * \
            np.sum(
                np.sum(Hess_at_Star[u]*theta_minus_thetaStar_outer_prod[u], axis=1), axis=1)
    else:
        sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star = sum_quants
        qsum = sum_dens_at_Star + np.dot(sum_grad_at_Star, theta_minus_thetaStar[0]) + 0.5*np.dot(
            theta_minus_thetaStar[0], np.dot(sum_Hess_at_Star, theta_minus_thetaStar[0]))
        q_k_u = dens_at_Star[u] + np.sum(grad_at_Star[u]*theta_minus_thetaStar[u], axis=1) + \
            0.5*np.sum(np.sum(Hess_at_Star[u]*np.outer(
                theta_minus_thetaStar[0], theta_minus_thetaStar[0]), axis=1), axis=1)

    return q_k_u, qsum


def control_variates_coresets(theta, u, weights, indices):
    # Computes the q_k for the sample in u evaluated at theta. In addition, returns the sum over all q_k in the population.
    q_sum = np.sum([whittle_likelihood(indices[item], theta, *args_whittle, FI_term, TFI_term,
                   SV_model, use_w=True, w=weights[item], return_sum=True) for item in range(G)])
    q_k_u = np.array([whittle_likelihood(indices[item], theta, *args_whittle, FI_term,
                     TFI_term, SV_model, use_w=True, w=weights[item], return_sum=True) for item in u])
    return q_k_u, q_sum


def prepare_control_variates_Coresets(obs, I, args, x0, M, projection_dim, MAP_for_Coreset=None, bnds=False, file_control_variates=None):
    # Compute control variates for observations in obs based on coresets.
    # Constructs stuff in parallel

    # For some insane reason, can't pass M and projection_dim further as inputs to corest_subset_unpack (gives pickling error)

    # For map (used for debugging) the input is different than Pool.map()
    # obs_list = [obs for item in range(G)] # [obs, I, ind, args, bnds, x0, gtol, M, projection_dim]
    #I_list = [I for item in range(G)]
    #ind_list = [item for item in range(G)]
    #args_list = [args for item in range(G)]
    #bnds_list = [bnds for item in range(G)]
    #x0_list =[x0 for item in range(G)]
    #gtol_list =[gtol for item in range(G)]
    #mode = MAP_for_Coreset

    #log_likelihood = lambda x : -np.sum(log_density(obs, x, *args))

    #Hess_log_likelihood = hessian(log_likelihood)

    #CovMatrix = np.linalg.inv(Hess_log_likelihood(mode))
    #mode_and_Cov = (mode, CovMatrix)

    #mode_and_Cov_list = [mode_and_Cov for item in range(G)]

    #res = list(map(coreset_subset, obs_list, I_list, ind_list, args_list, bnds_list, x0_list, gtol_list, mode_and_Cov_list))
    # End map for debugging

    #arguments = [[obs, I, item, args, bnds, x0, gtol, mode_and_Cov] for item in range(G)]

    # Debug coreset
    #wts = coreset_subset(obs, I, 1, args_whittle, bnds, x0, gtol, M, projection_dim)

    # idcs = wts > 0  # pull out the indices of datapoints that were included in the coreset
    #ind_range_sub = ind_range[I[ind]]
    # ind_range_sub_list.append(ind_range_sub) # for debugging

    #coreset_n = np.sum(wts != 0)
    #w_cs, ind_cs = wts[wts != 0], ind_range_sub[wts != 0]
    # w_cs_list.append(w_cs)
    # ind_cs_list.append(ind_cs)
    #coreset_sizes[ind] = coreset_n

    # Whittle group vs coreset
    ##
    # whittle_group = [np.sum(log_density(ind_range_sub_list[ind], x0, *args_whittle)) for item in range(G)] #, idx = None : log_density(obs, x, *args)
    #whittle_coreset = [whittle_likelihood(ind_cs_list[item], x0, *args_whittle, use_w = True, w=w_cs_list[item], return_sum=True) for item in range(G)]

    nWorkers = 8
    chunksize = None

    if singleTheta:

        def log_likelihood(x): return -np.sum(log_density(obs, x, *args))
        # Run an optimization using all data.
        # First find MAP and Covariance matrix to construct the posterior approximation.
        if not bnds:
            bnds = ()

        # Get gradient for optimization
        gr_log_likelihood = grad(log_likelihood)

        Hess_log_likelihood = hessian(log_likelihood)

        print("Constructing coreset control variates based on a single thetaStar")
        if MAP_for_Coreset is None:
            res = minimize(log_likelihood, bounds=bnds, jac=gr_log_likelihood, hess=Hess_log_likelihood,
                           method='trust-constr', x0=x0, options={'gtol': gtol, 'maxiter': max_iter_optim})
            assert(res['success'])
            mode = res.x
            CovMatrix = np.linalg.inv(Hess_log_likelihood(mode))
            mode_and_Cov = (mode, CovMatrix)
        else:
            mode = MAP_for_Coreset
            CovMatrix = np.linalg.inv(Hess_log_likelihood(mode))
            mode_and_Cov = (mode, CovMatrix)

    arguments = [[obs, I, item, args, bnds, x0, gtol, mode_and_Cov]
                 for item in range(G)]
    # Initialize workers
    pool = mp.Pool(processes=nWorkers)
    # Evaluate function
    result = pool.map_async(coreset_subset_unpack,
                            arguments, chunksize=chunksize)
    pool.close()
    pool.join()
    res = result.get()

    w_cs_list = []
    ind_cs_list = []
    coreset_sizes = np.zeros(G)

    # For debugging:
    ind_range_sub_list = []
    # Post process
    for ind in range(G):
        wts = res[ind]
        idcs = wts > 0  # pull out the indices of datapoints that were included in the coreset
        ind_range = obs[I[ind]]
        ind_range_sub = ind_range
        ind_range_sub_list.append(ind_range_sub)  # for debugging

        coreset_n = np.sum(wts != 0)
        w_cs, ind_cs = wts[wts != 0], ind_range_sub[wts != 0]
        w_cs_list.append(w_cs)
        ind_cs_list.append(ind_cs)
        coreset_sizes[ind] = coreset_n

    # Check: Whittle group vs coreset
    whittle_group = [np.sum(log_density(ind_range_sub_list[item], x0, *args_whittle))
                     for item in range(G)]  # , idx = None : log_density(obs, x, *args)
    whittle_coreset = [whittle_likelihood(ind_cs_list[item], x0, *args_whittle, FI_term,
                                          TFI_term, SV_model, use_w=True, w=w_cs_list[item], return_sum=True) for item in range(G)]
    return w_cs_list, ind_cs_list, coreset_sizes


####
# The dataset is divided into groups stored in I. The proxy for group ind is then for the sum of the log-likelihood of the observations indicated by I[ind]
####
n = 450001  # 1000001 #2001 # 2001 #5000001 #2001 #1000001 #450001 #176001 #1000001 #450001
# n =  # 44001 #1000001 #2001 # 2001 #5000001 #2001 #1000001 #450001 #176001 #1000001 #450001

run_MCMC_true_likelihood = False
FI_term = True  # False # False #True # if FI_term is true then d is estimated and the process is an ARFIMA. If FI_term = False then d = 0 and process is an ARMA
TFI_term = True  # False # True #False #True #False #True
SV_model = False  # True #False # True # ARTFIMA model for stochastic volatilities
if TFI_term:
    assert(FI_term)
if SV_model:
    assert(TFI_term)

if FI_term:
    assert(not run_MCMC_true_likelihood)  # Too expensive on large data.

# Priors for (transformed) var2, (transformed) lambda (transformed) d and (transform) sigma2 (prior for d only if FI_term = True, prior for lambda only if TFI_term = True)
prior_mean_ginv_var2_param = 0
prior_std_ginv_var2_param = 0.01

prior_mean_ginv_lambda_param = 0
prior_std_ginv_lambda_param = 1

prior_mean_ginv_d_param = 0
prior_std_ginv_d_param = 1

prior_mean_ginv_sigma2_param = -3  # 0 #np.log(np.var(x))
prior_std_ginv_sigma2_param = 1

# Set AR and MA lags
q = 2
p = 2

if FI_term:
    if TFI_term:
        if SV_model:
            n_params = q + p + 4
        else:
            n_params = q + p + 3
    else:
        n_params = q + p + 2
    # prior will be defined after reading in data
else:
    n_params = q + p + 1  # d is not estimated. NOTE: no prior for d here


def log_prior(theta):
    # theta - first part contain process parameter.

    # Last AR or MA
    if FI_term:
        if TFI_term:
            if SV_model:
                Last_MA_AR = 4
            else:
                Last_MA_AR = 3
        else:
            Last_MA_AR = 2
    else:
        Last_MA_AR = 1

    if not any(abs(theta[:-Last_MA_AR]) > 1):
        prior_process_params = -len(theta[:-Last_MA_AR]) * np.log(2)
    else:
        prior_process_params = -np.inf

    # Prior for ginv(sigma2)
    prior_ginv_sigma2_param = sps_autograd.norm.logpdf(
        theta[-2], loc=prior_mean_ginv_sigma2_param, scale=prior_std_ginv_sigma2_param)

    if FI_term:
        if TFI_term:

            prior_ginv_lambda_param = sps_autograd.norm.logpdf(
                theta[-3], loc=prior_mean_ginv_lambda_param, scale=prior_std_ginv_lambda_param)

            if SV_model:
                prior_ginv_var2_param = sps_autograd.norm.logpdf(
                    theta[-4], loc=prior_mean_ginv_var2_param, scale=prior_std_ginv_var2_param)
            else:
                prior_ginv_var2_param = 0
        else:
            prior_ginv_lambda_param = 0
            prior_ginv_var2_param = 0

        # Prior for ginv(d)
        prior_ginv_d_param = sps_autograd.norm.logpdf(
            theta[-1], loc=prior_mean_ginv_d_param, scale=prior_std_ginv_d_param)

    else:
        prior_ginv_d_param = 0
        prior_ginv_lambda_param = 0
        prior_ginv_var2_param = 0

    return prior_process_params + prior_ginv_d_param + prior_ginv_sigma2_param + prior_ginv_lambda_param + prior_ginv_var2_param


data_set_name = "Bromma_AR2_TFI_MA2"  # "Bitcoin_AR1_TFI_MA1_SV_new" #"Bitcoin_AR1_TFI_MA1_SV_new" # "Bromma_AR2_TFI_MA2_smalldata" # "Bitcoin_AR1_TFI_MA1_SV_smalldata" # "Bitcoin_AR1_TFI_MA1_SV" # "Bromma_AR1_TFI_MA1_SV" #"Gold_AR5_TFI_MA0_SV" #"Bitcoin_AR1_TFI_MA1_SV" #"Bromma_AR2_TFI_MA2"  #"Vancouver_AR2_MA3_smalldata" #"Villani_AR2_TFI_MA2" #  "Villani_AR5_FI_MA3" #"Vancouver_AR2_MA3" # "Vancouver_AR2_MA3_Gaussian_approximation" #"Vancouver_AR2_MA3" #_smalldata"  #"Vancouver_AR2_MA3" #"Vancouver_AR9_MA3" #"Vancouver_AR3_MA4" # "ARFIMA_AISTATS_2million" # "ARFIMA_AISTATS_200K" #"ARFIMA_AISTATS_2million" # 200K" #"ARFIMA_AISTATS_200K" #"LAtemp_44K" #"ARFIMA_AISTATS_200K"  #"Moretti_20K" #"ARFIMA_AR3_MA2_d.49_20K" #"new_ARFIMA_example_2K" # 'new_ARFIMA_example_20K' #'new_ARFIMA_example_2million' #"new_ARFIMA_example_200K" # "LAtemp" #"new_ARFIMA_example_200K" # 'new_ARFIMA_example_2million' # Note: this data was generated with the following parameters
data_set_file_name = wrkDir + '/Data/%s' % data_set_name + '.npy'

if not os.path.exists(wrkDir + '/Data/StoreForRuns/'):
    os.makedirs(wrkDir + '/Data/StoreForRuns/')

file_control_variates = wrkDir + \
    '/Data/StoreForRuns/Taylor_Control_variates_%s' % data_set_name + '.npy'

if data_set_name == "Simulated_AR2_FI_MA1":
    assert(n == 5000001)
    assert(q == 2 and p == 1)
    assert(FI_term and not TFI_term and not SV_model)
elif data_set_name == "Simulated_AR2_FI_MA1_smalldata":
    assert(n == 2001)
    assert(q == 2 and p == 1)
    assert(FI_term and not TFI_term and not SV_model)
elif data_set_name == "Bitcoin_AR1_TFI_MA1_SV_smalldata":
    assert(n == 2001)
    assert(q == 1 and p == 1)
    assert(FI_term and TFI_term and SV_model)
elif data_set_name == "Bitcoin_AR1_TFI_MA1_SV":
    assert(n == 1000001)
    assert(q == 1 and p == 1)
    assert(FI_term and TFI_term and SV_model)
elif data_set_name == "Bitcoin_AR1_TFI_MA1_SV_new":
    assert(n == 1000001)
    assert(q == 1 and p == 1)
    assert(FI_term and TFI_term and SV_model)
elif data_set_name == "Bitcoin_AR1_TFI_MA1_SV_new_smalldata":
    assert(n == 2001)
    assert(q == 1 and p == 1)
    assert(FI_term and TFI_term and SV_model)
elif data_set_name == "Simulated_AR2_FI_MA1":
    assert(n == 2000001)
    assert(q == 2 and p == 1)
    assert(FI_term and not TFI_term and not SV_model)
elif data_set_name == "Bromma_AR2_TFI_MA2":
    assert(n == 450001)
    assert(q == 2 and p == 2)
    assert(FI_term and TFI_term and not SV_model)
elif data_set_name == "Bromma_AR2_TFI_MA2_smalldata":
    assert(n == 2001)
    assert(q == 2 and p == 2)
    assert(FI_term and TFI_term and not SV_model)
elif data_set_name == "Vancouver_AR2_MA3":
    assert(n == 44001)
    assert(q == 2 and p == 3)
    assert(FI_term is False and not TFI_term and not SV_model)
elif data_set_name == "Vancouver_AR2_MA3_smalldata":
    assert(n == 2001)
    assert(q == 2 and p == 3)
    assert(FI_term is False and not TFI_term and not SV_model)
else:
    raise ValueError()

x = np.load(data_set_file_name)
assert(len(x) == n)
if FI_term:
    true_param = 0.01*np.ones(n_params)
    true_param[-1] = np.arctanh(2*0.45)

else:
    true_param = 0.01*np.ones(n_params)


# Data has been read in. Set subsampling options. We will subsample the frequencies, whose indices are ind_range
# and are of length floor((n-1)/2)
n = np.int(np.floor((n-1)/2))

G = 1000  # 500000 # 100 #1000 # 500000 #1000  # Number of groups.

file_control_variates = file_control_variates + 'G%s/' % G
if not os.path.exists(file_control_variates):
    os.makedirs(file_control_variates)

groupSize = np.int(n/G)

# I = [np.arange(start, start + groupSize) for start in np.arange(0, n, groupSize)] # Every element in list contains groupSize indices
# Here every group contains one observation from each frequency band. Every element in list contains groupSize indices
I = [item + np.arange(0, n, G) for item in range(G)]
# I = [item + np.arange(25000, n, G) for item in range(G)] # TEMP for debugging: want to always include the first 25000 frequencies. These are never subsampled.

# "coreset" #"Taylor" # "coreset" # "Taylor" #"coreset" # "Taylor" # "coreset"  # "Taylor" # "coreset"
control_variate_type = "Taylor"

if control_variate_type == "Taylor":
    # Settings for the control variates based on Taylor approximations
    Taylor_order = 2
    singleTheta = True
    Taylor_options = {'Taylor_order': Taylor_order, 'singleTheta': singleTheta}
    coreset_options = None
elif control_variate_type == "coreset":
    # Options for coreset.
    # Only makes sense to do coreset control variates if we have many observations per group
    assert(n/G >= 100)
    # NOTE: For some insane reason, we can't have these (M and projection_dim) as inputs to coreset_subset if we want to run in parallel. Define them in the top of this file instead
    # projection_dim = 500  # random projection dimension, K
    # frac_coreset = 0.01 # Maximum size of the coreset in terms of the fraction of the observations allocated to one group (np.floor(n/G)).
    # M = np.int(frac_coreset*n/G)
    coreset_options = {
        'M': M, 'projection_dim': projection_dim, 'singleTheta': singleTheta}
    Taylor_options = None

# m = 10 #220 #10000 # Number of "subsampled groups". number of density evaluations is m*groupSize.
m = np.int(0.01*G)
#m = 20
nBlocks = 10  # 20 #20 #10 # 10 #10 #10 #1 #10 # Number of blocks for PMMH. If 1 then standard PMMH, otherwise blockwise with nBlocks blocks
assert(m >= 2)  # Need to be able to estimate a variance for the bias-correction
assert(nBlocks <= m)  # Can have more if we want to do blocking.
params = true_param
ind_range = np.arange(1, int(np.floor((len(x))/2))+1)
# compute periodogram for Fourier frequencies : O(n log n)
I_pg = p_gram(fft(x))
y = ind_range  # Data are now the frequencies

params = true_param + sps.norm.rvs(0, 0.1, size=len(true_param))

#############
# Laplace approximation before MCMC. For starting value and proposal covariance matrix.
#############


def log_p(x): return log_prior(x) + whittle_likelihood(ind_range, x, I_pg, p, q, FI_term, TFI_term, SV_model,
                                                       return_sum=True)

# for paramsStar


def log_l(x): return whittle_likelihood(ind_range, x, I_pg, p, q, FI_term, TFI_term, SV_model,
                                        return_sum=True)


args_whittle = (I_pg, p, q)


lb = [-1]*len(true_param)
ub = [1]*len(true_param)


if TFI_term:
    if SV_model:
        lb[-4:] = [-30, -30, -30, -30]
        ub[-4:] = [30, 30, 30, 30]
    else:
        lb[-3:] = [-30, -30, -30]
        ub[-3:] = [30, 30, 30]
else:
    if FI_term:
        lb[-2:] = [-30, -30]
        ub[-2:] = [30, 30]
    else:
        # Just ARMA
        lb[-1:] = [-30]
        ub[-1:] = [30]

bnds = Bounds(lb, ub, keep_feasible=True)


def obj(prm): return -log_p(prm)


jacobian = grad(obj)
gr_logp, H_logp = grad(log_p), hessian(log_p)


def obj_likelihood(prm): return -log_l(prm)


jacobian_likelihood = grad(obj_likelihood)

if use_Hessians:
    hess_likelihood = hessian(obj_likelihood)
else:
    hess_likelihood = None

gr_logp, H_logp = grad(log_p), hessian(log_p)

if use_Hessians:
    hs = hessian(obj)
else:
    hs = None

print("Optimizing for MAP...")


# Try reading in MAP, if it exist.
if FI_term:
    if TFI_term:
        if SV_model:
            file_name = path + \
                'MAP_ARTFIMA_SV_%s_n%s_AR%s_MA%s_seed%s' % (
                    data_set_name, n, q, p, seed) + '.npy'
        else:
            file_name = path + \
                'MAP_ARTFIMA_%s_n%s_AR%s_MA%s_seed%s' % (
                    data_set_name, n, q, p, seed) + '.npy'

    else:
        file_name = path + \
            'MAP_ARFIMA_%s_n%s_AR%s_MA%s_seed%s' % (
                data_set_name, n, q, p, seed) + '.npy'
else:
    file_name = path + \
        'MAP_ARMA_%s_n%s_AR%s_MA%s_seed%s' % (
            data_set_name, n, q, p, seed) + '.npy'

if not os.path.isfile(file_name):
    def callbackF(Xi, state):
        if (state.nit % 10) == 0:
            print('{0:4d} {1: 3.6f} {2:.2e}'.format(
                state.nit, state.fun, np.linalg.norm(state.grad)))
            print(Xi)

    res = minimize(obj, bounds=bnds, jac=jacobian, hess=hs, method='trust-constr',
                   x0=true_param, options={'gtol': 1e-4, 'maxiter': max_iter_optim}, callback=callbackF)
    MAP = res.x
    assert(res['success'])

    # NOTE: Should inspect if the likelihood optimization is very different than the posterior optimization. If so paramStar should be chosen as point for Taylor approx.
    paramsStar = res.x
    res2 = minimize(obj_likelihood, bounds=bnds, jac=jacobian_likelihood, hess=hess_likelihood,
                    method='trust-constr', x0=res.x, options={'gtol': gtol, 'maxiter': max_iter_optim}, callback=callbackF)
    # res2.x # When they are very dissimilar we store the MAP instead.
    paramsStar = paramsStar
    assert(res2['success'])
    np.save(file_name, np.hstack((MAP.reshape(-1, 1), paramsStar.reshape(-1, 1))))
else:
    MAP_and_paramsStar = np.load(file_name)
    MAP = MAP_and_paramsStar[:, 0]
    paramsStar = MAP_and_paramsStar[:, 1]

print('MAP', MAP)
print('Likelihood mode', paramsStar)

Laplace_Cov = np.linalg.inv(-H_logp(MAP))
C = np.linalg.cholesky(Laplace_Cov)  # Proposal covariance

print('Laplace Approximation is Normal with mean MAP and covariance matrix \n')
print(Laplace_Cov)


############
# Start MCMC sampling
############
def RWMH(log_p, x_init, C, mcmc_length, h, report=True, flag=' '):
    # Standard Random Walk Metropolis-Hastings Sampler
    # log_p: function that takes parameter value as its single argument
    # x_init: initial point for the chain
    # C : (Cholesky) preconditioning matrix, proposals have covariance C @ C.T
    # report: set to False to suppress print output

    if report:
        print('\nRunning', flag,  'MCMC...')

    tic = time.time()
    n_params = len(x_init)
    samples = np.zeros([mcmc_length+1, n_params])
    samples[0, :] = x_init
    stats = np.zeros((mcmc_length+1, 2))
    log_p_draws = np.zeros(mcmc_length + 1)

    bar = progressbar.progressbar(range(1, mcmc_length+1))

    for i in bar:

        if i % 100 == 0:
            print("\nacceptance: {:.2f} , ESJD: {:.2e}, time: {:.2f}".format(
                np.mean(stats[:i, 1]), np.mean(stats[:i, 0]), time.time()-tic))

        temp = h * np.dot(C, np.random.randn(n_params))
        #temp[-1] = 0.01*temp[-1]

        prop = samples[i-1, :] + temp  # last * was @?
        log_p_Curr = log_p(samples[i-1, :])
        log_p_draws[i - 1] = log_p_Curr
        alph = np.min([1, np.exp(log_p(prop) - log_p_Curr)])

        stats[i, :] = [np.linalg.norm(prop - samples[i-1, :]) * alph, alph]

        if np.random.rand() < alph:
            samples[i, :] = prop
        else:
            samples[i, :] = samples[i-1, :]

    log_p_draws[i] = log_p(samples[i, :])

    if report:
        print("\nacceptance: {:.2f} , ESJD: {:.2e}, time: {:.2f}".format(
            np.mean(stats[:, 1]), np.mean(stats[:, 0]), time.time()-tic))

    return samples, log_p_draws, stats


def pseudo_marginal_RWMH(log_density_group, x_init, paramsStar, C, mcmc_length, h, args_whittle,  bnds,
                         control_variate_type, MAP_for_Taylor=None, coreset_options=None, Taylor_options=None, report=True, flag=' '):
    # Pseudo marginal MH. Modification of RWMH above
    # log_density_group: the log-density for each group (we use grouping for controlvariates)
    # x_init: initial point for the chain
    # C : (Cholesky) preconditioning matrix, proposals have covariance C @ C.T
    # report: set to False to suppress print output

    # Prepare proxies:
    ########
    # Prepare for proxies
    ########
    Taylor_options = Taylor_options
    dens_eval = 0

    if control_variate_type == "Taylor":
        Taylor_order, singleTheta = Taylor_options['Taylor_order'], Taylor_options['singleTheta']

        if G == n:
            thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star = prepare_control_variates_Taylor_no_grouping(ind_range, Taylor_order, log_density, args_whittle,
                                                                                                                          x_init, singleTheta, MAP_singleTheta=MAP_for_Taylor, bnds=bnds, file_control_variates=file_control_variates)

            success = True
        else:
            thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star, success = prepare_control_variates_Taylor(ind_range, I, Taylor_order, log_density,
                                                                                                                       args_whittle, x_init, MAP_singleTheta=MAP_for_Taylor, singleTheta=singleTheta, bnds=bnds, file_control_variates=file_control_variates)

        coreset_sizes = None
    elif control_variate_type == "coreset":
        M, projection_dim = coreset_options['M'], coreset_options['projection_dim']
        weights, indices, coreset_sizes = prepare_control_variates_Coresets(
            ind_range, I, args_whittle, x_init, M, projection_dim, bnds=bnds, MAP_for_Coreset=MAP_for_Taylor)
        # Cost of evaluating the coreset for each of the groups (when constructing the control variates)
        coreset_eval = np.sum(coreset_sizes)
        q_sum = np.sum([whittle_likelihood(indices[item], x_init, *args_whittle, FI_term, TFI_term,
                       SV_model, use_w=True, w=weights[item], return_sum=True) for item in range(G)])
    else:
        raise ValueError("Wrong value for control_variate_typ")

    if control_variate_type == "Taylor":
        if singleTheta:
            sum_quants = np.sum(sum_dens_at_Star), np.sum(
                sum_grad_at_Star, axis=0), np.sum(sum_Hess_at_Star, axis=0)
        else:
            sum_quants = None

    # G is the number of groups for forming the control variate
    uCurr = npr.randint(0, G, m)

    # Divide the random variates into blocks
    if nBlocks > 1:
        blockIndicators = np.hstack((np.repeat(np.arange(nBlocks-1), m/nBlocks), np.repeat(
            nBlocks-1, m - len(np.repeat(np.arange(nBlocks-1), m/nBlocks)))))

    # Estimate the likelihood at the first parameter
    if control_variate_type == "Taylor":
        q_k_sub, q_sum = control_variates_Taylor(
            x_init, uCurr, thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star,  singleTheta, sum_quants)
        dens_eval += m*groupSize
        # Evaluating q_sum considered O(1)
    else:
        q_k_sub, q_sum = control_variates_coresets(
            x_init, uCurr, weights, indices)
        dens_eval += m*groupSize
        dens_eval += coreset_eval  # Evaluating q_sum

    l_k_sub = log_density_groups(y, itemgetter(*uCurr)(I), x_init, I_pg, p, q)
    # Note: G is the new population size
    sigma2_LL_hat = G**2*np.var(l_k_sub - q_k_sub, ddof=0)/m
    l_hat_Curr = q_sum + G*np.mean(l_k_sub - q_k_sub) - sigma2_LL_hat/2
    # TEMP: if we want to always include X frequencies l_hat_Curr = l_hat_Curr + whittle_likelihood(ind_range[:25000], x_init, I_pg, p, q, FI_term, TFI_term, SV_model, return_sum=True) # Adds 25 K observations with lowest frequencies

    if report:
        print('\nRunning', flag,  'pseudo marginal MCMC...')

    tic = time.time()
    n_params = len(x_init)
    samples = np.zeros([mcmc_length+1, n_params])
    sigma2_LL = np.zeros(mcmc_length+1)
    l_hats = np.zeros(mcmc_length+1)
    samples[0, :] = x_init  # No need to save u's
    sigma2_LL[0] = sigma2_LL_hat
    l_hats[0] = l_hat_Curr
    stats = np.zeros((mcmc_length+1, 2))

    bar = progressbar.progressbar(range(1, mcmc_length+1))
    debug_proxy = False
    for i in bar:
        temp = h * np.dot(C, np.random.randn(n_params))
        #temp[-1] = 0.01*temp[-1]
        prop = samples[i-1, :] + temp

        if i % 100 == 0:
            print("\nacceptance: {:.2f} , ESJD: {:.2e}, time: {:.2f}".format(
                np.mean(stats[:i, 1]), np.mean(stats[:i, 0]), time.time()-tic))

        # Update u:
        if nBlocks > 1:
            toUpdate = npr.randint(0, nBlocks, 1)[0]
            uProp = copy.copy(uCurr)
            update = (blockIndicators == toUpdate)
            uProp[update] = npr.randint(0, G, np.sum(
                update))  # G is the population size
        else:
            uProp = npr.randint(0, G, m)  # Sample independently

        # Estimate likelihood at proposed point
        if control_variate_type == "Taylor":
            q_k_sub, q_sum = control_variates_Taylor(
                prop, uProp, thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star,  singleTheta, sum_quants)
            dens_eval += m*groupSize
        else:
            q_k_sub, q_sum = control_variates_coresets(
                prop, uProp, weights, indices)
            dens_eval += m*groupSize
            dens_eval += coreset_eval  # Evaluating q_sum

        l_k_sub = log_density_groups(
            y, itemgetter(*uProp)(I), prop, I_pg, p, q)
        # Note: G is the new population size
        sigma2_LL_hat = G**2*np.var(l_k_sub - q_k_sub, ddof=0)/m
        sigma2_LL[i] = sigma2_LL_hat
        l_hat_Prop = q_sum + G*np.mean(l_k_sub - q_k_sub) - sigma2_LL_hat/2
        # TEMP: if we want to have X (=25000) frequencies we always include: l_hat_Prop = l_hat_Prop + whittle_likelihood(ind_range[:25000], prop, I_pg, p, q, FI_term, TFI_term, return_sum=True)

        # Debugging
        if debug_proxy:

            dens_All = log_density_groups(y, I, prop, I_pg, p, q)

            if control_variate_type == "Taylor":
                q_k_All, crap = q_k_sub, q_sum = control_variates_Taylor(prop, np.arange(
                    0, G), thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star,  singleTheta, sum_quants)

            else:
                q_k_All, crap = control_variates_coresets(
                    prop, np.arange(0, G), weights, indices)

            sigma2_LL_true = G**2 * \
                np.var(dens_All - q_k_All, ddof=0)/m  # True variance
            plt.plot(dens_All, q_k_All, '.')
            plt.plot(dens_All, dens_All, '-r')
            plt.show()

        # np.min([1, np.exp(log_p(prop) - log_p(samples[i-1, :]))])
        alph = np.min([1, np.exp(l_hat_Prop + log_prior(prop) -
                      (l_hat_Curr + log_prior(samples[i-1, :])))])
        stats[i, :] = [np.linalg.norm(prop - samples[i-1, :]) * alph, alph]

        if np.random.rand() < alph:
            samples[i, :] = prop
            l_hat_Curr = l_hat_Prop  # Accept random numbers
            uCurr = uProp
        else:
            samples[i, :] = samples[i-1, :]

        l_hats[i] = l_hat_Curr

    if report:
        print("\nacceptance: {:.2f} , ESJD: {:.2e}, time: {:.2f}".format(
            np.mean(stats[:, 1]), np.mean(stats[:, 0]), time.time()-tic))

    return samples, sigma2_LL, l_hats, dens_eval, stats, coreset_sizes


run_pseudo_marginal = True
mcmc_length = 10000  # 5000 #2500 #3000
burnIn = np.int(mcmc_length/10)
h = 2.38/np.sqrt(len(MAP))
n_params = len(paramsStar)

params = MAP + sps.norm.rvs(0, 0.01, len(MAP))

if run_pseudo_marginal:
    PMMH_samples, sigma2_LL, l_hats, dens_eval, stats, coreset_sizes = pseudo_marginal_RWMH(log_density_group, params, paramsStar, C, mcmc_length, h, args_whittle,
                                                                                            bnds, control_variate_type, MAP_for_Taylor=paramsStar, coreset_options=coreset_options, Taylor_options=Taylor_options, report=True, flag=' ')
    PMMH_samples_pre_BurnIn = PMMH_samples
    PMMH_samples = PMMH_samples[burnIn:, :]


# Random walk MH on the Whittle approximation
if FI_term:
    if TFI_term:
        if SV_model:
            path = wrkDir + '/Results/Runs/ARTFIMA_SV/'
            file_name = path + 'MCMC_ARTFIMA_SV_data%s_n%s_AR%s_MA%s_seed%s_MCMCiter%s' % (
                data_set_name, n, q, p, seed, mcmc_length) + '.pkl'
        else:
            path = wrkDir + '/Results/Runs/ARTFIMA/'
            file_name = path + 'MCMC_ARTFIMA_data%s_n%s_AR%s_MA%s_seed%s_MCMCiter%s' % (
                data_set_name, n, q, p, seed, mcmc_length) + '.pkl'
    else:
        path = wrkDir + '/Results/Runs/ARFIMA/'
        file_name = path + 'MCMC_ARFIMA_data%s_n%s_AR%s_MA%s_seed%s_MCMCiter%s' % (
            data_set_name, n, q, p, seed, mcmc_length) + '.pkl'
else:
    path = wrkDir + '/Results/Runs/ARMA/'
    file_name = path + 'MCMC_ARMA_data%s_n%s_AR%s_MA%s_seed%s_MCMCiter%s' % (
        data_set_name, n, q, p, seed, mcmc_length) + '.pkl'

if not os.path.exists(path):
    os.makedirs(path)


if not os.path.isfile(file_name):
    samples, log_p_draws, stats = RWMH(
        log_p, params, C, mcmc_length, h, report=True, flag=' ')

    saveDict = {'samples': samples, 'log_p_draws': log_p_draws, 'stats': stats}

    f = open(file_name, "wb")
    pickle.dump(saveDict, f)
    f.close()

else:
    f = open(file_name, "rb")
    loadDict = pickle.load(f)
    f.close()

    samples = loadDict['samples']
    log_p_draws = loadDict['log_p_draws']


samples = samples[burnIn:, :]

if run_MCMC_true_likelihood:
    path = wrkDir + '/Results/Runs/ARMA/'
    file_name = path + 'MCMC_true_likelihood_ARMA_data%s_n%s_AR%s_MA%s_seed%s_MCMCiter%s' % (
        data_set_name, n, q, p, seed, mcmc_length) + '.pkl'

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.isfile(file_name):
        def log_p_exact(z): return log_prior(
            z) + exact_log_likelihood_arma(x, z, p, q)

        def obj_log_p_exact(z): return -log_p_exact(z)
        res = minimize(obj_log_p_exact, bounds=bnds, method='trust-constr', x0=np.zeros(
            len(true_param)), options={'gtol': gtol, 'maxiter': max_iter_optim})
        assert(res.success)
        # For proposal covariance matrix
        Hess_likelihood_FD = Hess_finite_diff(obj_log_p_exact)
        Laplace_Cov_exact = np.linalg.inv(Hess_likelihood_FD(res.x))
        C = np.linalg.cholesky(Laplace_Cov)  # Proposal covariance

        samples_exact, log_p_draws_exact, stats_exact = RWMH(
            log_p_exact, res.x, C, mcmc_length, h, report=True, flag=' ')

        saveDict = {'samples': samples_exact,
                    'log_p_draws': log_p_draws_exact, 'stats': stats_exact}

        f = open(file_name, "wb")
        pickle.dump(saveDict, f)
        f.close()

    else:
        f = open(file_name, "rb")
        loadDict = pickle.load(f)
        f.close()

        samples_exact = loadDict['samples']
        log_p_draws_exact = loadDict['log_p_draws']

    samples_exact = samples_exact[burnIn:, :]


n_rows = np.int(np.ceil(n_params/4))
n_cols = np.min([n_params, 4])

# Save results
resultsSaved = path + control_variate_type + '/'
resultsSaved = resultsSaved + \
    str(datetime.datetime.now().date()) + '/' + \
    str(datetime.datetime.now().time())[:5] + '/'

if not os.path.exists(resultsSaved):
    os.makedirs(resultsSaved)

if run_pseudo_marginal:
    # Save pseudo marginal stuff

    saveDict = {'samples': PMMH_samples_pre_BurnIn, 'sigma2_LL': sigma2_LL, 'l_hats': l_hats, 'dens_eval': dens_eval,
                'stats': stats, 'coreset_sizes': coreset_sizes}  # Can't pickle dictionary of dictionaries

    f = open(resultsSaved + 'Pseudo_marginal_results' + '.pkl', "wb")
    pickle.dump(saveDict, f)
    f.close()

    if Taylor_options is not None:
        f = open(resultsSaved + 'Taylor_options' + '.pkl', "wb")
        pickle.dump(Taylor_options, f)
        f.close()

    if coreset_options is not None:
        f = open(resultsSaved + 'Coreset_options' + '.pkl', "wb")
        pickle.dump(coreset_options, f)
        f.close()

file_ = open(resultsSaved + 'options', 'w')

# Write all above into this file
file_.write(
    '-------------------------------------------------------------------------------------------------\n')
file_.write('FI term = %s. \n' % FI_term)
file_.write('TFI term = %s. \n' % TFI_term)
file_.write('SV model  = %s. \n' % SV_model)
file_.write('AR(%s) and MA(%s). \n' % (q, p))
file_.write('Data set name = %s. \n' % data_set_name)
file_.write('Data set file name = %s. \n' % data_set_file_name)
file_.write('Subsampling and other settings \n')
file_.write('m = %s, n = %s, G = %s (groupSize = %s), nBlocks = %s \n' %
            (m, n, G, groupSize, nBlocks))
file_.write('Settings for control variates \n')
if control_variate_type == "Taylor":
    file_.write('Taylor control variates \n')
    file_.write('Taylor order = %s \n' % Taylor_order)
    file_.write('Single theta = %s \n' % singleTheta)
elif control_variate_type == "coreset":
    file_.write('Coreset control variates \n')
    file_.write('Projection_dim = %s \n' % projection_dim)
    file_.write('M = %s \n' % M)
    file_.write('Actual coreset size (average) = %s \n' %
                np.mean(coreset_sizes))


file_.close()
print(coreset_sizes)

plt.clf()
for i in range(1, n_params+1):
    plt.subplot(n_rows, n_cols, i)

    full = sns.kdeplot(pd.DataFrame(samples)[i-1], shade=False,
                       legend=False, linewidth=3, c='b')
    cs = sns.kdeplot(pd.DataFrame(PMMH_samples)[i-1], shade=False,
                     legend=False, c='r', linewidth=2)

    if run_MCMC_true_likelihood:
        exact = sns.kdeplot(pd.DataFrame(samples_exact)[i-1], shade=False,
                            legend=False, c='g', linewidth=2)

    l, r = plt.xlim()  # return the current xlim
    xx = np.linspace(l, r, 500)


if not run_MCMC_true_likelihood:
    plt.legend(('Full Data', 'Pseudo'))
else:
    plt.legend(('Full Data', 'Pseudo', 'Exact'))

plt.savefig(resultsSaved + "KDEs.pdf")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(sigma2_LL, color='blue', lw=1)
ax.set_yscale('log')
plt.savefig(resultsSaved + "sigma2_LL.pdf")

plt.figure()
for i in range(1, n_params+1):
    plt.subplot(n_rows, n_cols, i)
    plt.plot(PMMH_samples[:, i-1])


plt.savefig(resultsSaved + "PMMH_samples_trace_plots.pdf")


plt.figure()
for i in range(1, n_params+1):
    plt.subplot(n_rows, n_cols, i)
    plt.plot(samples[:, i-1])

plt.savefig(resultsSaved + "MCMC_samples_trace_plots.pdf")
