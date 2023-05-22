"""
This file is to investigate how efficient the control variates are. It is basically the main file but with some added loops.
It requires running the main program to produce all inputs of interest for the datasets with 2001 observations.

Obviously, the file paths need to be changed to where the input data is stored.
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
import pickle

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

max_iter_optim = 500  # How many iteration of the optimizer before giving up.

wrkDir = sys.path[0]
path = wrkDir + '/Results/Runs/'

# For some insane reason, we can't pass these as arguments to the function if we want to avoid pickling error with multiprocessing.
M, projection_dim, singleTheta = 200, 500, True


gtol = 1e-4  # 1e-8 # 1e-4

# ARFIMA reparametrized
np.random.seed(123)

seed = 1
np.random.seed(seed)  # This seed gives hard data if frequencies are grouped.


def log_density(ind, params, I_pg, p, q):
    # Returns the whittle log-density for ind
    return whittle_likelihood(ind, params, I_pg, p, q, FI_term, TFI_term, SV_model, w=[],
                              use_w=False, return_sum=False)


def log_density_groups(y, I, params, I_pg, p, q):
    # Log-density summed within a group for all groups in I.
    # u is a list of length G, where each element in the list contains an array of size groupSize (with indices)
    return np.array([log_density_group(y, I[ind], params, I_pg, p, q) for ind in range(len(I))])


def sum_log_density_groups(y, I, params, I_pg, p, q):
    # The sum over all log_density groups
    return np.sum(log_density_groups(y, I, params, I_pg, p, q))


def log_density_group(y, I_single, params, I_pg, p, q):
    # Log-density summed within a single group indicated by u_single.
    return np.sum(log_density(y[I_single], params, I_pg, p, q))


# Example model: The Whittle likelihood for ARFIMA. Has additional functions for reparameterization to induce stationarity
def reparam(params, MA=False):
    """
    Transforms params to induce stationarity/invertability.
    Parameters
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


def reparam_var_and_d(var, d):
    # Takes the unconstrained parameters as input and casts them to the constrained space
    return np.exp(var), 0.5*sc_autograd.expit(2*d)


def p_gram(x):  # Construct Periodogram
    id = int(np.floor((len(x)-1)/2))
    return np.square(np.abs(x[0:(id+1)]))/(2 * np.pi * len(x))


def f_ARTFIMA(id, phi, theta, var, d, lambda_, n):
    # vectorized spectral density of ARFIMA(len(phi), d, len(theta)) process.
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
        * (np.real(vv2)**2 + np.imag(vv2)**2)
    return f


def f_ARTFIMA(id, phi, theta, var, d, lambda_, var2, n):
    # Note: var2 only exist if there is an ARTFIMA process on an SV model. Otherwise it is zero
    # Note: This is for the Stochastic volatility ARTFIMA - see paper for details
    # vectorized spectral density of ARFIMA(len(phi), d, len(theta)) process.
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
    # NOTE: in this version (v2), the variance is unknown, as is the fractional
    # integration parameter (d). We parameterize the prior in terms of unrestricted parameters for both var and d.
    # We then write the likelihood in terms of the transformed (restricted) parameters

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
            # ARTFIMA model - d is unrestricted, may take any values (except integers, measure zero sets)
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


Nfeval = 1


def callbackF(Xi, nn):
    global Nfeval
    print('{0:4d} {1: 3.6f} {2:.2e}'.format(
        Nfeval, obj(Xi), np.linalg.norm(gr_logp(Xi))))
    Nfeval += 1


def log_likelihood(y, params):
    # This one is only for MCMC so no need for grouping.
    return np.sum(log_density(y, params))


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

    if not os.path.exists(file_control_variates):
        os.makedirs(file_control_variates)

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
    if not os.path.exists(file_control_variates):
        os.makedirs(file_control_variates)
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
                obs[ind], thetaStar[ind], *args, FI_term, TFI_term,  SV_model, w=1, use_w=True)

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


####
# The dataset is divided into groups stored in I. The proxy for group ind is then for the sum of the log-likelihood of the observations indicated by I[ind]
####

# Number of observations in the ungrouped dataset. Same for all cases except ARTFIMA-SV (needed more data for posterior to behave well)
n = 2001

global FI_term, TFI_term, SV_model
case = "ARTFIMA-SV"  # "ARFIMA" # "ARTFIMA-SV"
if case == "ARMA":
    data_set_name = "Vancouver_AR2_MA3_smalldata"
    q = 2
    p = 3
    FI_term = False
    TFI_term = False
    SV_model = False
    if TFI_term:
        assert(FI_term)
elif case == "ARTFIMA":
    data_set_name = "Bromma_AR2_TFI_MA2_smalldata"
    q = 2
    p = 2
    FI_term = True
    TFI_term = True
    SV_model = False
    if TFI_term:
        assert(FI_term)
elif case == "ARFIMA":
    data_set_name = "Simulated_AR2_FI_MA1_smalldata"
    q = 2
    p = 1
    FI_term = True
    TFI_term = False
    SV_model = False
    if TFI_term:
        assert(FI_term)
elif case == "ARTFIMA-SV":
    data_set_name = "Bitcoin_AR1_TFI_MA1_SV_new_smalldata"
    assert(n == 2001)
    #assert(n == 20001)
    q = 1
    p = 1
    FI_term = True
    TFI_term = True
    SV_model = True
    # TODO: Add SV_model thing

    if TFI_term:
        assert(FI_term)


alias = "GARCH-man"

data_set_file_name = '/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Data/%s' % (
    alias, data_set_name) + '.npy'
file_control_variates_main = '/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Data/StoreForRuns/Taylor_Control_variates_%s' % (
    alias, data_set_name)

x = np.load(data_set_file_name)
assert(len(x) == n)


# Data has been generated or read in. Set subsampling options. We will subsample the frequencies, whose indices are ind_range in Dr Salomone's notation
# and are of length floor((n-1)/2)
n = np.int(np.floor((n-1)/2))

if case == "ARTFIMA-SV":
    # Need more observations for posterior to be well-behaved
    GList = [10, 100, 1000]  # [100, 1000, 10000]
else:
    # [1000, 10000, 100000] # , 100000] #[200000, 500000] #1000, 10000, 100000, 1000000] # nbr of groups
    GList = [10, 100, 1000]
sigma2_G_collect = []


def variance_Taylor_control_variates_iterations(G, log_density_group, MCMC_draws, paramsStar, args_whittle,  full_data, file_control_variates=None, MAP_for_Taylor=None,
                                                Taylor_options=None, report=True, flag=' '):
    # This function takes as input a sequence of parameters (generated by say MCMC) and computes the variance based on the Taylor control_variate (true variance by using the full sample). This is to illustrate
    # that the grouping helps the grouped likelihood to be more quadratic

    ########
    # Prepare for proxies
    ########
    # MCMC_draws = MCMC_draws[np.arange(0, 9000, 90)] # s
    mcmc_length = MCMC_draws.shape[0]
    sigma2_LL = np.zeros(mcmc_length)
    Taylor_order, singleTheta = Taylor_options['Taylor_order'], Taylor_options['singleTheta']
    x_init = MCMC_draws[0, :]  # Just set to something. Not really needed

    if full_data:
        # No grouping
        thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star = prepare_control_variates_Taylor_no_grouping(
            ind_range, Taylor_order, log_density, args_whittle, x_init, singleTheta, file_control_variates=file_control_variates, MAP_singleTheta=MAP_for_Taylor, bnds=None)
    else:
        thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star, success = prepare_control_variates_Taylor(ind_range, I, Taylor_order, log_density,
                                                                                                                   args_whittle, x_init, file_control_variates=file_control_variates, MAP_singleTheta=MAP_for_Taylor, singleTheta=singleTheta, bnds=None)

    if singleTheta:
        sum_quants = np.sum(sum_dens_at_Star), np.sum(
            sum_grad_at_Star, axis=0), np.sum(sum_Hess_at_Star, axis=0)
    else:
        sum_quants = None

    x_init = MCMC_draws[0]

    # Compute the variance at the proposed draws.
    bar = progressbar.progressbar(range(mcmc_length))
    uProp = np.arange(G)  # range(G)
    for i in bar:

        if i % 100 == 0:
            print("\n Computing variance for iteration %s out of 10000" % i)

        prop = MCMC_draws[i]

        # Estimate likelihood at proposed point
        q_k_sub, q_sum = control_variates_Taylor(
            prop, uProp, thetaStar, sum_dens_at_Star, sum_grad_at_Star, sum_Hess_at_Star,  singleTheta, sum_quants)
        if full_data:
            l_k_sub = log_density(y,  prop, *args_whittle)
        else:
            l_k_sub = log_density_groups(
                y, itemgetter(*uProp)(I), prop, I_pg, p, q)
        # Note: G is the new population size
        sigma2_LL[i] = G**2*np.var(l_k_sub - q_k_sub, ddof=0)/m

    return sigma2_LL


print('--------------------------------------------------------------')

for G in GList:

    file_control_variates = file_control_variates_main + '.npyG%s/' % G
    print('Running for G = %s' % G)

    groupSize = np.int(n/G)

    # I = [np.arange(start, start + groupSize) for start in np.arange(0, n, groupSize)] # Every element in list contains groupSize indices
    # Here every group contains one observation from each frequency band. Every element in list contains groupSize indices
    I = [item + np.arange(0, n, G) for item in range(G)]

    control_variate_type = "Taylor"

    if control_variate_type == "Taylor":
        # Settings for the control variates based on Taylor approximations
        Taylor_order = 2
        singleTheta = True
        Taylor_options = {'Taylor_order': Taylor_order,
                          'singleTheta': singleTheta}
        coreset_options = None

    # Take 20% of data for all cases
    if G == 10:
        m = 2
    elif G == 100:
        m = 20
    elif G == 1000:
        m = 200
    elif G == 10000:
        m = 2000

    ind_range = np.arange(1, int(np.floor((len(x)-1)/2)) + 1)
    # compute periodogram for Fourier frequencies : O(n log n)
    I_pg = p_gram(fft(x))
    y = ind_range  # Data are now the frequencies

    args_whittle = (I_pg, p, q)

    # Reading in MAP and MCMC draws (both must be created using the main program)
    if case == "ARMA":
        file_name = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/MAP_ARMA_Vancouver_AR2_MA3_smalldata_n1000_AR2_MA3_seed1.npy" % alias
        file_name_MCMC = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARMA/MCMC_ARMA_dataVancouver_AR2_MA3_smalldata_n1000_AR2_MA3_seed1_MCMCiter10000.pkl" % alias
    elif case == "ARTFIMA":
        file_name = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/MAP_ARTFIMA_Bromma_AR2_TFI_MA2_smalldata_n1000_AR2_MA2_seed1.npy" % alias
        file_name_MCMC = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARTFIMA/MCMC_ARTFIMA_dataBromma_AR2_TFI_MA2_smalldata_n1000_AR2_MA2_seed1_MCMCiter10000.pkl" % alias
    elif case == "ARFIMA":
        file_name = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/MAP_ARFIMA_Simulated_AR2_FI_MA1_smalldata_n1000_AR2_MA1_seed1.npy" % alias
        file_name_MCMC = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARFIMA/MCMC_ARFIMA_dataSimulated_AR2_FI_MA1_smalldata_n1000_AR2_MA1_seed1_MCMCiter10000.pkl" % alias
    elif case == "ARTFIMA-SV":
        file_name = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/MAP_ARTFIMA_SV_Bitcoin_AR1_TFI_MA1_SV_new_smalldata_n1000_AR1_MA1_seed1.npy" % alias
        file_name_MCMC = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARTFIMA_SV/Taylor/2020-02-06/18:26/Pseudo_marginal_results.pkl" % alias
    f = open(file_name_MCMC, "rb")
    loadDict = pickle.load(f)
    f.close()

    MAP_and_paramsStar = np.load(file_name)
    MAP = MAP_and_paramsStar[:, 0]
    paramsStar = MAP_and_paramsStar[:, 1]

    print('MAP', MAP)
    print('Likelihood mode', paramsStar)

    MCMC_draws = loadDict['samples']
    if G == n:
        full_data = True
    else:
        full_data = False

    sigma2_G = variance_Taylor_control_variates_iterations(G, log_density_group, MCMC_draws, paramsStar, args_whittle, full_data, file_control_variates, MAP_for_Taylor=paramsStar,
                                                           Taylor_options=Taylor_options, report=True, flag=' ')

    sigma2_G_collect.append(sigma2_G)


fig = plt.figure()
sigma2_LL_groupingG10 = sigma2_G_collect[0][1:]  # [1::10]
sigma2_LL_groupingG100 = sigma2_G_collect[1][1:]  # [1::10]
sigma2_LL_no_grouping = sigma2_G_collect[2][1:]  # [1::10]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(sigma2_LL_groupingG10, color='blue', lw=0.6)
ax.plot(sigma2_LL_groupingG100, color='green', lw=0.6)
ax.plot(sigma2_LL_no_grouping, color='red', lw=0.6)
ax.set_yscale('log')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
G10, = plt.plot(sigma2_LL_no_grouping/sigma2_LL_groupingG10,
                color='blue', lw=0.5)  # , label = r'$|\mathcal{G}|=100$')
G100, = plt.plot(sigma2_LL_no_grouping/sigma2_LL_groupingG100,
                 color='green', lw=0.5)  # , label = r'$|\mathcal{G}|=10$')
G1000, = plt.plot(np.arange(0, len(sigma2_LL_no_grouping)), np.ones(len(
    sigma2_LL_no_grouping)), color='red')  # , label = r'$|\mathcal{G}|=1$', lw=0.5)

# NOTE: for case ARTFIMA-SV the above names should be G100, G1000, G10000, but leave as is for convenience.
plt.legend([G10, G100, G1000], [r'$|G|=100$', r'$|G|=10$',
           r'$|G|=1$'], loc="upper left", prop={'size': 14})


# Save
file_name = path + 'sigma2_different_Gs_%s_ICML.pkl' % case
saveDict = {'sigma2_Gs': sigma2_G_collect, 'Gs': GList}
f = open(file_name, "wb")
pickle.dump(saveDict, f)
f.close()

plt.show()
