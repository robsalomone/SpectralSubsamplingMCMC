"""
Figures for Spectral Subsampling MCMC for Stationary Time series data.

Figures of interest:

1. Effects of grouping for all four examples. 
   Option: 'GroupingEffectOnVariance'
1. For all four examples, Relative Computing Time (relative to MCMC) for Taylor and Coreset (coreset only on large dataset, does not make sense to do coreset for small groupsize).
   Option: 'GroupingEffectOnVariance'
2. KDE for Subsampling MCMC, Taylor and Coreset and when possible (for ARMA) the MCMC on the true likelihood. This is a four by 2 (or 3) figure where each row corresponds to one example, and the columns are parameters.
3. As 2. but for all parameters. One figure for each example. These will go to the Supplement.

NOTE 1: To replicate the figures, the main code must be done to produce output for each of the models (see settings in the submission). This file then reads in that output to produce the figures.
NOTE 2: Obviously, the file paths in this program need to be changed.
"""
from __future__ import division
import autograd.scipy.special as sc_autograd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import sys
import os
import copy
import pandas as pd
import statsmodels.api as sm
import rpy2.robjects as robjects
import pickle
import numpy as np
import scipy.stats as sps
import scipy.special as sc
import random

alias = "GARCH-man"

which_plots = ['GroupingEffectOnVariance', 'RCT_plots',
               'KDEs_all', 'KDEs_selected', 'Spectral_plots']

plotSaveDir = '/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/FigsICMLPaper/' % alias

if not os.path.exists(plotSaveDir):
    os.makedirs(plotSaveDir)

set_font_size = 22  # 20 #10
set_tick_size = 22
set_title_size = 18  # 35 #50

sns.set(style="ticks")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 18

plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=22)
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True

# For computing IF factors using R:
r = robjects.r
r.library('coda')
EffectiveSize = r['effectiveSize']  # Get function of interest


# from d to unconstrained
# z = np.arctanh(d) # 'log'-scales
# from unconstrained to d (-0.5, 0.5)
# d = 0.5*np.tanh(z) # 'exp'-scale


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


def reparam(params, MA=False):
    """
    Transforms params to induce stationarity/invertability.
    Transformation if from partial-autocorrelations to standard parameterization.
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

############
# Figure: Relative variance to study the effect of grouping
############


if 'GroupingEffectOnVariance' in which_plots:

    # 2 x 2 plot
    nrow = 2
    ncol = 2

    titles = ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
    #file_names = ['sigma2_different_Gs_ARMA_ICML.pkl', 'sigma2_different_Gs_ARTFIMA_ICML.pkl', 'sigma2_different_Gs_ARFIMA_ICML.pkl', 'sigma2_different_Gs_ARTFIMA-SV_ICML.pkl']
    file_names = ['sigma2_different_Gs_ARMA_ICML.pkl', 'sigma2_different_Gs_ARTFIMA_ICML.pkl',
                  'sigma2_different_Gs_ARFIMA_ICML.pkl', 'sigma2_different_Gs_ARTFIMA-SV_ICML.pkl']
    file_path = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/" % alias

    fig, ax = plt.subplots(nrow, ncol)
    ind = 0
    for ii in range(nrow):
        for jj in range(ncol):
            ax[ii][jj].spines['top'].set_visible(False)
            ax[ii][jj].spines['right'].set_visible(False)
            ax[ii][jj].get_xaxis().tick_bottom()
            ax[ii][jj].get_yaxis().tick_left()

            file_name = file_path + file_names[ind]
            f = open(file_name, "rb")
            loadDict = pickle.load(f)
            f.close()

            sigma2_G_collect = loadDict['sigma2_Gs']
            GList = loadDict['Gs']
            mcmc_length = len(sigma2_G_collect[0])
            burnIn = 1001
            sigma2_LL_groupingG10 = sigma2_G_collect[0][burnIn:]
            sigma2_LL_groupingG100 = sigma2_G_collect[1][burnIn:]
            sigma2_LL_no_grouping = sigma2_G_collect[2][burnIn:]
            iteration = range(burnIn - 1, mcmc_length - 1)

            ax[ii][jj].set_ylabel(
                r'$\sigma^2_{\widehat{\ell}_{\mathrm{diff}}}/\sigma^2_{\widehat{\ell}_{\mathrm{gr}}}$')
            ax[ii][jj].set_xlabel('MCMC iteration')
            ax[ii][jj].set_title(titles[ind], fontsize=set_title_size)
            G10, = ax[ii][jj].plot(iteration, sigma2_LL_no_grouping/sigma2_LL_groupingG10,
                                   color='blue', lw=0.5, alpha=1)  # , label = r'$|G|=100$')
            G100, = ax[ii][jj].plot(iteration, sigma2_LL_no_grouping/sigma2_LL_groupingG100,
                                    color='green', lw=0.5, alpha=1)  # , label = r'$|G|=10$')
            G1000, = ax[ii][jj].plot(iteration, np.ones(len(
                sigma2_LL_no_grouping)), color='red', lw=0.5, alpha=1)  # , label = r'$|G|=1$', lw=0.5)

            ax[ii][jj].set_xticks([1000, 5000, 10000])
            pow_ = 0.3
            if ind == 0:
                #ax[ii][jj].set_ylim([0, 30])
                ax[ii][jj].set_ylim([10**(-.5*pow_), 400])
                ax[ii][jj].set_yticks([10**(-pow_), 1, 10])
            elif ind == 1:
                #ax[ii][jj].set_ylim([0, 20])
                ax[ii][jj].set_ylim([10**(-.5*pow_), 20])
                ax[ii][jj].set_yticks([10**(-pow_), 1, 5, 10, 15])

            elif ind == 2:
                #ax[ii][jj].set_ylim([0, 20])
                ax[ii][jj].set_ylim([10**(-.5*pow_), 20])
                ax[ii][jj].set_yticks([10**(-pow_), 1, 5, 10, 15])

            elif ind == 3:
                #ax[ii][jj].set_ylim([0, 30])
                ax[ii][jj].set_ylim([10**(-.5*pow_), 30])
                ax[ii][jj].set_yticks([10**(-pow_), 1, 5, 10, 15, 20])

            if ind == 0:
                ax[ii][jj].legend((G10, G100, G1000), (r'$|G|=100$', r'$|G|=10$', r'$|G|=1$'), prop={
                                  'size': set_tick_size-12}, loc="upper right")

            try:
                plt.tight_layout()
            except ValueError:
                print("Plots are corrupted")

            ax[ii][jj].set_yscale('log')
            ind = ind + 1

    fig.subplots_adjust(hspace=0.8, wspace=0.4)
    plt.savefig(plotSaveDir + "GroupingEffectOnVariance.pdf")

########################################
# RCTs for all examples 2 x 2.
########################################
if 'RCT_plots' in which_plots:

    ####
    # 2 x 2 plot
    nrow = 2
    ncol = 2

    set_tick_size = 13

    file_path = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/" % alias
    titles = ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
    file_names_MCMC = ['ARMA/MCMC_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl',
                       'ARTFIMA/MCMC_ARTFIMA_dataBromma_AR2_TFI_MA2_n225000_AR2_MA2_seed1_MCMCiter10000.pkl',
                       'ARFIMA/MCMC_ARFIMA_dataSimulated_AR2_FI_MA1_n2500000_AR2_MA1_seed1_MCMCiter10000.pkl',
                       'ARTFIMA_SV/MCMC_ARTFIMA_SV_dataBitcoin_AR1_TFI_MA1_SV_new_n500000_AR1_MA1_seed1_MCMCiter10000.pkl']

    file_names_MCMC = [file_path + file_names_MCMC[item] for item in range(4)]

    file_names_pseudo_MCMC_Taylor = ['ARMA/Taylor/2020-01-30/20:40/Pseudo_marginal_results.pkl',
                                     'ARTFIMA/Taylor/2020-01-30/22:22/Pseudo_marginal_results.pkl',
                                     'ARFIMA/Taylor/2020-02-01/12:13/Pseudo_marginal_results.pkl',
                                     'ARTFIMA_SV/Taylor/2020-02-06/16:10/Pseudo_marginal_results.pkl']
    file_names_pseudo_MCMC_Taylor = [
        file_path + file_names_pseudo_MCMC_Taylor[item] for item in range(4)]

    file_names_pseudo_MCMC_Coreset = [None,
                                      'ARTFIMA/coreset/2020-01-31/00:29/Pseudo_marginal_results.pkl',
                                      'ARFIMA/coreset/2020-02-01/06:35/Pseudo_marginal_results.pkl',
                                      'ARTFIMA_SV/coreset/2020-02-06/17:31/Pseudo_marginal_results.pkl']

    file_names_pseudo_MCMC_Coreset = [
        None] + [file_path + file_names_pseudo_MCMC_Coreset[item] for item in range(1, 4)]

    fig, ax = plt.subplots(nrow, ncol)
    ind = 0
    for ii in range(nrow):
        for jj in range(ncol):

            # Specific for each model
            if ind == 0:
                # ARMA(2,3)
                n = 22000
                G = 1000
                m = 20
                groupSize = n/G
                mcmc_length = 10000
                M = 200
                q = 2
                p = 3
            elif ind == 1:
                # ARTFIMA(2,2)
                n = 225000
                G = 1000
                m = 10
                groupSize = n/G
                mcmc_length = 10000
                M = 200
                q = 2
                p = 2

            elif ind == 2:
                # ARFIMA(2,1)
                n = 2500000
                G = 1000
                m = 10
                groupSize = n/G
                mcmc_length = 10000
                M = 200
                q = 2
                p = 1
            elif ind == 3:
                # ARTFIMA-SV(1,1)
                n = 500000
                G = 1000
                m = 10
                groupSize = n/G
                mcmc_length = 10000
                M = 200
                q = 1
                p = 1

            ax[ii][jj].spines['top'].set_visible(False)
            ax[ii][jj].spines['right'].set_visible(False)
            ax[ii][jj].get_xaxis().tick_bottom()
            ax[ii][jj].get_yaxis().tick_left()
            dict_MCMC = pickle.load(open(file_names_MCMC[ind], 'rb'))
            dict_pseudo_MCMC_Taylor = pickle.load(
                open(file_names_pseudo_MCMC_Taylor[ind], 'rb'))

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                dict_pseudo_MCMC_Coreset = pickle.load(
                    open(file_names_pseudo_MCMC_Coreset[ind], 'rb'))

            burnIn = 1000
            samples_MCMC = dict_MCMC['samples'][burnIn + 1:, :]
            samples_pseudo_MCMC_Taylor = dict_pseudo_MCMC_Taylor['samples'][burnIn + 1:, :]

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                samples_pseudo_MCMC_Coreset = dict_pseudo_MCMC_Coreset['samples'][burnIn + 1:, :]

            # Cast partial auto-correlation to ordinary parameterization.
            samples_MCMC[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_MCMC[:, :q]))
            samples_pseudo_MCMC_Taylor[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Taylor[:, :q]))

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                samples_pseudo_MCMC_Coreset[:, :q] = np.vstack(
                    map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Coreset[:, :q]))

            samples_MCMC[:, q:(
                q + p)] = np.vstack(map(lambda x: reparam(x, MA=True), samples_MCMC[:, q:(q + p)]))
            samples_pseudo_MCMC_Taylor[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Taylor[:, q:(q + p)]))
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                samples_pseudo_MCMC_Coreset[:, q:(q + p)] = np.vstack(
                    map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Coreset[:, q:(q + p)]))
            # End Cast partial auto-correlation

            # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
            if ind == 0:
                # ARMA
                sigma2_ind_from_last = -1
                d_ind_from_last = None
                lambda_ind_from_last = None
                var2_ind_from_last = None
            elif ind == 1:
                # ARTFIMA
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = -3
                var2_ind_from_last = None
            elif ind == 2:
                # ARFIMA
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = None
                var2_ind_from_last = None
            elif ind == 3:
                # ARTFIMA-SV
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = -3
                var2_ind_from_last = -4

            # d parameter
            if d_ind_from_last is not None and ind == 2:
                samples_MCMC[:, d_ind_from_last] = 0.5 * \
                    np.tanh(samples_MCMC[:, d_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, d_ind_from_last] = 0.5 * \
                    np.tanh(samples_pseudo_MCMC_Taylor[:, d_ind_from_last])
                samples_pseudo_MCMC_Coreset[:, d_ind_from_last] = 0.5 * \
                    np.tanh(samples_pseudo_MCMC_Coreset[:, d_ind_from_last])
            # NOTE: For ind = 1 and ind = 3 the parameterization is just d (unrestricted).

            # variance param (all models have it)
            samples_MCMC[:, sigma2_ind_from_last] = np.exp(
                samples_MCMC[:, sigma2_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last])
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last])

            # Lambda
            if lambda_ind_from_last is not None:
                samples_MCMC[:, lambda_ind_from_last] = np.exp(
                    samples_MCMC[:, lambda_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last])
                samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last])

            # var2 parameter

            if var2_ind_from_last is not None:
                samples_MCMC[:, var2_ind_from_last] = np.exp(
                    samples_MCMC[:, var2_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, var2_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Taylor[:, var2_ind_from_last])
                samples_pseudo_MCMC_Coreset[:, var2_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Coreset[:, var2_ind_from_last])  # Coreset included for this example

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                coreset_sizes = dict_pseudo_MCMC_Coreset['coreset_sizes']

            pVar = samples_MCMC.shape[1]

            # Cost:
            cost_MCMC = mcmc_length*n
            # all data once, then groupSize x number of samples groups
            cost_pseudo_MCMC_Taylor = n + mcmc_length*m*groupSize
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                # Constructing coreset for every group, then evaluating the subsample plus evaluating the coreset. NOTE: One could argue that if we run in paralllel the construction of the coreset is M*n/G
                cost_pseudo_MCMC_Coreset = M*n + mcmc_length * \
                    (m*groupSize + np.sum(coreset_sizes))

            IF_MCMC = np.zeros(pVar)
            IF_pseudo_MCMC_Taylor = np.zeros(pVar)

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                IF_pseudo_MCMC_Coreset = np.zeros(pVar)

            CT_MCMC = np.zeros(pVar)
            CT_pseudo_MCMC_Taylor = np.zeros(pVar)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                CT_pseudo_MCMC_Coreset = np.zeros(pVar)

            for j in range(pVar):
                # Compute effective samples
                PosteriorDrawsBetaRFormat = nparray2rmatrix(samples_MCMC[:, j])
                ESS_MCMC = np.array(EffectiveSize(PosteriorDrawsBetaRFormat))

                PosteriorDrawsBetaRFormat = nparray2rmatrix(
                    samples_pseudo_MCMC_Taylor[:, j])
                ESS_pseudo_MCMC_Taylor = np.array(
                    EffectiveSize(PosteriorDrawsBetaRFormat))

                if file_names_pseudo_MCMC_Coreset[ind] is not None:
                    PosteriorDrawsBetaRFormat = nparray2rmatrix(
                        samples_pseudo_MCMC_Coreset[:, j])
                    ESS_pseudo_MCMC_Coreset = np.array(
                        EffectiveSize(PosteriorDrawsBetaRFormat))

                IF_MCMC[j] = (len(samples_MCMC)/ESS_MCMC[0])
                IF_pseudo_MCMC_Taylor[j] = (
                    len(samples_pseudo_MCMC_Taylor)/ESS_pseudo_MCMC_Taylor)
                if file_names_pseudo_MCMC_Coreset[ind] is not None:
                    IF_pseudo_MCMC_Coreset[j] = (
                        len(samples_pseudo_MCMC_Coreset)/ESS_pseudo_MCMC_Coreset)

            CT_MCMC = cost_MCMC*IF_MCMC
            CT_pseudo_MCMC_Taylor = cost_pseudo_MCMC_Taylor*IF_pseudo_MCMC_Taylor

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                CT_pseudo_MCMC_Coreset = cost_pseudo_MCMC_Coreset*IF_pseudo_MCMC_Coreset

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                RCT_pseudo_MCMC_Coreset = CT_MCMC/CT_pseudo_MCMC_Coreset

            RCT_pseudo_MCMC_Taylor = CT_MCMC/CT_pseudo_MCMC_Taylor

            if ind == 0:
                # Plot without coreset.
                RCT_pseudo_MCMC_Taylor = CT_MCMC/CT_pseudo_MCMC_Taylor

                ax[ii][jj].set_title(titles[ind], fontsize=set_title_size)

                # Weights for CT
                collist = ['#ffff33', '#377eb8', 'r']
                collist = ['#4daf4a', '#377eb8', 'r']  # NOTE: Green, blue, red

                h0 = RCT_pseudo_MCMC_Taylor

                NN = len(h0)

                ind_ = np.arange(NN)  # the x locations for the groups
                width = 0.15       # the width of the bars

                rects = ax[ii][jj].bar(
                    2*ind_, h0, width, color=collist[1], edgecolor=collist[1])

                # add some text for labels, title and axes ticks
                ax[ii][jj].set_ylabel(
                    r'$\mathrm{RCT}$', fontsize=set_font_size)

                ax[ii][jj].set_xlim([-1, 2*NN])

                ax[ii][jj].set_ylim([0, 100])

                ax[ii][jj].set_xticks(ind_*2)

                #ax[ii][jj].legend((rects[0]), ('sisen'), prop={'size' : set_tick_size-4}, loc = "upper right")

                ax[ii][jj].set_xticklabels(
                    (r'$\phi_{1}$', r'$\phi_{2}$', r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$', r'$\sigma^2$'))
                ax[ii][jj].tick_params(
                    axis='both', which='major', labelsize=set_tick_size)

            else:

                ax[ii][jj].set_title(titles[ind], fontsize=set_title_size)

                # Weights for CT
                collist = ['#ffff33', '#377eb8', 'r']
                collist = ['#4daf4a', '#377eb8', 'r']  # NOTE: Green, blue, red

                h0 = RCT_pseudo_MCMC_Coreset
                h1 = RCT_pseudo_MCMC_Taylor

                NN = len(h0)

                ind_ = np.arange(NN)  # the x locations for the groups
                width = 0.15       # the width of the bars

                rects1 = ax[ii][jj].bar(
                    ind_, h0, width, color=collist[0], edgecolor=collist[0])
                rects2 = ax[ii][jj].bar(
                    ind_+width, h1, width, color=collist[1], edgecolor=collist[1])

                # add some text for labels, title and axes ticks
                # NOTE: More parameters, need to change this
                ax[ii][jj].set_ylabel(
                    r'$\mathrm{RCT}$', fontsize=set_font_size)

                if ind == 1:
                    ax[ii][jj].legend((rects1[0], rects2[0]), (r'$\mathrm{Coreset}$', r'$\mathrm{Taylor}$'), prop={
                                      'size': set_tick_size-4}, loc="upper right")
                    ax[ii][jj].set_ylim([0, 180])
                    ax[ii][jj].set_xlim([-1, NN + 1])

                else:
                    ax[ii][jj].set_ylim([0, 130])
                    ax[ii][jj].set_xlim([-1, NN + 1])

                ax[ii][jj].set_yticks([0, 25, 50, 75, 100])
                ax[ii][jj].set_xticks(ind_+0.5*width)

                # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
                if ind == 1:
                    ax[ii][jj].set_xticklabels(
                        (r'$\phi_1$', r'$\phi_2$', r'$\theta_1$', r'$\theta_2$', r'$\lambda$', r'$\sigma^2$', r'$d$'))

                elif ind == 2:
                    ax[ii][jj].set_xticklabels(
                        (r'$\phi_1$', r'$\phi_2$', r'$\theta_1$',  r'$\sigma^2$', r'$d$'))
                elif ind == 3:
                    ax[ii][jj].set_xticklabels(
                        (r'$\phi_1$', r'$\theta_1$', r'$\sigma_{\epsilon}^2$', r'$\lambda$', r'$\sigma^2$', r'$d$'))

                ax[ii][jj].tick_params(
                    axis='both', which='major', labelsize=set_tick_size)

            ind = ind + 1

    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig(plotSaveDir + "RCT_all_models.pdf")


if 'KDEs_selected' in which_plots:

    bandwidthKDE = 0.5
    ####
    # 4 x 3 plot
    nrow = 4
    ncol = 3

    # Params to plot for each example. The full list
    # ARMA(2,3) ind 0
    # ARTFIMA(2,2) ind 1
    # ARFIMA(2,1) ind 2
    # ARTFIMA-SV(1,1) ind 3
    param_names = [[r'$\phi_{1}$', r'$\phi_{2}$', r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$', r'$\sigma^2$'],
                   [r'$\phi_1$', r'$\phi_2$', r'$\theta_1$',
                       r'$\theta_2$', r'$\lambda$', r'$\sigma^2$', r'$d$'],
                   [r'$\phi_1$', r'$\phi_2$', r'$\theta_1$',  r'$\sigma^2$', r'$d$'],
                   [r'$\phi_1$', r'$\theta_1$', r'$\sigma_{\epsilon}^2$', r'$\lambda$', r'$\sigma^2$', r'$d$']]
    density_names = [[r'$\pi(\phi_{1})$', r'$\pi(\phi_{2})$', r'$\pi(\theta_{1})$', r'$\pi(\theta_{2})$', r'$\pi(\theta_{3})$', r'$\pi(\sigma^2)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\phi_2)$', r'$\pi(\theta_1)$',
                      r'$\pi(\theta_2)$', r'$\pi(\lambda)$', r'$\pi(\sigma^2)$', r'$\pi(d)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\phi_2)$', r'$\pi(\theta_1)$',
                      r'$\pi(\sigma^2)$', r'$\pi(d)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\theta_1)$', r'$\pi(\sigma_{\epsilon}^2)$', r'$\pi(\lambda)$', r'$\pi(\sigma^2)$', r'$\pi(d)$']]

    paramsToPlot = [[1, 4, 5], [0, 4, 6], [0, 3, 4], [0, 2, 5]]

    set_tick_size = 7

    file_path = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/" % alias
    titles = ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
    file_names_MCMC = ['ARMA/MCMC_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl',
                       'ARTFIMA/MCMC_ARTFIMA_dataBromma_AR2_TFI_MA2_n225000_AR2_MA2_seed1_MCMCiter10000.pkl',
                       'ARFIMA/MCMC_ARFIMA_dataSimulated_AR2_FI_MA1_n2500000_AR2_MA1_seed1_MCMCiter10000.pkl',
                       'ARTFIMA_SV/MCMC_ARTFIMA_SV_dataBitcoin_AR1_TFI_MA1_SV_new_n500000_AR1_MA1_seed1_MCMCiter10000.pkl']
    file_names_MCMC = [file_path + file_names_MCMC[item] for item in range(4)]

    file_names_pseudo_MCMC_Taylor = ['ARMA/Taylor/2020-01-30/20:40/Pseudo_marginal_results.pkl',
                                     'ARTFIMA/Taylor/2020-01-30/22:22/Pseudo_marginal_results.pkl',
                                     'ARFIMA/Taylor/2020-02-01/12:13/Pseudo_marginal_results.pkl',
                                     'ARTFIMA_SV/Taylor/2020-02-06/16:10/Pseudo_marginal_results.pkl']
    file_names_pseudo_MCMC_Taylor = [
        file_path + file_names_pseudo_MCMC_Taylor[item] for item in range(4)]

    file_names_pseudo_MCMC_Coreset = [None,
                                      'ARTFIMA/coreset/2020-01-31/00:29/Pseudo_marginal_results.pkl',
                                      'ARFIMA/coreset/2020-02-01/06:35/Pseudo_marginal_results.pkl',
                                      'ARTFIMA_SV/coreset/2020-02-06/17:31/Pseudo_marginal_results.pkl']

    file_names_pseudo_MCMC_Coreset = [
        None] + [file_path + file_names_pseudo_MCMC_Coreset[item] for item in range(1, 4)]

    fig, ax = plt.subplots(nrow, ncol)
    for ind in range(nrow):
        # things that are constant for each parameter within a model

        # Specific for each model
        if ind == 0:
            # ARMA(2,3)
            q = 2
            p = 3
        elif ind == 1:
            # ARTFIMA(2,2)
            q = 2
            p = 2

        elif ind == 2:
            # ARFIMA(2,1)
            q = 2
            p = 1
        elif ind == 3:
            # ARTFIMA-SV(1,1)
            q = 1
            p = 1

        dict_MCMC = pickle.load(open(file_names_MCMC[ind], 'rb'))
        dict_pseudo_MCMC_Taylor = pickle.load(
            open(file_names_pseudo_MCMC_Taylor[ind], 'rb'))

        if ind == 0:
            # Time domain likelihood:
            dict_time_domain = pickle.load(open(
                '/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARMA/MCMC_true_likelihood_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl' % alias, 'rb'))

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            dict_pseudo_MCMC_Coreset = pickle.load(
                open(file_names_pseudo_MCMC_Coreset[ind], 'rb'))

        burnIn = 1000
        samples_MCMC = dict_MCMC['samples'][burnIn + 1:, :]
        samples_pseudo_MCMC_Taylor = dict_pseudo_MCMC_Taylor['samples'][burnIn + 1:, :]
        if ind == 0:
            samples_time_domain_MCMC = dict_time_domain['samples'][burnIn + 1:, :]

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset = dict_pseudo_MCMC_Coreset['samples'][burnIn + 1:, :]

        # Cast partial auto-correlation to ordinary parameterization.
        samples_MCMC[:, :q] = np.vstack(
            map(lambda x: reparam(x, MA=False), samples_MCMC[:, :q]))
        samples_pseudo_MCMC_Taylor[:, :q] = np.vstack(
            map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Taylor[:, :q]))
        if ind == 0:
            samples_time_domain_MCMC[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_time_domain_MCMC[:, :q]))
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Coreset[:, :q]))

        samples_MCMC[:, q:(
            q + p)] = np.vstack(map(lambda x: reparam(x, MA=True), samples_MCMC[:, q:(q + p)]))
        samples_pseudo_MCMC_Taylor[:, q:(q + p)] = np.vstack(
            map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Taylor[:, q:(q + p)]))
        if ind == 0:
            samples_time_domain_MCMC[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_time_domain_MCMC[:, q:(q + p)]))
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Coreset[:, q:(q + p)]))
        # End Cast partial auto-correlation

        # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
        if ind == 0:
            # ARMA
            sigma2_ind_from_last = -1
            d_ind_from_last = None
            lambda_ind_from_last = None
            var2_ind_from_last = None
        elif ind == 1:
            # ARTFIMA
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = -3
            var2_ind_from_last = None
        elif ind == 2:
            # ARFIMA
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = None
            var2_ind_from_last = None
        elif ind == 3:
            # ARTFIMA-SV
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = -3
            var2_ind_from_last = -4

        # d parameter
        if d_ind_from_last is not None and ind == 2:
            samples_MCMC[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_MCMC[:, d_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_pseudo_MCMC_Taylor[:, d_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_pseudo_MCMC_Coreset[:, d_ind_from_last])
        # NOTE: For ind = 1 and ind = 3 the parameterization is just d (unrestricted).

        # variance param (all models have it)
        samples_MCMC[:, sigma2_ind_from_last] = np.exp(
            samples_MCMC[:, sigma2_ind_from_last])
        samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last] = np.exp(
            samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last])
        if ind == 0:
            samples_time_domain_MCMC[:, sigma2_ind_from_last] = np.exp(
                samples_time_domain_MCMC[:, sigma2_ind_from_last])
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last])

        # Lambda
        if lambda_ind_from_last is not None:
            samples_MCMC[:, lambda_ind_from_last] = np.exp(
                samples_MCMC[:, lambda_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last])

        # var2 parameter
        if var2_ind_from_last is not None:
            samples_MCMC[:, var2_ind_from_last] = np.exp(
                samples_MCMC[:, var2_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, var2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, var2_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, var2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, var2_ind_from_last])  # Coreset included for this example

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            coreset_sizes = dict_pseudo_MCMC_Coreset['coreset_sizes']

        # end things that are constant...
        for jj, param in enumerate(paramsToPlot[ind]):

            ax[ind][jj].spines['top'].set_visible(False)
            ax[ind][jj].spines['right'].set_visible(False)
            ax[ind][jj].get_xaxis().tick_bottom()
            ax[ind][jj].get_yaxis().tick_left()

            # BEGIN PLOT three of the parameters.
            theta_MCMC = samples_MCMC[:, param]
            theta_pseudo_MCMC_Taylor = samples_pseudo_MCMC_Taylor[:, param]
            if ind == 0:
                theta_time_domain_MCMC = samples_time_domain_MCMC[:, param]

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                theta_pseudo_MCMC_Coreset = samples_pseudo_MCMC_Coreset[:, param]

            min_x_MCMC = np.min(theta_MCMC)
            max_x_MCMC = np.max(theta_MCMC)
            min_x_pseudo_MCMC_Taylor = np.min(theta_pseudo_MCMC_Taylor)
            max_x_pseudo_MCMC_Taylor = np.max(theta_pseudo_MCMC_Taylor)
            if ind == 0:
                min_x_time_domain_MCMC = np.min(theta_time_domain_MCMC)
                max_x_time_domain_MCMC = np.max(theta_time_domain_MCMC)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                min_x_pseudo_MCMC_Coreset = np.min(theta_pseudo_MCMC_Coreset)
                max_x_pseudo_MCMC_Coreset = np.max(theta_pseudo_MCMC_Coreset)

            # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
            # Which KDEs to we plot for different examples?
            if ind == 0:
                # ARMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_time_domain_MCMC])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, min_x_time_domain_MCMC])

            elif ind == 1:
                # ARTFIMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            elif ind == 2:
                # ARFIMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            elif ind == 3:
                # ARTFIMA-SV
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            x_grid = np.linspace(min_x, max_x, 500)

            kde_MCMC = sps.gaussian_kde(theta_MCMC)
            kde_pseudo_MCMC_Taylor = sps.gaussian_kde(theta_pseudo_MCMC_Taylor)
            if ind == 0:
                kde_time_domain_MCMC = sps.gaussian_kde(theta_time_domain_MCMC)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                kde_pseudo_MCMC_Coreset = sps.gaussian_kde(
                    theta_pseudo_MCMC_Coreset)

            if bandwidthKDE is not None:
                kde_MCMC.set_bandwidth(bandwidthKDE)
                kde_pseudo_MCMC_Taylor.set_bandwidth(bandwidthKDE)
                if ind == 0:
                    kde_time_domain_MCMC.set_bandwidth(bandwidthKDE)
                if file_names_pseudo_MCMC_Coreset[ind] is not None:
                    kde_pseudo_MCMC_Coreset.set_bandwidth(bandwidthKDE)

            pdfvals_MCMC = kde_MCMC(x_grid)
            pdfvals_pseudo_MCMC_Taylor = kde_pseudo_MCMC_Taylor(x_grid)
            if ind == 0:
                pdfvals_time_domain_MCMC = kde_time_domain_MCMC(x_grid)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                pdfvals_pseudo_MCMC_Coreset = kde_pseudo_MCMC_Coreset(x_grid)

            rects1 = ax[ind][jj].plot(
                x_grid, pdfvals_MCMC, 'k', linewidth=2, label=r'$\mathrm{MCMC}$')
            rects2 = ax[ind][jj].plot(x_grid, pdfvals_pseudo_MCMC_Taylor, 'magenta',
                                      linewidth=2, linestyle='--', label=r'$\mathrm{Taylor}$')
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                rects3 = ax[ind][jj].plot(x_grid, pdfvals_pseudo_MCMC_Coreset,
                                          'orange', linewidth=2, linestyle='--', label=r'$\mathrm{Coreset}$')
            if ind == 0:
                rects4 = ax[ind][jj].plot(x_grid, pdfvals_time_domain_MCMC, 'green',
                                          linewidth=2, linestyle='--', label=r'$\mathrm{Coreset}$')

            ax[ind][jj].set_yticks([])
            ax[ind][jj].set_ylabel(density_names[ind][param], fontsize=10)
            ax[ind][jj].set_title(param_names[ind][param], fontsize=12)

            # Legend
            if jj == 0:
                if ind == 0:
                    ax[ind][jj].legend((rects1[0], rects2[0], rects4[0]), (
                        r'$\mathrm{MCMC}$', r'$\mathrm{Taylor}$', r'$\mathrm{Gauss-MCMC}$'), prop={'size': 6}, loc="upper right")
                    ax[ind][jj].set_xlim([min_x, max_x + 0.75])
                else:
                    # for all other inds we include the coreset, but no time domain MCMC
                    ax[ind][jj].legend((rects1[0], rects2[0], rects3[0]), (
                        r'$\mathrm{MCMC}$', r'$\mathrm{Taylor}$', r'$\mathrm{Coreset}$'), prop={'size': 6}, loc="upper right")

                    if ind == 1:
                        ax[ind][jj].set_xlim([min_x, max_x + 0.02])
                    elif ind == 2:
                        ax[ind][jj].set_xlim([min_x, max_x + 0.007])
                    elif ind == 3:
                        ax[ind][jj].set_xlim([min_x, max_x + 0.25])

            try:
                plt.tight_layout()
            except ValueError:
                print("Plots are corrupted")

    #fig.subplots_adjust(hspace = 0.5, wspace = 0.4)
    plt.savefig(plotSaveDir + "KDE_all_models_selected_params.pdf")


if 'KDEs_all' in which_plots:

    bandwidthKDE = 0.5

    # Makes one figure for each of the examples, using ALL parameters
    ####

    ncol = 1

    # Params to plot for each example. The full list
    # ARMA(2,3) ind 0
    # ARTFIMA(2,2) ind 1
    # ARFIMA(2,1) ind 2
    # ARTFIMA-SV(1,1) ind 3
    param_names = [[r'$\phi_{1}$', r'$\phi_{2}$', r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$', r'$\sigma^2$'],
                   [r'$\phi_1$', r'$\phi_2$', r'$\theta_1$',
                       r'$\theta_2$', r'$\lambda$', r'$\sigma^2$', r'$d$'],
                   [r'$\phi_1$', r'$\phi_2$', r'$\theta_1$',  r'$\sigma^2$', r'$d$'],
                   [r'$\phi_1$', r'$\theta_1$', r'$\sigma_{\epsilon}^2$', r'$\lambda$', r'$\sigma^2$', r'$d$']]
    density_names = [[r'$\pi(\phi_{1})$', r'$\pi(\phi_{2})$', r'$\pi(\theta_{1})$', r'$\pi(\theta_{2})$', r'$\pi(\theta_{3})$', r'$\pi(\sigma^2)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\phi_2)$', r'$\pi(\theta_1)$',
                      r'$\pi(\theta_2)$', r'$\pi(\lambda)$', r'$\pi(\sigma^2)$', r'$\pi(d)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\phi_2)$', r'$\pi(\theta_1)$',
                      r'$\pi(\sigma^2)$', r'$\pi(d)$'],
                     [r'$\pi(\phi_1)$', r'$\pi(\theta_1)$', r'$\pi(\sigma_{\epsilon}^2)$', r'$\pi(\lambda)$', r'$\pi(\sigma^2)$', r'$\pi(d)$']]

    paramsToPlot = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]

    set_tick_size = 7

    file_path = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/" % alias
    titles = ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
    file_names_MCMC = ['ARMA/MCMC_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl',
                       'ARTFIMA/MCMC_ARTFIMA_dataBromma_AR2_TFI_MA2_n225000_AR2_MA2_seed1_MCMCiter10000.pkl',
                       'ARFIMA/MCMC_ARFIMA_dataSimulated_AR2_FI_MA1_n2500000_AR2_MA1_seed1_MCMCiter10000.pkl',
                       'ARTFIMA_SV/MCMC_ARTFIMA_SV_dataBitcoin_AR1_TFI_MA1_SV_new_n500000_AR1_MA1_seed1_MCMCiter10000.pkl']
    file_names_MCMC = [file_path + file_names_MCMC[item] for item in range(4)]

    file_names_pseudo_MCMC_Taylor = ['ARMA/Taylor/2020-01-30/20:40/Pseudo_marginal_results.pkl',
                                     'ARTFIMA/Taylor/2020-01-30/22:22/Pseudo_marginal_results.pkl',
                                     'ARFIMA/Taylor/2020-02-01/12:13/Pseudo_marginal_results.pkl',
                                     'ARTFIMA_SV/Taylor/2020-02-06/16:10/Pseudo_marginal_results.pkl']
    file_names_pseudo_MCMC_Taylor = [
        file_path + file_names_pseudo_MCMC_Taylor[item] for item in range(4)]

    file_names_pseudo_MCMC_Coreset = [None,
                                      'ARTFIMA/coreset/2020-01-31/00:29/Pseudo_marginal_results.pkl',
                                      'ARFIMA/coreset/2020-02-01/06:35/Pseudo_marginal_results.pkl',
                                      'ARTFIMA_SV/coreset/2020-02-06/17:31/Pseudo_marginal_results.pkl']

    file_names_pseudo_MCMC_Coreset = [
        None] + [file_path + file_names_pseudo_MCMC_Coreset[item] for item in range(1, 4)]

    for ind in range(4):  # loop over examples
        nrow = len(paramsToPlot[ind])

        if ind == 0:
            fig, ax = plt.subplots(2, 3)
        elif ind == 1:
            fig, ax = plt.subplots(3, 3)
            ax[2, 1].axis('off')
            ax[2, 2].axis('off')
        elif ind == 2:
            fig, ax = plt.subplots(2, 3)
            ax[1, -1].axis('off')
        elif ind == 3:
            fig, ax = plt.subplots(2, 3)

        # BEGIN PASTE

        if ind == 0:
            pass
        elif ind == 1:
            if jj == 2 and (ll == 1 or ll == 2):
                ax[jj][ll].axis('off')
        elif ind == 2:
            if jj == 1 and ll == 2:
                ax[jj][ll].axis('off')
        elif ind == 3:
            pass

        # END PASTE

        # things that are constant for each parameter within a model

        # Specific for each model
        if ind == 0:
            # ARMA(2,3)
            q = 2
            p = 3
        elif ind == 1:
            # ARTFIMA(2,2)
            q = 2
            p = 2

        elif ind == 2:
            # ARFIMA(2,1)
            q = 2
            p = 1
        elif ind == 3:
            # ARTFIMA-SV(1,1)
            q = 1
            p = 1

        dict_MCMC = pickle.load(open(file_names_MCMC[ind], 'rb'))
        dict_pseudo_MCMC_Taylor = pickle.load(
            open(file_names_pseudo_MCMC_Taylor[ind], 'rb'))

        if ind == 0:
            # Time domain likelihood:
            dict_time_domain = pickle.load(open(
                '/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/ARMA/MCMC_true_likelihood_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl' % alias, 'rb'))

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            dict_pseudo_MCMC_Coreset = pickle.load(
                open(file_names_pseudo_MCMC_Coreset[ind], 'rb'))

        burnIn = 1000
        samples_MCMC = dict_MCMC['samples'][burnIn + 1:, :]
        samples_pseudo_MCMC_Taylor = dict_pseudo_MCMC_Taylor['samples'][burnIn + 1:, :]
        if ind == 0:
            samples_time_domain_MCMC = dict_time_domain['samples'][burnIn + 1:, :]

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset = dict_pseudo_MCMC_Coreset['samples'][burnIn + 1:, :]

        # Cast partial auto-correlation to ordinary parameterization.
        samples_MCMC[:, :q] = np.vstack(
            map(lambda x: reparam(x, MA=False), samples_MCMC[:, :q]))
        samples_pseudo_MCMC_Taylor[:, :q] = np.vstack(
            map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Taylor[:, :q]))
        if ind == 0:
            samples_time_domain_MCMC[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_time_domain_MCMC[:, :q]))
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Coreset[:, :q]))

        samples_MCMC[:, q:(
            q + p)] = np.vstack(map(lambda x: reparam(x, MA=True), samples_MCMC[:, q:(q + p)]))
        samples_pseudo_MCMC_Taylor[:, q:(q + p)] = np.vstack(
            map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Taylor[:, q:(q + p)]))
        if ind == 0:
            samples_time_domain_MCMC[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_time_domain_MCMC[:, q:(q + p)]))
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Coreset[:, q:(q + p)]))
        # End Cast partial auto-correlation

        # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
        if ind == 0:
            # ARMA
            sigma2_ind_from_last = -1
            d_ind_from_last = None
            lambda_ind_from_last = None
            var2_ind_from_last = None
        elif ind == 1:
            # ARTFIMA
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = -3
            var2_ind_from_last = None
        elif ind == 2:
            # ARFIMA
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = None
            var2_ind_from_last = None
        elif ind == 3:
            # ARTFIMA-SV
            sigma2_ind_from_last = -2
            d_ind_from_last = -1
            lambda_ind_from_last = -3
            var2_ind_from_last = -4

        # d parameter
        if d_ind_from_last is not None and ind == 2:
            samples_MCMC[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_MCMC[:, d_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_pseudo_MCMC_Taylor[:, d_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, d_ind_from_last] = 0.5 * \
                np.tanh(samples_pseudo_MCMC_Coreset[:, d_ind_from_last])
        # NOTE: For ind = 1 and ind = 3 the parameterization is just d (unrestricted).

        # variance param (all models have it)
        samples_MCMC[:, sigma2_ind_from_last] = np.exp(
            samples_MCMC[:, sigma2_ind_from_last])
        samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last] = np.exp(
            samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last])
        if ind == 0:
            samples_time_domain_MCMC[:, sigma2_ind_from_last] = np.exp(
                samples_time_domain_MCMC[:, sigma2_ind_from_last])
        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, sigma2_ind_from_last])

        # Lambda
        if lambda_ind_from_last is not None:
            samples_MCMC[:, lambda_ind_from_last] = np.exp(
                samples_MCMC[:, lambda_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, lambda_ind_from_last])

        # var2 parameter
        if var2_ind_from_last is not None:
            samples_MCMC[:, var2_ind_from_last] = np.exp(
                samples_MCMC[:, var2_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, var2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, var2_ind_from_last])
            samples_pseudo_MCMC_Coreset[:, var2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Coreset[:, var2_ind_from_last])  # Coreset included for this example

        if file_names_pseudo_MCMC_Coreset[ind] is not None:
            coreset_sizes = dict_pseudo_MCMC_Coreset['coreset_sizes']

        # end things that are constant...
        ll = 0
        jj = 0
        for param in range(nrow):
            if ll > 2:
                ll = 0
                jj = jj + 1

            ax[jj][ll].spines['top'].set_visible(False)
            ax[jj][ll].spines['right'].set_visible(False)
            ax[jj][ll].get_xaxis().tick_bottom()
            ax[jj][ll].get_yaxis().tick_left()

            # BEGIN PLOT all of the parameters.
            theta_MCMC = samples_MCMC[:, param]
            theta_pseudo_MCMC_Taylor = samples_pseudo_MCMC_Taylor[:, param]
            if ind == 0:
                theta_time_domain_MCMC = samples_time_domain_MCMC[:, param]

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                theta_pseudo_MCMC_Coreset = samples_pseudo_MCMC_Coreset[:, param]

            min_x_MCMC = np.min(theta_MCMC)
            max_x_MCMC = np.max(theta_MCMC)
            min_x_pseudo_MCMC_Taylor = np.min(theta_pseudo_MCMC_Taylor)
            max_x_pseudo_MCMC_Taylor = np.max(theta_pseudo_MCMC_Taylor)
            if ind == 0:
                min_x_time_domain_MCMC = np.min(theta_time_domain_MCMC)
                max_x_time_domain_MCMC = np.max(theta_time_domain_MCMC)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                min_x_pseudo_MCMC_Coreset = np.min(theta_pseudo_MCMC_Coreset)
                max_x_pseudo_MCMC_Coreset = np.max(theta_pseudo_MCMC_Coreset)

            scale_positive = 1
            scale_negative = 1

            # Fix minimum
            if min_x_MCMC < 0:
                min_x_MCMC = min_x_MCMC*scale_negative
            else:
                min_x_MCMC = min_x_MCMC*scale_positive

            if min_x_pseudo_MCMC_Taylor < 0:
                min_x_pseudo_MCMC_Taylor = min_x_pseudo_MCMC_Taylor*scale_negative
            else:
                min_x_pseudo_MCMC_Taylor = min_x_pseudo_MCMC_Taylor*scale_positive

            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                if min_x_pseudo_MCMC_Coreset < 0:
                    min_x_pseudo_MCMC_Coreset = min_x_pseudo_MCMC_Coreset*scale_negative
                else:
                    min_x_pseudo_MCMC_Coreset = min_x_pseudo_MCMC_Coreset*scale_positive

            if ind == 0:
                if min_x_time_domain_MCMC < 0:
                    min_x_time_domain_MCMC = min_x_time_domain_MCMC*scale_negative
                else:
                    min_x_time_domain_MCMC = min_x_time_domain_MCMC*scale_positive

            # Fix maximum
            if max_x_MCMC < 0:
                max_x_MCMC = max_x_MCMC*scale_positive
            else:
                max_x_MCMC = max_x_MCMC*scale_negative
            if max_x_pseudo_MCMC_Taylor < 0:
                max_x_pseudo_MCMC_Taylor = max_x_pseudo_MCMC_Taylor*scale_positive
            else:
                max_x_pseudo_MCMC_Taylor = max_x_pseudo_MCMC_Taylor*scale_negative
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                if max_x_pseudo_MCMC_Coreset < 0:
                    max_x_pseudo_MCMC_Coreset = max_x_pseudo_MCMC_Coreset*scale_positive
                else:
                    max_x_pseudo_MCMC_Coreset = max_x_pseudo_MCMC_Coreset*scale_negative
            if ind == 0:
                if max_x_time_domain_MCMC < 0:
                    max_x_time_domain_MCMC = max_x_time_domain_MCMC*scale_positive
                else:
                    max_x_time_domain_MCMC = max_x_time_domain_MCMC*scale_negative

            # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
            # Which KDEs to we plot for different examples?
            if ind == 0:
                # ARMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_time_domain_MCMC])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, min_x_time_domain_MCMC])

            elif ind == 1:
                # ARTFIMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            elif ind == 2:
                # ARFIMA
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            elif ind == 3:
                # ARTFIMA-SV
                min_x = np.min(
                    [min_x_MCMC, min_x_pseudo_MCMC_Taylor, min_x_pseudo_MCMC_Coreset])
                max_x = np.max(
                    [max_x_MCMC, max_x_pseudo_MCMC_Taylor, max_x_pseudo_MCMC_Coreset])

            x_grid = np.linspace(min_x, max_x, 500)

            kde_MCMC = sps.gaussian_kde(theta_MCMC)
            kde_pseudo_MCMC_Taylor = sps.gaussian_kde(theta_pseudo_MCMC_Taylor)
            if ind == 0:
                kde_time_domain_MCMC = sps.gaussian_kde(theta_time_domain_MCMC)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                kde_pseudo_MCMC_Coreset = sps.gaussian_kde(
                    theta_pseudo_MCMC_Coreset)

            if bandwidthKDE is not None:
                kde_MCMC.set_bandwidth(bandwidthKDE)
                kde_pseudo_MCMC_Taylor.set_bandwidth(bandwidthKDE)
                if ind == 0:
                    kde_time_domain_MCMC.set_bandwidth(bandwidthKDE)
                if file_names_pseudo_MCMC_Coreset[ind] is not None:
                    kde_pseudo_MCMC_Coreset.set_bandwidth(bandwidthKDE)

            pdfvals_MCMC = kde_MCMC(x_grid)
            pdfvals_pseudo_MCMC_Taylor = kde_pseudo_MCMC_Taylor(x_grid)
            if ind == 0:
                pdfvals_time_domain_MCMC = kde_time_domain_MCMC(x_grid)
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                pdfvals_pseudo_MCMC_Coreset = kde_pseudo_MCMC_Coreset(x_grid)

            rects1 = ax[jj][ll].plot(
                x_grid, pdfvals_MCMC, 'k', linewidth=2, label=r'$\mathrm{MCMC}$')
            rects2 = ax[jj][ll].plot(x_grid, pdfvals_pseudo_MCMC_Taylor, 'magenta',
                                     linewidth=2, linestyle='--', label=r'$\mathrm{Taylor}$')
            if file_names_pseudo_MCMC_Coreset[ind] is not None:
                rects3 = ax[jj][ll].plot(x_grid, pdfvals_pseudo_MCMC_Coreset, 'orange',
                                         linewidth=2, linestyle='--', label=r'$\mathrm{Coreset}$')
            if ind == 0:
                rects4 = ax[jj][ll].plot(x_grid, pdfvals_time_domain_MCMC, 'green',
                                         linewidth=2, linestyle='--', label=r'$\mathrm{Coreset}$')

            ax[jj][ll].set_ylim(0, ax[jj][ll].get_ylim()[1])
            ax[jj][ll].set_yticks([])
            ax[jj][ll].set_ylabel(density_names[ind][param], fontsize=10)
            ax[jj][ll].set_title(param_names[ind][param], fontsize=12)

            # Legend
            if jj == 0 and ll == 0:
                if ind == 0:
                    ax[jj][ll].legend((rects1[0], rects2[0], rects4[0]), (
                        r'$\mathrm{MCMC}$', r'$\mathrm{Taylor}$', r'$\mathrm{Gauss-MCMC}$'), prop={'size': 6}, loc="upper right")
                    ax[jj][ll].set_xlim([min_x, max_x + 0.65])
                else:
                    # for all other inds we include the coreset, but no time domain MCMC
                    ax[jj][ll].legend((rects1[0], rects2[0], rects3[0]), (
                        r'$\mathrm{MCMC}$', r'$\mathrm{Taylor}$', r'$\mathrm{Coreset}$'), prop={'size': 6}, loc="upper right")

                    if ind == 1:
                        ax[jj][ll].set_xlim([min_x, max_x + 0.02])
                    elif ind == 2:
                        ax[jj][ll].set_xlim([min_x, max_x + 0.007])
                    elif ind == 3:
                        ax[jj][ll].set_xlim([min_x, max_x + 0.05])

            ax[jj][ll].tick_params(axis='x', which='major', labelsize=10)
            try:
                plt.tight_layout()
            except ValueError:
                print("Plots are corrupted")

            ll = ll + 1

        fig.subplots_adjust(hspace=0.6, wspace=0.4)
        if ind == 0:
            plt.savefig(plotSaveDir + "KDE_all_params_ARMA_model.pdf")
        elif ind == 1:
            plt.savefig(plotSaveDir + "KDE_all_params_ARTFIMA_model.pdf")
        elif ind == 2:
            plt.savefig(plotSaveDir + "KDE_all_params_ARFIMA_model.pdf")
        elif ind == 3:
            plt.savefig(plotSaveDir + "KDE_all_params_ARTFIMA-SV_model.pdf")


if 'Spectral_plots' in which_plots:
    ####
    # 2 x 2 plot
    nrow = 2
    ncol = 2

    set_tick_size = 13

    file_path = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Results/Runs/" % alias
    titles = ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
    file_names_MCMC = ['ARMA/MCMC_ARMA_dataVancouver_AR2_MA3_n22000_AR2_MA3_seed1_MCMCiter10000.pkl',
                       'ARTFIMA/MCMC_ARTFIMA_dataBromma_AR2_TFI_MA2_n225000_AR2_MA2_seed1_MCMCiter10000.pkl',
                       'ARFIMA/MCMC_ARFIMA_dataSimulated_AR2_FI_MA1_n2500000_AR2_MA1_seed1_MCMCiter10000.pkl',
                       'ARTFIMA_SV/MCMC_ARTFIMA_SV_dataBitcoin_AR1_TFI_MA1_SV_new_n500000_AR1_MA1_seed1_MCMCiter10000.pkl']
    file_names_MCMC = [file_path + file_names_MCMC[item] for item in range(4)]

    file_names_pseudo_MCMC_Taylor = ['ARMA/Taylor/2020-01-30/20:40/Pseudo_marginal_results.pkl',
                                     'ARTFIMA/Taylor/2020-01-30/22:22/Pseudo_marginal_results.pkl',
                                     'ARFIMA/Taylor/2020-02-01/12:13/Pseudo_marginal_results.pkl',
                                     'ARTFIMA_SV/Taylor/2020-02-06/16:10/Pseudo_marginal_results.pkl']
    file_names_pseudo_MCMC_Taylor = [
        file_path + file_names_pseudo_MCMC_Taylor[item] for item in range(4)]

    file_path_data = "/home/%s/Dropbox/Research/Papers/Started 2019/Temporal Subsampling code/Data/" % alias
    file_names_data_set_names = ['Vancouver_AR2_MA3', 'Bromma_AR2_TFI_MA2',
                                 'Simulated_AR2_FI_MA1', 'Bitcoin_AR1_TFI_MA1_SV_new']

    file_names_data_set_names = [
        file_path_data + file_names_data_set_names[item] + '.npy' for item in range(4)]

    fig, ax = plt.subplots(nrow, ncol)
    ind = 0
    for ii in range(nrow):
        for jj in range(ncol):

            # Specific for each model
            if ind == 0:
                # ARMA(2,3)
                q = 2
                p = 3
            elif ind == 1:
                # ARTFIMA(2,2)
                q = 2
                p = 2

            elif ind == 2:
                # ARFIMA(2,1)
                q = 2
                p = 1
            elif ind == 3:
                # ARTFIMA-SV(1,1)
                q = 1
                p = 1

            ax[ii][jj].spines['top'].set_visible(False)
            ax[ii][jj].spines['right'].set_visible(False)
            ax[ii][jj].get_xaxis().tick_bottom()
            ax[ii][jj].get_yaxis().tick_left()
            dict_MCMC = pickle.load(open(file_names_MCMC[ind], 'rb'))
            dict_pseudo_MCMC_Taylor = pickle.load(
                open(file_names_pseudo_MCMC_Taylor[ind], 'rb'))

            burnIn = 1000
            samples_MCMC = dict_MCMC['samples'][burnIn + 1:, :]
            samples_pseudo_MCMC_Taylor = dict_pseudo_MCMC_Taylor['samples'][burnIn + 1:, :]

            # Cast partial auto-correlation to ordinary parameterization.
            samples_MCMC[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_MCMC[:, :q]))
            samples_pseudo_MCMC_Taylor[:, :q] = np.vstack(
                map(lambda x: reparam(x, MA=False), samples_pseudo_MCMC_Taylor[:, :q]))

            samples_MCMC[:, q:(
                q + p)] = np.vstack(map(lambda x: reparam(x, MA=True), samples_MCMC[:, q:(q + p)]))
            samples_pseudo_MCMC_Taylor[:, q:(q + p)] = np.vstack(
                map(lambda x: reparam(x, MA=True), samples_pseudo_MCMC_Taylor[:, q:(q + p)]))

            # End Cast partial auto-correlation

            # ['ARMA(2,3)', 'ARTFIMA(2,2)', 'ARFIMA(2,1)', 'ARTFIMA-SV(1,1)']
            if ind == 0:
                # ARMA
                sigma2_ind_from_last = -1
                d_ind_from_last = None
                lambda_ind_from_last = None
                var2_ind_from_last = None
            elif ind == 1:
                # ARTFIMA
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = -3
                var2_ind_from_last = None
            elif ind == 2:
                # ARFIMA
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = None
                var2_ind_from_last = None
            elif ind == 3:
                # ARTFIMA-SV
                sigma2_ind_from_last = -2
                d_ind_from_last = -1
                lambda_ind_from_last = -3
                var2_ind_from_last = -4

            # d parameter
            if d_ind_from_last is not None and ind == 2:
                samples_MCMC[:, d_ind_from_last] = 0.5 * \
                    np.tanh(samples_MCMC[:, d_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, d_ind_from_last] = 0.5 * \
                    np.tanh(samples_pseudo_MCMC_Taylor[:, d_ind_from_last])

            # NOTE: For ind = 1 and ind = 3 the parameterization is just d (unrestricted).

            # variance param (all models have it)
            samples_MCMC[:, sigma2_ind_from_last] = np.exp(
                samples_MCMC[:, sigma2_ind_from_last])
            samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last] = np.exp(
                samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last])

            # Lambda
            if lambda_ind_from_last is not None:
                samples_MCMC[:, lambda_ind_from_last] = np.exp(
                    samples_MCMC[:, lambda_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Taylor[:, lambda_ind_from_last])

            # var2 parameter
            if var2_ind_from_last is not None:
                samples_MCMC[:, var2_ind_from_last] = np.exp(
                    samples_MCMC[:, var2_ind_from_last])
                samples_pseudo_MCMC_Taylor[:, var2_ind_from_last] = np.exp(
                    samples_pseudo_MCMC_Taylor[:, var2_ind_from_last])

            # We have all parameters in constrained space. Begin spectral density:
            x = np.load(file_names_data_set_names[ind])
            log_I_pg = np.log(p_gram(np.fft.fft(x)))[1:]
            ind_range = np.arange(1, int(np.floor((len(x))/2))+1)
            n = len(x)
            T = len(log_I_pg)
            ax[ii][jj].plot(log_I_pg, 'k-', alpha=0.15, label='Periodogram')

            N_MCMC = samples_MCMC.shape[1]
            f_spectral_samples_MCMC = np.zeros((N_MCMC, T))
            f_spectral_samples_pseudo_MCMC = np.zeros((N_MCMC, T))
            if ind == 0:
                # ARMA
                phi = samples_MCMC[:, :q]
                theta = samples_MCMC[:, q:(q + p)]
                var = samples_MCMC[:, sigma2_ind_from_last]
                d = np.zeros(N_MCMC)
                lambda_ = np.zeros(N_MCMC)
                var2 = np.zeros(N_MCMC)

                phipseudo = samples_pseudo_MCMC_Taylor[:, :q]
                thetapseudo = samples_pseudo_MCMC_Taylor[:, q:(q + p)]
                varpseudo = samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last]
                dpseudo = np.zeros(N_MCMC)
                lambda_pseudo = np.zeros(N_MCMC)
                var2pseudo = np.zeros(N_MCMC)

            elif ind == 1:
                # ARTFIMA
                phi = samples_MCMC[:, :q]
                theta = samples_MCMC[:, q:(q + p)]
                var = samples_MCMC[:, sigma2_ind_from_last]
                d = samples_MCMC[:, d_ind_from_last]
                lambda_ = samples_MCMC[:, lambda_ind_from_last]
                var2 = np.zeros(N_MCMC)

                phipseudo = samples_pseudo_MCMC_Taylor[:, :q]
                thetapseudo = samples_pseudo_MCMC_Taylor[:, q:(q + p)]
                varpseudo = samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last]
                dpseudo = samples_pseudo_MCMC_Taylor[:, d_ind_from_last]
                lambda_pseudo = samples_pseudo_MCMC_Taylor[:,
                                                           lambda_ind_from_last]
                var2pseudo = np.zeros(N_MCMC)

            elif ind == 2:
                # ARFIMA
                phi = samples_MCMC[:, :q]
                theta = samples_MCMC[:, q:(q + p)]
                var = samples_MCMC[:, sigma2_ind_from_last]
                d = samples_MCMC[:, d_ind_from_last]
                lambda_ = np.zeros(N_MCMC)
                var2 = np.zeros(N_MCMC)

                phipseudo = samples_pseudo_MCMC_Taylor[:, :q]
                thetapseudo = samples_pseudo_MCMC_Taylor[:, q:(q + p)]
                varpseudo = samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last]
                dpseudo = samples_pseudo_MCMC_Taylor[:, d_ind_from_last]
                lambda_pseudo = np.zeros(N_MCMC)
                var2pseudo = np.zeros(N_MCMC)

            elif ind == 3:
                # ARTFIMA-SV
                phi = samples_MCMC[:, :q]
                theta = samples_MCMC[:, q:(q + p)]
                var = samples_MCMC[:, sigma2_ind_from_last]
                d = samples_MCMC[:, d_ind_from_last]
                lambda_ = samples_MCMC[:, lambda_ind_from_last]
                var2 = samples_MCMC[:, var2_ind_from_last]

                phipseudo = samples_pseudo_MCMC_Taylor[:, :q]
                thetapseudo = samples_pseudo_MCMC_Taylor[:, q:(q + p)]
                varpseudo = samples_pseudo_MCMC_Taylor[:, sigma2_ind_from_last]
                dpseudo = samples_pseudo_MCMC_Taylor[:, d_ind_from_last]
                lambda_pseudo = samples_pseudo_MCMC_Taylor[:,
                                                           lambda_ind_from_last]
                var2pseudo = samples_pseudo_MCMC_Taylor[:, var2_ind_from_last]

            for kk in range(N_MCMC):
                f_spectral_samples_MCMC[kk, :] = f_ARTFIMA(
                    ind_range, phi[kk], theta[kk], var[kk], d[kk], lambda_[kk], var2[kk], n)
                f_spectral_samples_pseudo_MCMC[kk, :] = f_ARTFIMA(
                    ind_range, phipseudo[kk], thetapseudo[kk], varpseudo[kk], dpseudo[kk], lambda_pseudo[kk], var2pseudo[kk], n)

            # Credible intervals
            lower = np.quantile(np.log(f_spectral_samples_MCMC), 0.025, axis=0)
            upper = np.quantile(np.log(f_spectral_samples_MCMC), 0.975, axis=0)

            lowerpseudo = np.quantile(
                np.log(f_spectral_samples_pseudo_MCMC), 0.025, axis=0)
            upperpseudo = np.quantile(
                np.log(f_spectral_samples_pseudo_MCMC), 0.975, axis=0)

            if ind == 2:
                f_spectral_true = f_ARTFIMA(ind_range, np.array(
                    [0.22, -0.1]), np.array([0.5]), 1.0, 0, 0, 0, n)
                leg_true = ax[ii][jj].plot(
                    np.log(f_spectral_true), 'g', linewidth=1)
                ax[ii][jj].legend((leg_MCMC[0], leg_pseudo[0], leg_true[0]), ("MCMC", "Subsampling MCMC", "True"), prop={
                                  'size': set_tick_size-4}, loc="upper right")
                #ax[ii][jj].set_xlim([0, 1.2*len(f_spectral_true)])
                ax[ii][jj].set_ylim([-20, 30])

            ax[ii][jj].set_title(titles[ind], fontsize=set_title_size)

            ax[ii][jj].fill_between(
                ind_range, lower, upper, color='b', alpha=0.15)
            ax[ii][jj].plot(lower, 'b', linewidth=0.5, alpha=1)
            ax[ii][jj].plot(upper, 'b', linewidth=0.5, alpha=1)
            leg_MCMC = ax[ii][jj].plot(np.log(
                np.mean(f_spectral_samples_MCMC, axis=0)), 'b', linewidth=1)  # , label = "MCMC"

            ax[ii][jj].fill_between(
                ind_range, lower, upper, color='r', alpha=0.15)
            ax[ii][jj].plot(lower, 'r', linewidth=0.5, alpha=1)
            ax[ii][jj].plot(upper, 'r', linewidth=0.5, alpha=1)
            leg_pseudo = ax[ii][jj].plot(np.log(np.mean(
                f_spectral_samples_pseudo_MCMC, axis=0)), 'r', linewidth=1)  # , label = "Subsampling MCMC")

            n_freq = f_spectral_samples_MCMC.shape[1]
            ax[ii][jj].set_xticks([0, n_freq/4, n_freq/2, 3*n_freq/4, n_freq])
            ax[ii][jj].set_xticklabels(
                [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
            #            ,  [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
            #ax[ii][jj].set_xticklabels( [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
            # plt.xticks([0,n/4, n/2, 3*n/4,n],[r'$0$', r'$\pi/4$', r'$\pi/2$',
            #           r'$3\pi/4$', r'$\pi$'])

            ax[ii][jj].set_xlabel('Frequency')
            ax[ii][jj].set_ylabel('Log Spectral Density')

            # if ind == 2:
            # Plot true spectral density for the ARFIMA case. Recall that data is ARFIMA(2, 1) with d = 0, and phi = (0.22, -1.1), theta = (0.5), sigma2 =1
            #    f_spectral_true = f_ARTFIMA(ind_range, np.array([0.22, -0.1]), np.array([0.5]), 1.0, 0, 0, 0, n)
            #    leg_true = ax[ii][jj].plot(np.log(f_spectral_true), 'g', linewidth=1)
            #    ax[ii][jj].legend((leg_true[0]), ("True"), prop={'size' : set_tick_size-4}, loc = "upper right")
            #    ax[ii][jj].set_xlim([0, 25000])

            plt.tight_layout()

            ind = ind + 1

    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.savefig(plotSaveDir + "log_spectral_dens_all_models.png", format='png')
    #plt.savefig(plotSaveDir + "log_spectral_dens_all_models.pdf", format='pdf')


aaa = 1
