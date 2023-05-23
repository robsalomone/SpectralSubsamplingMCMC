This repository contains code for the paper 
Salomone R., Quiroz, M., Kohn, R., Villani, M., and Tran, M.N. (2020), *Spectral Subsampling MCMC for  Stationary Time Series*,  Proceedings of the International Conference on Machine Learning (ICML) 2020.  

Note that code was originally available as part of the ICML supplementary material, but the link on the ICML website is no longer functioning. 

The main program to execute is `SpectralSubsamplingMCMC.py`. The user-specified settings should be as in the paper. For a given data_set_name (corresponds to one model),
the code does several assertions to ensure the correct setup (for example, how many lags a process has).

The required packages are those we import.

The folder `Data` contains the datasets we use in the paper. Preprocessing steps as well as instructions where to obtain the datasets are in the paper.

The `Inspect_variance_grouping.py` computes the variance reduction of different grouping strategies. See the  paper. Also, it assumes that the main code ("SpectralSubsamplingMCMC") has been executed
to create the relevant inputs.

Finally, `ICML_Figures_and_Tables.py` contains the code to generate the figures in the paper.
