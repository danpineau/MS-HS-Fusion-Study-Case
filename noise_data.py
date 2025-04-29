#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:55:41 2022

@author: dpineau
"""

import numpy as np
from numpy.random import randn, seed

# pas sûr de cette fonction
def set_snr_in_y(y, snr):
    n_pix = np.prod(y.shape)
    P_signal = np.sum(y**2) / n_pix
    
    std = 10**(0.5 * np.log10(P_signal) - 0.05 * snr)
    return std

def AWGN(signal, std):
    """additive white gaussian noise"""

    # generate zero-mean gaussian noise
    seed(9001)
    m, n = signal.shape
    noise = std * randn(m, n)

    return signal + noise

def add_noise_to_data(y, std):
    return np.asarray([AWGN(signal=y[f], std=std) for f in range(y.shape[0])])

def add_noise_to_data2(y, snr):
    std = set_snr_in_y(y, snr)
    return np.asarray([AWGN(signal=y[f], std=std) for f in range(y.shape[0])])

# same as AWGN, but noise added directly on a 3 cube instead of individual images
def AWGN2(signal, std):
    """additive white gaussian noise"""

    # generate zero-mean gaussian noise
    seed(9001) # useful to assume that the noise is exactly the same on all data. But data acquired at different times. So no need.
    # alpha, beta, lamb = signal.shape
    # noise = std * randn(alpha, beta, lamb)
    
    noise = std * randn(*signal.shape) # écriture avec * pour ne plus dépendre du nombre de dimensions de signal

    return signal + noise

# adding poisson noise as the shot noise
# np.sqrt(signal) is the std of the shot noise at each pixel
def add_poisson_noise(signal, coeff = 1):
    alpha, beta, lamb = signal.shape
    noise = coeff * np.sqrt(signal) * randn(alpha, beta, lamb)
    return signal + noise

def gaussian_noise_cube(shape, std):
    a, b, c = shape
    noise_cube = std * randn(a, b, c)
    return noise_cube

def gaussian_noise_maps(shape, std):
    n_maps, alpha, beta = shape
    noise_maps = std * randn(n_maps, alpha, beta)
    return noise_maps    

#%%

# seed(9001)

# #%%
# std = 10
# noise = std * randn(10000)

# abs_noise = np.abs(noise)

# print(np.std(abs_noise))
# print(std / np.sqrt(2))

#%%

def get_snr(sig_ref, sig_nsy=None, noise=None, noise_var=None, log=True):
    """Signal-to-Noise ratio (SNR): dB; Gonzalez book, Digital image processing
    snr = (sig_ref**2).sum() / (noise**2).sum()
    snr = (sig_ref**2).sum() / ((sig_ref-sig_nsy)**2).sum()
    snr = (sig_ref**2).mean() / var_noise # Only for zero-mean noise
    20 log10(||noiseless||_2/||noise||_2)
    """

    if noise_var: #or noise_var == 0:
        # Measure original signal power
        n_pix = np.prod(sig_ref.shape)
        P_signal = np.sum(sig_ref**2) / n_pix
        # print("P_signal = {0:f}".format(P_signal))
        # Measure noise power
        P_noise = noise_var
    else:
        # Measure original signal power
        P_signal = np.sum(sig_ref**2)

        # Measure noise power
        if np.any(noise):
            P_noise = np.sum(noise**2)
        elif np.any(sig_nsy):
            noise = sig_ref - sig_nsy
            P_noise = np.sum(noise**2)

    snr = P_signal / P_noise
    if log:
        return 10 * np.log10(snr)
    else:
        return snr

def get_snr_astro(sig_ref, sigma):
    return np.mean(sig_ref) / sigma

# function to use to calculate the std of the white gaussian noise to use
# in order to have a given snr. This function uses the following formula
# for the calculation of the snr: snr = (sig_ref**2).mean() / var_noise
# which works only for zero-mean noise.
def give_std_for_desired_snr(y, snr):
    n_pix = np.prod(y.shape)
    P_signal = np.sum(y**2) / n_pix
    
    std = np.sqrt(P_signal / 10**(snr/10))
    return std

#%%

def make_obs_data_with_snr(model, true_maps, snr):
    obs_data_without_noise = model.forward(true_maps)
    std_of_noise_to_apply = give_std_for_desired_snr(obs_data_without_noise, snr)
    noised_obs_data = AWGN2(obs_data_without_noise, std_of_noise_to_apply)
    return noised_obs_data, std_of_noise_to_apply

def noise_data_for_snr(data_wout_noise, L_snr):
    L_std = []
    for snr in L_snr:
        L_std.append(give_std_for_desired_snr(data_wout_noise, snr))
    
    L_obs_data = []
    for i in range(len(L_std)):
        obs_data = AWGN2(data_wout_noise, L_std[i])
        L_obs_data.append(obs_data)
    
    return L_obs_data, L_std