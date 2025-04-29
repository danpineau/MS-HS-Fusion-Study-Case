#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:09:50 2025

@author: dpineau
"""


import numpy as np
import os
from astropy.io import fits
from scipy.ndimage import gaussian_filter

#%%


def inputs_for_models(data_path, a=0, b=5):
    os.chdir(data_path)

    # import du cube
    fname_cube = "synthetic_cube_orion_bar.fits"
    fits_cube = fits.open(fname_cube)
    cube = np.asarray(fits_cube[0].data, dtype=np.float32)

    # import des longueurs d'onde du cube
    lamb_cube = np.load("lamb_cube.npy")

    # import des spectres LMM
    L_specs = (
        np.load("lmm_specs.npy")[a:b] / 1e3
    )  # CHANGEMENT PAR RAPPORT À inputs_for_models_1_1
    # conversion de µJy/arcsec-2 à mJy/arcsec-2
    # print("spectres en mJy/arcsec-2 (nouveau)")

    new_L_specs = np.copy(L_specs)
    
    # CHANGEMENTS PAR RAPPORT À inputs_for_models_1_3:
    for i in range(len(new_L_specs)):
        for j in range(len(new_L_specs[i])):
            if new_L_specs[i][j] > 150:
                new_L_specs[i][j] = (new_L_specs[i][j+1] + new_L_specs[i][j-1]) / 2

    # import des psfs
    fname_psfs = (
        "cropped_psfs_"
        + str(len(lamb_cube))
        + "_size_"
        + str(cube[0].shape[0])
        + ".fits"
    )
    fits_cube = fits.open(fname_psfs)
    size_psf = 20
    half_size = size_psf // 2
    center = (125, 250)
    psfs_monoch = np.asarray(
        fits_cube[0].data, dtype=np.float32
    )[:, center[0] - half_size : center[0] + half_size, center[1] - half_size : center[1] + half_size]
    # print("FoV des PSF réduit") # CHANGEMENTS PAR RAPPORT À inputs_for_models_1_2

    # decimation factor for the spectrometer model
    decim = 2

    # INPUTS MIRIM

    # import des PCE de MIRIM
    L_pce_mirim = np.load("pce_" + str(len(lamb_cube)) + ".npy")

    # INPUTS SPECTRO

    # import des PCE du spectro
    L_pce_spectro = np.load("spectro_pce_" + str(len(lamb_cube)) + ".npy")

    thres = 0.01
    L_pce_spectro[L_pce_spectro < thres] = thres

    return psfs_monoch, L_pce_mirim, L_pce_spectro, lamb_cube, new_L_specs, decim


def min_not_zero(data):
    Ni, Nj = data.shape
    data_min = np.max(data)
    for i in range(Ni):
        for j in range(Nj):
            data_point = data[i, j]
            if data[i, j] != 0.0 and data_point < data_min:
                data_min = data_point
    return data_min

def rescale_0_1(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


def abundance_maps_inputs(data_path, a=0, b=5):
    os.chdir(data_path)

    # import true abundance maps
    fname_true_maps = "decimated_abundance_maps_orion.fits"
    fits_cube = fits.open(fname_true_maps)
    true_maps = np.asarray(fits_cube[0].data, dtype=np.float32)[a:b, :250, :]
    true_maps.shape

    shape_target = true_maps[0].shape
    
    
    # MODIFYING ABUNDANCE MAP 1
    
    true_maps[0][true_maps[0] > 0.8] = 0.8

    # MODIFYING ABUNDANCE MAP 4

    n_map = 3

    map4 = true_maps[n_map]
    # plt.imshow(map4)

    d = 20
    i1, j1 = 104, 202
    # star1 = map4[i1 - d : i1 + d, j1 - d : j1 + d]
    i2, j2 = 121, 318
    # star2 = map4[i2 - d : i2 + d, j2 - d : j2 + d]
    i3, j3 = 113, 345
    # star3 = map4[i3 - d : i3 + d, j3 - d : j3 + d]
    # star3.shape

    mask = np.zeros((2 * d, 2 * d))
    mask.shape

    # plt.imshow(star3)

    map4[i1 - d : i1 + d, j1 - d : j1 + d] = mask
    map4[i2 - d : i2 + d, j2 - d : j2 + d] = mask
    map4[i3 - d : i3 + d, j3 - d : j3 + d] = mask

    # plt.imshow(map4)

    # changing values of map 4
    map4[map4 <= 0.35] = 0
    min_not_zero_map_4 = min_not_zero(map4)
    map4[map4 == 0] = min_not_zero_map_4
    map4_rescaled = rescale_0_1(map4)

    map4_rescaled_blurred = gaussian_filter(map4_rescaled, 1.4)
    
    map4_rerescaled = rescale_0_1(map4_rescaled_blurred)
    
    true_maps[n_map] = map4_rerescaled

    # print("Maps chosen: map1 wout stars and map4 rescaled and blurred.")

    # plt.imshow(true_maps[n_map])

    return true_maps, shape_target



