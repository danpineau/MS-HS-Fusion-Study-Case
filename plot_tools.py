#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:39:03 2023

@author: dpineau
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

# comparison of maps WITHOUT color gradient adjustment
def plot_maps_no_adjustement(true_maps, estimated_maps):
    n_rows = true_maps.shape[0]
    n_col = 2
    
    x = np.copy(estimated_maps)
    fig = plt.figure()
    i = 0
    for x_map in x:
        fig.add_subplot(n_rows, n_col, 2*i + 1)
        plt.imshow(true_maps[i])
        plt.colorbar()
        
        fig.add_subplot(n_rows, n_col, 2*i + 2)
        plt.imshow(x_map)
        plt.colorbar()
        i += 1

# comparison of maps WITH color gradient adjustment
def plot_maps(true_maps, estimated_maps, plot_axis = True, plot_title = True):
    n_rows = true_maps.shape[0]
    n_col = 2
    
    x = np.copy(estimated_maps)

    min_true, min_est = np.min(true_maps), np.min(x)
    min_val = min(min_true, min_est)
    max_true, max_est = np.max(true_maps), np.max(x)
    max_val = min(max_true, max_est)

    min_val, max_val = np.min(true_maps), np.max(true_maps)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharex = True, sharey = True)
    m = 0
    n = 0
    for ax in axes.flat:
        if (m+n)%2 == 0:
            im = ax.imshow(true_maps[m], vmin=min_val, vmax=max_val)
            m += 1
            
            if plot_title:
                ax.set_title("Original map {}".format(m))
            
        else:
            im = ax.imshow(x[n], vmin=min_val, vmax=max_val)
            n += 1
            ax.set_title("Reconstructed map {}".format(n))
        
        if plot_axis == False:
            ax.axis('Off')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    return fig

def plot_list_maps(L_maps, plot_axis = True, colorbar = True, plot_title = True):
    n_rows = len(L_maps[0])
    n_cols = len(L_maps)

    min_val, max_val = np.min(L_maps[-1]), np.max(L_maps[-1])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex = True, sharey = True)
    for j in range(n_cols):
        for i in range(n_rows):
            im = axes[i, j].imshow(L_maps[j][i], vmin=min_val, vmax=max_val)
            
            if plot_title:
                axes[i, j].set_title("Map {}".format(i + 1))
            
            if plot_axis == False:
                axes[i, j].axis('Off')

    if colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout()
    
    return fig


# comparison of maps without adjustment, axes shared
def plot_maps_share(true_maps, estimated_maps):
    n_rows = true_maps.shape[0]
    n_col = 2
    
    x = np.copy(estimated_maps)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharex = True, sharey = True)
    m = 0
    n = 0
    for ax in axes.flat:
        if (m+n)%2 == 0:
            ax.imshow(true_maps[m])
            m += 1
        else:
            ax.imshow(x[n])
            n += 1


def plot_true_maps(true_maps, n_rows = 5, n_col = 1, colorbar = True, plot_axis = True, plot_title = True):
    
    # plt.rcParams["figure.figsize"] = (4,12)

    # min_val, max_val = np.min(true_maps), np.max(true_maps)
    min_val, max_val = 0, 1
    print("min, max = ", min_val, max_val)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharex = True, sharey = True, figsize=(4, 12))
    m = 0
    for ax in axes.flat:
        im = ax.imshow(true_maps[m], vmin=min_val, vmax=max_val)
        
        m += 1
        
        if plot_title:
            ax.set_title("Map {}".format(m))
        
        if not plot_axis:
            ax.axis('Off')

    if colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    return fig


def plot_maps_no_readj(true_maps, n_rows = 5, n_col = 1, colorbar = True, plot_axis = True, plot_title = True):

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, sharex = True, sharey = True)
    m = 0
    for ax in axes.flat:
        im = ax.imshow(true_maps[m])
        m += 1
        
        if not plot_axis:
            ax.axis("Off")
        
        if plot_title:
            ax.set_title("Map {}".format(m))
        
        if colorbar:
            fig.colorbar(im, ax=ax)
        
        if m == true_maps.shape[0]:
            break
    
    return fig


def maps_to_cube(maps, L_specs):  # L_specs.shape = (K, N) = (5, 300)
    cube = np.sum(L_specs[:, :, np.newaxis, np.newaxis] * maps[:, np.newaxis, :, :], axis=0)
    return cube

def plot_imager_data(y_imager):
    n_rows = 5
    n_cols = 2
    number_of_images = y_imager.shape[0]
    
    # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex = True, sharey = True)
    for i in range(number_of_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(y_imager[i])
        plt.title("band {}".format(i))
        plt.colorbar()
    
    return None

def plot_spectro_data(y_spectro, start_index):
    n_rows = 5
    n_cols = 2
    
    # fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex = True, sharey = True)
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(y_spectro[i + start_index])
        plt.title("index {}".format(i + start_index))
        plt.colorbar()
        
    return None

def plot_spectra(wavelength, spectra, L_label = []):
    fig = plt.figure()
    for i in range(len(spectra)):
        if L_label == []:
            plt.plot(wavelength, spectra[i], label = "spectrum {}".format(i + 1))
        else:
            plt.plot(wavelength, spectra[i], label = L_label[i])
    plt.xlabel("Wavelength")
    plt.ylabel("Spectra")
    plt.grid()
    plt.legend()
    
    return fig

def plot_for_article(image, colorbar = True, vmin = None, vmax = None, size=37):
    plt.rcParams["figure.figsize"] = [8, 8]

    fig = plt.figure()
    
    if vmin == None:
        im = plt.imshow(image)        
    else:
        im = plt.imshow(image, vmin = vmin, vmax = vmax)
        
    plt.axis("Off")
    
    if colorbar:
        cbar_ax = fig.add_axes([0.135, 0.065, 0.75, 0.03]) # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
        cbar.ax.tick_params(labelsize=size)

    return fig


def plot_for_article_difference(image, vext, size=37):
    plt.rcParams["figure.figsize"] = [8, 8]

    fig = plt.figure()
    
    im = plt.imshow(image, vmin = -vext, vmax = vext, cmap="seismic")      
        
    plt.axis("Off")
    
    cbar_ax = fig.add_axes([0.135, 0.065, 0.75, 0.03]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=size)

    return fig


def plot_for_article_abs_difference(image, vext, size=37):
    plt.rcParams["figure.figsize"] = [8, 8]

    fig = plt.figure()
    
    im = plt.imshow(np.abs(image), vmin = 0, vmax = vext, cmap="Reds")      
        
    plt.axis("Off")
    
    cbar_ax = fig.add_axes([0.135, 0.065, 0.75, 0.03]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=size)

    return fig

