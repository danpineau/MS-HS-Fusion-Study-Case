#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:02:25 2025

@author: dpineau
"""

import os
import numpy as np
import torch

class Saver():
    def __init__(self, main_path):
        self.main_path = main_path
    
    def save_numpy_file(self, name, file, check_if_exists = True, verbose = True):
        full_name = self.main_path + name + ".npy"
        if os.path.exists(full_name) and check_if_exists:
            print("Numpy file already exists ! File not saved !")
            print("File name : " + name + ".npy")
        else:
            np.save(full_name, file)
            if verbose:
                print("Numpy file saved.")
                print("File name : " + name + ".npy")
                
    def save_torch_file(self, name, file, check_if_exists = True, verbose = True):
        full_name = self.main_path + name + ".pt"
        if os.path.exists(full_name) and check_if_exists:
            print("Torch file already exists ! File not saved !")
            print("File name : " + name + ".pt")
        else:
            torch.save(file, full_name)
            if verbose:
                print("Torch file saved.")
                print("File name : " + name + ".pt")
    
    def save_figure(self, name, fig, check_if_exists = True):
        full_name = self.main_path + name + ".pdf"
        if os.path.exists(full_name) and check_if_exists:
            print("Figure already exists ! File not saved !")
            print("File name : " + name + ".pdf")
        else:
            fig.savefig(full_name, bbox_inches='tight', pad_inches=0)
            print("Figure saved.")
            print("File name : " + name + ".pdf")
            
    def save_figure_png(self, name, fig, check_if_exists = True):
        full_name = self.main_path + name + ".png"
        if os.path.exists(full_name) and check_if_exists:
            print("Figure already exists ! File not saved !")
            print("File name : " + name + ".png")
        else:
            fig.savefig(full_name, bbox_inches='tight', pad_inches=0)
            print("Figure saved.")
            print("File name : " + name + ".png")
    
    def save_checkpoint(self, name, checkpoint, check_if_exists = True):
        full_name = self.main_path + name + '.pth'
        if os.path.exists(full_name) and check_if_exists:
            print("Checkpoint already exists ! File not saved !")
            print("File name : " + name + '.pth')
        else:
            torch.save(checkpoint, full_name)
            print("Checkpoint saved.")
            print("File name : " + name + '.pth')

class Loader():
    def __init__(self, main_path):
        self.main_path = main_path
    
    def load_numpy_file(self, name):
        full_name = self.main_path + name + ".npy"
        if os.path.exists(full_name) == False:
            print("Numpy file does not exist!")
            print("Full file name : " + full_name)
        else:
            print("Numpy file loaded :", full_name)
            return np.load(full_name)
        
    def load_torch_file(self, name):
        full_name = self.main_path + name + ".pt"
        if os.path.exists(full_name) == False:
            print("Torch file does not exist!")
            print("Full file name : " + full_name)
        else:
            print("Torch file loaded :", full_name)
            return torch.load(full_name)
    
    def load_checkpoint(self, name):
        full_name = self.main_path + name + '.pth'
        if os.path.exists(full_name) == False:
            print("Checkpoint does not exist!")
            print("Full file name : " + full_name)
        else:
            print("Checkpoint loaded :", full_name)
            return torch.load(full_name)