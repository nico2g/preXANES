#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:26:26 2018

@author: nicolasgrossogiordano
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator
from glob import glob
import matplotlib.lines as lines
import pandas as pd
import scipy.optimize as opt


def maxfinder(energy,signal):
    #Find the maximum and the pre-edge position
    
    #Find within range of 2965 and 4975
    low_exc = 4965
    high_exc = 4972
    E_maxcenter = []
    S_maxcenter = []
    
    for index, value in enumerate(energy):
        if value > low_exc:
            if value < high_exc:
                E_maxcenter.append(value)
                S_maxcenter.append(signal[index])
    
    #Now find the maximum and its corresponding energy
    maximum = max(S_maxcenter)
    loc = S_maxcenter.index(maximum)
    center = E_maxcenter[loc]
    return maximum, center

def normalizer(material_name):
    #Normalizes the data
    
    #Use the coadder to add all data!
    energy, signal = coadder(material_name)

    # Now find the ranges for normalization
    pre_edge_values_Energy = []
    post_edge_values_Energy = []
    pre_edge_values_Signal = []
    post_edge_values_Signal = []
    
    #Pre edge values are 4820 - 4940, post edge from 5020 to 5760
    preL = 4820.
    preH = 4940.
    postL = 5020.
    postH = 5200.
    
    #Now scan through items and find for pre edge
    for index, value in enumerate(energy):
        if value < preL: #skip while energy is less than preL
            continue
        #After we found preL, now start adding until preH
        while value < preH:
            pre_edge_values_Energy.append(value)
            pre_edge_values_Signal.append(signal[index])
            break
    
    #Now scan through items and find for post edge
    for index, value in enumerate(energy):
        if value < postL: #skip while energy is less than preL
            continue
        #After we found preL, now start adding until preH
        while value < postH:
            post_edge_values_Energy.append(value)
            post_edge_values_Signal.append(signal[index])
            break
        
    #Now fit pre edge line to line
    x = np.array(pre_edge_values_Energy)
    y = np.array(pre_edge_values_Signal)
    p = np.polyfit(x,y,deg=1)
    pre_fit = energy*p[0] + p[1]
     
    #Now fit post edge line to 3rd degree polynomial
    x = np.array(post_edge_values_Energy)
    y = np.array(post_edge_values_Signal)
    p = np.polyfit(x,y,deg=2)
    post_fit = energy*energy*p[0] + energy*p[1] + p[2]
    
    
    #Calculate normalization constant
    edge = 4983.
    #Interpolate values
    e0_pre = np.interp(edge,energy,pre_fit)
    e0_post = np.interp(edge,energy,post_fit)
    norm_const_0 = (e0_post-e0_pre)
    
    #Now subtract and nomralize
    signal = signal - pre_fit
    signal = (signal - (post_fit-pre_fit))/norm_const_0
    minimum = min(signal)
    signal = signal - minimum
    return energy, signal

def coadder(material_name):
    
    #Merges data files
    
    #Get files to plot
    extension = material_name+'.*' #specify data extensions
    DataToPlot = glob(extension) # gets list of file names with extension
    
   
    
    # Iterate through all and extract the data
    k = 1.
    for datafile in DataToPlot:
        #Initiate variables on first go
        if k == 1:
            energy_temp, signal_temp = data_extract(datafile) 
            energy = np.array(energy_temp)
            signal = np.array(signal_temp)
            k = k+1
        else:
            energy_temp, signal_temp = data_extract(datafile) 
            energy = (energy + np.array(energy_temp))/k #average out E
            signal = signal + np.array(signal_temp)
    
    #Return the values of energy and signal
    return energy, signal
    

def data_extract(File):
    
    #Intialized variables
    headings = []
    headfind = 0 #0 nothing happens, 1 reads headings, 2 
    
    with open(File) as Data:
        for line in Data:
            #Read line by line, extract headings, extract data
            if 'Data:' in line: # Mark where actual data starts to find headings
                headfind = 1
                continue
            if headfind == 1:
                appendable = line.strip()
                headings.append(appendable)
            if 'Lytle' in line: #Marks the end of headings
                headfind = 2
                #Define dataframe with headings
                data = pd.DataFrame(columns=headings) # This will contain array of data
                continue
            if headfind == 2:
                appending = line.split()
                #Convert to float to avoid number errors
                for index, item in enumerate(appending):
                    appending[index] = float(item)
                
                #Skip line if empty space
                if len(appending) < 2:
                    continue
                
                #Append the line to the data
                else:
                    appendable = pd.DataFrame([appending], columns=headings)
                    data = data.append(appendable)
    
    #Get data from dataframe
    energy = data['Requested Energy']    
    Lytle_signal = data['Lytle']
    I0 = data['I0']

    #Signal nomrlize    
    signal = Lytle_signal/I0
    
    #return energy and singal
    return energy, signal

# Function definitions for objective functions in arctan, and also independent variation of amplitude, center, and sigma paramteres

#USE A LOGISTIC FUNCTION to fit everything. Note that legacy names related to previous arctan and gaussian fittins are used throughout the program, but the actual function here is a logistic function
def f_gauss_pre(x,cent,sigm,Amp):
    #x is data, cent is center, sigm is sigma, Amp is amplitude
    #return Amp*(0.5+(1/np.pi)*np.arctan((x-cent)/sigm))
    #return Amp*np.exp(-np.power((x - cent), 2.)/(2. * sigm**2.))
    alpha = (x-cent)/sigm
    return Amp*(1-1/(1+np.exp(alpha)))
    
def f_gauss_A(x,A_A1,A_A2,A_A3,A_B):
    #x is data, cent is center, sigm is sigma, Amp is amplitude
    A1 = A_A1*np.exp(-np.power((x - mu_A1), 2.)/(2. * sig_A1**2.))
    A2 = A_A2*np.exp(-np.power((x - mu_A2), 2.)/(2. * sig_A2**2.))
    A3 = A_A3*np.exp(-np.power((x - mu_A3), 2.)/(2. * sig_A3**2.))
    B = A_B*np.exp(-np.power((x - mu_B), 2.)/(2. * sig_B**2.))
    return A1+A2+A3+B

def f_gauss_mu(x,mu_A1,mu_A2,mu_A3,mu_B):
    #x is data, cent is center, sigm is sigma, Amp is amplitude
    A1 = A_A1*np.exp(-np.power((x - mu_A1), 2.)/(2. * sig_A1**2.))
    A2 = A_A2*np.exp(-np.power((x - mu_A2), 2.)/(2. * sig_A2**2.))
    A3 = A_A3*np.exp(-np.power((x - mu_A3), 2.)/(2. * sig_A3**2.))
    B = A_B*np.exp(-np.power((x - mu_B), 2.)/(2. * sig_B**2.))
    return A1+A2+A3+B

def f_gauss_sig(x,sig_A1,sig_A2,sig_A3,sig_B):
    #x is data, cent is center, sigm is sigma, Amp is amplitude
    A1 = A_A1*np.exp(-np.power((x - mu_A1), 2.)/(2. * sig_A1**2.))
    A2 = A_A2*np.exp(-np.power((x - mu_A2), 2.)/(2. * sig_A2**2.))
    A3 = A_A3*np.exp(-np.power((x - mu_A3), 2.)/(2. * sig_A3**2.))
    B = A_B*np.exp(-np.power((x - mu_B), 2.)/(2. * sig_B**2.))
    return A1+A2+A3+B

#Funciton defintions for plotting individual components

def plot_A1(x):
    return A_A1*np.exp(-np.power((x - mu_A1), 2.)/(2. * sig_A1**2.))

def plot_A2(x):
    return A_A2*np.exp(-np.power((x - mu_A2), 2.)/(2. * sig_A2**2.))

def plot_A3(x):
    return A_A3*np.exp(-np.power((x - mu_A3), 2.)/(2. * sig_A3**2.))

def plot_B(x):
    return A_B*np.exp(-np.power((x - mu_B), 2.)/(2. * sig_B**2.))

def sum_residuals(x,data):
    #Returns sum of squers of the residuals where x is the x data and data is the y data. x is inputted into function in order to calculate model.
    A1 = plot_A1(x)
    A2 = plot_A2(x)
    A3 = plot_A3(x)
    B = plot_B(x)
    
    fit = A1+A2+A3+B
    sumresidual = 0.
    
    for index, value_fit in enumerate(fit):
        residual = np.power((value_fit - data[index]),2)
        sumresidual = sumresidual + residual
    
    return sumresidual

def pre_edge_fitter(energy,signal):
        # pre_edge_fitter is a function that takes as input the value energy and signal data and returns the fitted preedge with values of fitted energy, pre edge step fit, and  A1, A2, A3, A4.
    
    #Truncate energy and signal to relevant fitting range
    low = 4940.
    high = 4979.
    energy_fit = []
    signal_fit = []
    
    #Now scan through items and find data for pre edge
    for index, value in enumerate(energy):
        if value < low: #skip while energy is less than preL
            continue
        #After we found preL, now start adding until preH
        while value < high: 
            energy_fit.append(value)
            signal_fit.append(signal[index])
            break
    
    #Remove items from 4965 to 4975 that will not be used in arctan
    low_exc = 4965
    high_exc = 4975
    E_atan = []
    S_atan = []
    
    for index, value in enumerate(energy_fit):
        if value > low_exc:
            if value < high_exc:
                continue
        
        E_atan.append(value)
        S_atan.append(signal_fit[index])
        
    #Fit the arctan function
    popt, pcov = opt.curve_fit(f_gauss_pre,E_atan,S_atan,p0 = [4975.,0.5,.5],maxfev = 5000)
    
    
    #START FITTING THE EDGE
    
    # First, subtract the atan function
    energy_fit = np.array(energy_fit)
    ATANfit = f_gauss_pre(energy_fit,popt[0],popt[1],popt[2])
    fit_peaks_data = signal_fit - ATANfit
    
    #parameters are A_A1,mu_A1,sig_A1,A_A2,mu_A2,sig_A2,A_A3,mu_A3,sig_A3,A_B,mu_B,sig_B
    #Ti K-edge was fit to an arctan function, while the pre-edge features were fit to four Gaussian peaks, labeled A1, A2, A3, and B, in order of ascending energy, with centers constrained to lie within, respectively, 4968–4969 eV, 4969–4971 eV, 4971–4972, and 4973–4974 eV
    
    global A_A1, A_A2, A_A3, A_B, mu_A1, sig_A1, mu_A2, sig_A2, mu_A3, sig_A3, mu_B, sig_B #Set parameters to global for use in functions
    
    #Initialize all parameters
    A_A1= 0.1
    A_A2 = 0.6
    A_A3 = 0.2
    A_B = 0.1
    mu_A1 = 4968.5
    sig_A1 = 1
    mu_A2 = 4970
    sig_A2 = 1
    mu_A3 = 4971.5
    sig_A3 = 1
    mu_B = 4973.5
    sig_B = 1
    
    
    RESIDUAL_for_criterion = [1.]
    RESIDUAL_criterion = 0.
    
    # Now imput allguesses
    guess_A = [0.1,0.6,0.2,0.1] #Intial guesses for amplitude A1, A2, A3, B
    limits_A = ([0.01,0.01,0.01,0.01],[1,1,1,1]) #[lower],[upper] bounds of amplitued
    #Guess, bound for mu
    guess_mu = [4968.5,4970,4971.5,4973.5]
    limits_mu = ([4968,4969,4971,4973],[4969,4971,4972,4974])
    
    #Guess bound for sigma
    guess_sig = [1.,1.,1.,1.]
    limits_sig = ([0.1,0.1,0.1,0.1],[1.0,1.0,1.0,1.0])

    # Index number of iterations
    k = 0
    
    #In a while loop, keep on iterating while we do not satisfy the criterion of getting close to no change in parameters
    while RESIDUAL_criterion <= 0.995:
    
        #Start by varying the edge height
        popt_A, pcov_A = opt.curve_fit(f_gauss_A,energy_fit,fit_peaks_data,p0=guess_A,bounds=limits_A)
        A_A1, A_A2, A_A3, A_B = popt_A #extract values
        guess_A = popt_A # New guess of A will be the value of the extracted parameters
        
        #Now fit mu (center)
        popt_mu, pcov_mu = opt.curve_fit(f_gauss_mu, energy_fit, fit_peaks_data,p0 = guess_mu,bounds=limits_mu,method='dogbox')
        mu_A1, mu_A2, mu_A3, mu_B = popt_mu
        guess_mu = popt_mu #Update guess
        
        #Now fit sig (center) 
        popt_sig, pcov_sig = opt.curve_fit(f_gauss_sig, energy_fit, fit_peaks_data,p0 = guess_sig,bounds=limits_sig,method='dogbox')
        sig_A1, sig_A2, sig_A3, sig_B = popt_sig
        guess_sig = popt_sig #Uptdate guess
        
        # Now calculate residual cirterion and add it to the one before
        RESIDUAL_for_criterion.append(sum_residuals(energy_fit,fit_peaks_data))
        
        #Decide if we are done accoridng to a residual criterion based on whetehr there is a change in calues of the sum of squer of the residual calculated above. If change is negligible, the residual criterion will be .99 or higher becaue there will be no change.
        RESIDUAL_criterion = RESIDUAL_for_criterion[-1]/RESIDUAL_for_criterion[-2]
        k = k+1    
    
    # Calculate individual componetns for fitting
    fitted_A1 = plot_A1(energy_fit)
    fitted_A2 = plot_A2(energy_fit)
    fitted_A3 = plot_A3(energy_fit)
    fitted_B = plot_B(energy_fit)
    
    return energy_fit, ATANfit, fitted_A1, fitted_A2, fitted_A3, fitted_B, A_A1, A_A2, A_A3, A_B