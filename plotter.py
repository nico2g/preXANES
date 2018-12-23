#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:31:48 2018

@author: nicolasgrossogiordano
"""

# Import necessar functionalities
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator
from glob import glob
import matplotlib.lines as lines
import pandas as pd
import scipy.optimize as opt
import os

from XAS_functions import *

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']



#HERE IS THE MAIN CODE
if __name__ == '__main__':
        
    #Import files with the text extension in this folder

    extension = '*.001' #grab the names by considering the 001 that all files start with
    Materials = glob(extension) # gets list of file names with extension
    
    #Make dictionary for plotting. Note that file formats are spearated by '_'; otherwise modify code below
    sorter = {}
    plotorder = []
    for DataSetSort in Materials:
        splitname = DataSetSort.split('_')
        materialname = splitname[-1].split('.')[0]
        sorter[materialname] = DataSetSort
        plotorder.append(materialname)
    
    #Define order to plot. Can be a preset or come from Materials. Use line below if you want to redefine
    #plotorder = ['Example1','Example2'] 
    
    #Prepare for XANES plotting
    
    # Plots staggered
    fig1 = plt.figure(1,figsize=(3.33,5))
    ax = plt.subplot(111)
    
    # Plots overlayed
    fig2 = plt.figure(2,figsize=(3.33,3.33))
    ax_1 = plt.subplot(111)
    fig2.tight_layout()
    
    # Plots fits
    fig3 = plt.figure(3,figsize=(10,7))
    
    axes = []
    for i in range(len(plotorder)):
        axes.append(plt.subplot(2,3,i+1))
   
    #Define the colorcoding to be used
    colorcode = {'1':'r','2':'k'} # Can be expanded for more colors
    
    k = 0 # keep track of iteration
    for iteration in plotorder:
        
        #Extract from diciontary
        materialname = sorter[iteration]
        nameholder = materialname.split('.')[0]
        print(nameholder)
    
        #Import the energy and singnals
        energy, signal = normalizer(nameholder)
        
        #Get the fit and data
        energy_fit, ATANfit, fitted_A1, fitted_A2, fitted_A3, fitted_B, A_A1, A_A2,A_A3,A_B = pre_edge_fitter(energy,signal)
        
        #Calculate the total fit
        totalfit = ATANfit + fitted_A1 + fitted_A2 + fitted_A3 + fitted_B
        A_rel = (A_A2+A_A3)/(A_A1+A_A2+A_A3+A_B)    
    
        #Find the max
        maximum, center = maxfinder(energy,signal)
        
        
        #Make label
        label1 = nameholder.split('_')[1]
        label2 = 'A$_{rel}$= '+str(round(A_rel,2))
        label3 = str(round(maximum,2))+', '+str(center)+' eV'
        
        #Plot the XANES fit--------------------------------------

        # Plot data
        marks = axes[k].scatter(energy,signal,marker='o',s=30,edgecolors='k',facecolors='none')
        axes[k].plot(energy_fit,ATANfit,'purple')
        axes[k].plot(energy_fit,fitted_A1,'green')
        axes[k].plot(energy_fit,fitted_A2,'blue')
        axes[k].plot(energy_fit,fitted_A3,'red')
        axes[k].plot(energy_fit,fitted_B,'orange')
        axes[k].plot(energy_fit,totalfit,'black')
        axes[k].set_xlim(4965,4978)
        axes[k].set_ylim(0,1.0)
        axes[k].text(0.98,0.9,label1,transform=axes[k].transAxes, horizontalalignment='right', fontsize=14,weight='bold')
        axes[k].text(0.98,0.83,label2,transform=axes[k].transAxes, horizontalalignment='right', fontsize=12)
        axes[k].text(0.98,0.76,label3,transform=axes[k].transAxes, horizontalalignment='right', fontsize=12)
        
        
        axes[k].set_xlabel('energy, eV',fontsize=12)
        axes[k].set_ylabel('normalized intensity, a.u.',fontsize=12)
        axes[k].xaxis.set_minor_locator(AutoMinorLocator(5)) #Five minor ticks in between
        axes[k].tick_params(axis='x', which='both', pad=5, direction='out',labelsize=12) #Inwards tick parameters
    
        #Manipulate x axis
        axes[k].tick_params(axis='y', which='both', left='off', labelleft='off', labelsize=12) #Y ticks are off

        plt.tight_layout()
        #Plot the XANES staggered --------------------------------------
        
        #Increase the value of signal for subsequent plots
        
        signal = signal + k*0.5
        style = colorcode[iteration]
        
        ax.plot(energy,signal,style)
        
        #Plot a line for the main peak
        ax.axvline(x=4970.1, color='grey',linestyle='--',linewidth = 0.5) # pre-edge
#        ax.axvline(x=4979.0, color='grey',linestyle='--',linewidth = 0.5)
        
        ax.set_xlim(4955,5000)
        ax.set_ylim(0,3.3)
        ax.text(4956,k*0.5+0.05,label1,fontsize=12,weight='bold')
        
        
        ax.set_xlabel('energy, eV',fontsize=12)
        ax.set_ylabel('normalized intensity, a.u.',fontsize=12)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5)) #Five minor ticks in between
        ax.tick_params(axis='x', which='both', pad=5, direction='out',labelsize=10) #Inwards tick parameters
        
        #Manipulate y axis
        ax.tick_params(axis='y', which='both', left='off', labelleft='off', labelsize=10) #Y ticks are off
        
        #Plot the XANES overlapped --------------------------------------
        
        #Increase the value of signal for subsequent plots

        signal = signal - k*0.5
        
        ax_1.plot(energy,signal,style)
        ax_1.set_xlim(4955,5000)
        
        ax_1.set_xlabel('energy, eV',fontsize=12)
        ax_1.set_ylabel('normalized intensity, a.u.',fontsize=12)
        ax_1.xaxis.set_minor_locator(AutoMinorLocator(5)) #Five minor ticks in between
        ax_1.tick_params(axis='x', which='both', pad=5, direction='out',labelsize=10) #Inwards tick parameters
    
        #Manipulate x axis
        ax_1.tick_params(axis='y', which='both', left='off', labelleft='off',labelsize=10)
        ax_1.set_ylim(0,1.5)
        k = k+1
    
    
    # Add (A) lable to graph if necessary
    ax.text(0.05,0.93,'(B)',fontsize=14,weight='bold',transform=ax.transAxes)
    
    plt.show()
    fig1.savefig('StaggerXASplot', dpi=600, format='png')
    fig2.savefig('OverlayXASplot', dpi=600, format='png')
    fig3.savefig('PeakFits', dpi=600, format='png')