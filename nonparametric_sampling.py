# venv "Coursera" .virtualenvs/Coursera is version matched to WinterShallDea conda env
# still raises LinAlgError("singular matrix")
# ???  could it be loading csv versus las ???
# what other packages are involved with scipy.stats.gaussian_kde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import gaussian_kde
import math
import statistics
from data_loading import well2_add_features
from packages.functions import *

import warnings
warnings.filterwarnings("ignore")

import scipy



#+++++++++++++++++++++++++++++++++++++#

print(f"scipy version: {scipy.__version__}")  # 1.8.1
print(f"numpy version: {np.__version__}")     # 1.22.4
print(f"pandas version: {pd.__version__}")
# print(f"pyplot version: {plt.__version__}")
print(f"seaborn version: {sns.__version__}")

#+++++++++++++++++++++++++++++++++++++#

# Parameters and rock properties
TEMP_RES = 70  #Celcius                                # or 77.2
PRESSURE_EFF = 16  #MPa                                # or 20
# PRESSURE_EFF = (den - den_fluid)*g x depth  # g=9.8

SALINITY = 80000  #PPM

# Fluid properties
RHO_BRINE = 1.09 # g/cm3
K_BRINE = 2.8  #GPa
RHO_OIL = 0.78  #g/cm3
K_OIL = 2.
OIL_GRAVITY = 19  #API                                # or 32
GOR = 100  #Sm3/Sm3                                   # or 64
GAS_GRAVITY = 0.6

# Rock properties
K_QUARTZ = 36.8  #GPa
MU_QUARTZ = 44  #GPa
K_CLAY = 15  #GPa
MU_CLAY = 5  #GPa

#+++++++++++++++++++++++++++++++++++++#

well1, well2, well3, well4, well5, well5_resist = well2_add_features()

#++++++++++++++#

facies_labels = ["shale", "sltShale", "clnSand", "sltSand1", "sltSand2", "cemSand"]

vp_vs_rho_phie_k0_0 = well2[well2.LABELS==facies_labels[0]]['VP'],\
                        well2[well2.LABELS==facies_labels[0]]['VS'],\
                        well2[well2.LABELS==facies_labels[0]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[0]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[0]]['K0']

vp_vs_rho_phie_k0_1 = well2[well2.LABELS==facies_labels[1]]['VP'],\
                        well2[well2.LABELS==facies_labels[1]]['VS'],\
                        well2[well2.LABELS==facies_labels[1]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[1]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[1]]['K0']

vp_vs_rho_phie_k0_2 = well2[well2.LABELS==facies_labels[2]]['VP'],\
                        well2[well2.LABELS==facies_labels[2]]['VS'],\
                        well2[well2.LABELS==facies_labels[2]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[2]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[2]]['K0']

vp_vs_rho_phie_k0_3 = well2[well2.LABELS==facies_labels[3]]['VP'],\
                        well2[well2.LABELS==facies_labels[3]]['VS'],\
                        well2[well2.LABELS==facies_labels[3]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[3]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[3]]['K0']

vp_vs_rho_phie_k0_4 = well2[well2.LABELS==facies_labels[4]]['VP'],\
                        well2[well2.LABELS==facies_labels[4]]['VS'],\
                        well2[well2.LABELS==facies_labels[4]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[4]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[4]]['K0']

vp_vs_rho_phie_k0_5 = well2[well2.LABELS==facies_labels[5]]['VP'],\
                        well2[well2.LABELS==facies_labels[5]]['VS'],\
                        well2[well2.LABELS==facies_labels[5]]['RHOB'],\
                        well2[well2.LABELS==facies_labels[5]]['PHIE'],\
                        well2[well2.LABELS==facies_labels[5]]['K0']

num_samples = 1000

kde_3d_layer0 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_0)
kde_3d_layer1 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_1)
kde_3d_layer2 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_2)
kde_3d_layer3 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_3)
kde_3d_layer4 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_4)
kde_3d_layer5 = scipy.stats.gaussian_kde(vp_vs_rho_phie_k0_5)

kde_3d_layer0_resample = kde_3d_layer0.resample(num_samples)
kde_3d_layer1_resample = kde_3d_layer1.resample(num_samples)
kde_3d_layer2_resample = kde_3d_layer2.resample(num_samples)
kde_3d_layer3_resample = kde_3d_layer3.resample(num_samples)
kde_3d_layer4_resample = kde_3d_layer4.resample(num_samples)
kde_3d_layer5_resample = kde_3d_layer5.resample(num_samples)

vp0,vs0,rho0,phie0,k00 = kde_3d_layer0_resample
vp1,vs1,rho1,phie1,k01 = kde_3d_layer1_resample
vp2,vs2,rho2,phie2,k02 = kde_3d_layer2_resample
vp3,vs3,rho3,phie3,k03 = kde_3d_layer3_resample
vp4,vs4,rho4,phie4,k04 = kde_3d_layer4_resample
vp5,vs5,rho5,phie5,k05 = kde_3d_layer5_resample

