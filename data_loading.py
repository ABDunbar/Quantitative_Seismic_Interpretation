import warnings

import numpy as np
import pandas as pd
import platform
import os
from packages.functions import frm, twolayer, r0g, vshale_from_gr, vrh

# warnings.filterwarnings("ignore")
import pip
try:
    __import__(lasio)
except ImportError:
    pip.main(['install', lasio])
# pip install lasio
# import lasio as lasio


rho_qz = 2.65
k_qz = 37
mu_qz = 44  # mineral properties, quartz (i.e., sands)
rho_sh = 2.81
k_sh = 15
mu_sh = 5  # mineral properties, clay (i.e., shales)
rho_b = 1.09
k_b = 2.8  # fluid properties, brine
rho_o = 0.78
k_o = 0.94  # fluid properties, oil
rho_g = 0.25
k_g = 0.06  # fluid properties, gas

RHO_qz = 2.6
K_qz = 37
MU_qz = 44
RHO_sh = 2.8
K_sh = 15
MU_sh = 5
RHO_b = 1.1
K_b = 2.8
RHO_o = 0.8
K_o = 0.9
RHO_g = 0.2
K_g = 0.06

# Parameters and rock properties
TEMP_RES = 70  # Celcius                                # or 77.2
PRESSURE_EFF = 16  # MPa                                # or 20
# PRESSURE_EFF = (den - den_fluid)*g x depth  # g=9.8

SALINITY = 80000  # PPM

# Fluid properties
RHO_BRINE = 1.09  # g/cm3
K_BRINE = 2.8  # GPa
RHO_OIL = 0.78  # g/cm3
K_OIL = 2.
OIL_GRAVITY = 19  # API                                # or 32
GOR = 100  # Sm3/Sm3                                   # or 64
GAS_GRAVITY = 0.6

# Rock properties
K_QUARTZ = 36.8  # GPa
MU_QUARTZ = 44  # GPa
K_CLAY = 15  # GPa
MU_CLAY = 5  # GPa

# dropbox = '%HOMEPATH%\\Dropbox'
data_folder = 'Work/QSI/data'
my_os = platform.system()
if my_os == 'Windows':
    path = r"C:\Users\Dunbar\Dropbox\Work\QSI\data"
elif my_os == 'Linux':
    path = '/home/alex/Dropbox/Work/QSI/data'

print(f"Platform system : {my_os}")
print(f"file path: {path}")
print(f"OS name: {os.name}")  # windows = 'nt', linux = 'posix'
print(f"Dropbox folder location: {dropbox}")

# # at work
# path = r'C:\Users\ALDUNBAR\OneDrive - Wintershall Dea AG\Python\data'
# file = 'well-3.las'


file = 'well_3.las'
path_file = path + "\\" + file

well1 = pd.read_csv("./data/well_1.txt", header=None, skiprows=1, sep="  ", usecols=[0, 4, 8, 11])
well1.rename(columns={0: "DEPTH", 4: "VP", 8: "DEN", 11: "GR"}, inplace=True)

well2 = pd.read_csv("./data/well_2.txt", header=None, skiprows=1, sep="  ", usecols=[0, 3, 6, 9, 11, 14])
well2.rename(columns={0: "DEPTH", 3: "VP", 6: "VS", 9: "DEN", 11: "GR", 14: "NPHI"}, inplace=True)

# well3 = pd.read_csv("./data/well_3.txt", header=None, skiprows=1, sep=" ", usecols=[5,13,17,24]) # sep="  ")#,
# well3.rename(columns={5:"DEPTH", 13:"VP", 17:"DEN", 24:"GR"}, inplace=True)

well = lasio.read(path_file)
well3 = well.df()
well3 = well3.replace(-9999.0, np.NaN)
well3 = well3[(well3.index > 2100) & (well3.index < 2300)]
# move DEPT from index to column
well3.reset_index(inplace=True)
well3.rename(columns={'DEPT': 'DEPTH'}, inplace=True)

well4 = pd.read_csv("./data/well_4.txt", header=None, skiprows=1, sep=" ", usecols=[4, 12, 20, 27])  # sep="  ")#,
well4.rename(columns={4: "DEPTH", 12: "VP", 20: "DEN", 27: "GR"}, inplace=True)

well5 = pd.read_csv("./data/well_5.txt", header=None, skiprows=1, sep=" ", usecols=[3, 6, 9, 12, 15])  # sep="  ")#,
well5.rename(columns={3: "DEPTH", 6: "t_p", 9: "t_s", 12: "GR", 15: "DEN"}, inplace=True)

well2 = vshale_from_gr(well2)

well2["PHIE"] = (2.65 - well2.DEN) / (2.65 - 1.05)
# well2["VSH"] = (well2.GR - well2.GR.min()) / (well2.GR.max()-well2.GR.min())
well2["IP"] = well2.VP * 1000 * well2.DEN
well2["IS"] = well2.VS * 1000 * well2.DEN
well2["VPVS"] = well2.VP / well2.VS
# well2['K'] = well2.DEN*(well2.VP**2 - 4/3.*(well2.VS)**2)  # ?? what is this bulk modulus ??

well2['sandy-shaly'] = np.where(well2['VSH'] >= 0.35, 'shaly', 'sandy')

#
shale = well2.VSH.values  #
# shale = df.VWCL.values

sand = 1 - shale - well2.PHIE.values
shaleN = shale / (shale + sand)  # normalized shale and sand volumes
sandN = sand / (shale + sand)

# mineral mixture bulk and shear moduli, k0 and mu0  # ?? A mineral mixture BUT NOT dry rock bulk modulus??
k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [K_CLAY, K_QUARTZ], [MU_CLAY, MU_QUARTZ])
# mineral mixture bulk and shear moduli, k0 and mu0
# k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

well2['K0'] = k0

## well2 should be
# Facies I: Gravels => not used
# Facies II: Thick bedded sandstones; IIa, IIb, IIc, IId
# Facies III: Interbedded sandstone-shale
# Facies IV: Silty shales and silt-laminated shale
# Facies V: Pure shales
# Facies VI: Chaotic deposits => not used


conditions = [
    (well2["DEPTH"].ge(2078.0) & well2["DEPTH"].lt(2105.0)),
    (well2["DEPTH"].ge(2143.2) & well2["DEPTH"].lt(2154.1)),
    (well2["DEPTH"].ge(2154.1) & well2["DEPTH"].lt(2164.1)),
    (well2["DEPTH"].ge(2168.1) & well2["DEPTH"].lt(2184.1)),
    (well2["DEPTH"].ge(2186.1) & well2["DEPTH"].lt(2200.1)),
    (well2["DEPTH"].ge(2254.0) & well2["DEPTH"].lt(2300.1)),
]
# facies = ["shale", "sltShale", "clnSand", "sltSand1", "sltSand2", "cemSand"]
facies = [1, 2, 3, 4, 5, 6]  # == FCODES[1:]
well2["FACIES"] = np.select(conditions, facies)

reservoir = [0, 0, 1, 1, 0, 1]
well2["RESERVOIR"] = np.select(conditions, reservoir)

facies_labels = ["shale", "sltShale", "clnSand", "sltSand1", "sltSand2", "cemSand"]
well2["LABELS"] = np.select(conditions, facies_labels)
# well2["FACIES"] = np.select(conditions, facies)  # can I use FCODES[1:] ???

facies_codes = [6, 0, 6, 1, 2, 6, 3, 6, 4, 6, 5, 6]  # for well log plot
conditions = [
    (well2["DEPTH"].ge(well2.DEPTH.min()) & well2["DEPTH"].lt(2078.0)),  # undef=0    6
    (well2["DEPTH"].ge(2078.0) & well2["DEPTH"].lt(2105.0)),  # shale=1    0
    (well2["DEPTH"].ge(2105.0) & well2["DEPTH"].lt(2143.2)),  # undef=0    6
    (well2["DEPTH"].ge(2143.2) & well2["DEPTH"].lt(2154.1)),  # sltShale=2 1
    (well2["DEPTH"].ge(2154.1) & well2["DEPTH"].lt(2164.1)),  # clnSand=3  2
    (well2["DEPTH"].ge(2164.1) & well2["DEPTH"].lt(2168.1)),  # undef=0    6
    (well2["DEPTH"].ge(2168.1) & well2["DEPTH"].lt(2184.1)),  # sltSand1=4 3
    (well2["DEPTH"].ge(2184.1) & well2["DEPTH"].lt(2186.1)),  # undef=0    6
    (well2["DEPTH"].ge(2186.1) & well2["DEPTH"].lt(2200.1)),  # sltSand2=5 4
    (well2["DEPTH"].ge(2200.1) & well2["DEPTH"].lt(2254.0)),  # undef=0    6
    (well2["DEPTH"].ge(2254.0) & well2["DEPTH"].lt(2300.1)),  # cemSand=6  5
    (well2["DEPTH"].ge(2300.1) & well2["DEPTH"].lt(well2.DEPTH.max()))  # undef=0    6
]
well2["FCODES"] = np.select(conditions, facies_codes)

print(well2[well2.LABELS != '0'].head())
print(well2.columns)
