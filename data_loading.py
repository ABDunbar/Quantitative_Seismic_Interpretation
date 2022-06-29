# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from packages.functions import vshale_from_gr, vrh
from load_las import load

# Rock properties
K_QUARTZ = 36.8  # GPa
MU_QUARTZ = 44  # GPa
K_CLAY = 15  # GPa
MU_CLAY = 5  # GPa


def well2_add_features():
    well1, well2, well3, well4, well5, well5_resist = load()

    # well2 adding features

    well2 = vshale_from_gr(well2)

    well2["PHIE"] = (2.65 - well2.RHOB) / (2.65 - 1.05)
    well2["IP"] = well2.VP * 1000 * well2.RHOB
    well2["IS"] = well2.VS * 1000 * well2.RHOB
    well2["VPVS"] = well2.VP / well2.VS
    # well2['K'] = well2.DEN*(well2.VP**2 - 4/3.*(well2.VS)**2)  # ?? what is this bulk modulus ??

    well2['sandy-shaly'] = np.where(well2['VSH'] >= 0.35, 'shaly', 'sandy')

    shale = well2.VSH.values  #
    # shale = df.VWCL.values

    sand = 1 - shale - well2.PHIE.values
    shaleN = shale / (shale + sand)  # normalized shale and sand volumes
    sandN = sand / (shale + sand)

    # mineral mixture bulk and shear moduli, k0 and mu0  # ?? A mineral mixture BUT NOT dry rock bulk modulus??
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [K_CLAY, K_QUARTZ], [MU_CLAY, MU_QUARTZ])

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

    return well1, well2, well3, well4, well5, well5_resist
