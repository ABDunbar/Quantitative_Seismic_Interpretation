import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# define basic styles for plotting log curves (sty0), sand (sty1) and shale (sty2)
sty0 = {'lw': 1, 'color': 'k', 'ls': '-'}
sty1 = {'marker': 'o', 'color': 'g', 'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}
sty2 = {'marker': 'o', 'color': 'r', 'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}

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

Cn = 8
phic = 0.4
f = 1

phi = np.linspace(0.01, 0.4)


def vrh(f, M1, M2):
    """
    Simple Voigt-Reuss-Hill bounds for 2-components mixture, (C) aadm 2017

    INPUT
    f: volumetric fraction of mineral 1
    M1: elastic modulus mineral 1
    M2: elastic modulus mineral 2

    OUTPUT
    M_Voigt: upper bound or Voigt average
    M_Reuss: lower bound or Reuss average
    M_VRH: Voigt-Reuss-Hill average
    """
    M_Voigt = f * M1 + (1 - f) * M2
    M_Reuss = 1 / (f / M1 + (1 - f) / M2)
    M_VRH = (M_Voigt + M_Reuss) / 2
    return M_Voigt, M_Reuss, M_VRH


def vels(K_DRY, G_DRY, K0, D0, Kf, Df, phi):
    """
    Calculates velocities and densities of saturated rock via Gassmann equation, (C) aadm 2015

    INPUT
    K_DRY,G_DRY: dry rock bulk & shear modulus in GPa
    K0, D0: mineral bulk modulus and density in GPa
    Kf, Df: fluid bulk modulus and density in GPa
    phi: porosity
    """
    rho = D0 * (1 - phi) + Df * phi
    K = K_DRY + (1 - K_DRY / K0) ** 2 / ((phi / Kf) + ((1 - phi) / K0) - (K_DRY / K0 ** 2))
    vp = np.sqrt((K + 4. / 3 * G_DRY) / rho) * 1e3
    vs = np.sqrt(G_DRY / rho) * 1e3
    return vp, vs, rho, K


def hertzmindlin(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    """
    Hertz-Mindlin model
    written by aadm (2015) from Rock Physics Handbook, p.246

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    """
    P /= 1e3  # converts pressure in same units as solid moduli (GPa)
    PR0 = (3 * K0 - 2 * G0) / (6 * K0 + 2 * G0)  # poisson's ratio of mineral mixture
    K_HM = (P * (Cn ** 2 * (1 - phic) ** 2 * G0 ** 2) / (18 * np.pi ** 2 * (1 - PR0) ** 2)) ** (1 / 3)
    G_HM = ((2 + 3 * f - PR0 * (1 + 3 * f)) / (5 * (2 - PR0))) * (
        (P * (3 * Cn ** 2 * (1 - phic) ** 2 * G0 ** 2) / (2 * np.pi ** 2 * (1 - PR0) ** 2))) ** (1 / 3)
    return K_HM, G_HM


def softsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    """
    Soft-sand (uncemented) model
    written by aadm (2015) from Rock Physics Handbook, p.258

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    """
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY = -4 / 3 * G_HM + (((phi / phic) / (K_HM + 4 / 3 * G_HM)) + ((1 - phi / phic) / (K0 + 4 / 3 * G_HM))) ** -1
    tmp = G_HM / 6 * ((9 * K_HM + 8 * G_HM) / (K_HM + 2 * G_HM))
    G_DRY = -tmp + ((phi / phic) / (G_HM + tmp) + ((1 - phi / phic) / (G0 + tmp))) ** -1
    return K_DRY, G_DRY


def stiffsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    """
    Stiff-sand model
    written by aadm (2015) from Rock Physics Handbook, p.260

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1=dry pack with perfect adhesion
       0=dry frictionless pack
    """
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY = -4 / 3 * G0 + (((phi / phic) / (K_HM + 4 / 3 * G0)) + ((1 - phi / phic) / (K0 + 4 / 3 * G0))) ** -1
    tmp = G0 / 6 * ((9 * K0 + 8 * G0) / (K0 + 2 * G0))
    G_DRY = -tmp + ((phi / phic) / (G_HM + tmp) + ((1 - phi / phic) / (G0 + tmp))) ** -1
    return K_DRY, G_DRY


well2 = pd.read_csv("../data/well_2.txt", header=None, skiprows=1, sep="  ", usecols=[0, 3, 6, 9, 11, 14])
well2.rename(columns={0: "DEPTH", 3: "VP", 6: "VS", 9: "DEN", 11: "GR", 14: "NPHI"}, inplace=True)
well2["VP"] = well2.VP * 1000
well2["VS"] = well2.VS * 1000
well2["PHIE"] = (2.65 - well2.DEN) / (2.65 - 1.05)
well2["VSH"] = (well2.GR - well2.GR.min()) / (well2.GR.max() - well2.GR.min())
well2["IP"] = well2.VP * well2.DEN
well2["IS"] = well2.VS * well2.DEN
well2["VPVS"] = well2.VP / well2.VS
well2['K'] = well2.DEN * (well2.VP ** 2 - 4 / 3. * (well2.VS) ** 2)
well2['sandy-shaly'] = np.where(well2['VSH'] >= 0.35, 'shaly', 'sandy')
L = well2.copy()

K0, MU0, RHO0 = K_qz, MU_qz, RHO_qz

Kdry, MUdry = softsand(K0, MU0, phi, phic, Cn, P=45)
vp_ssm, vs_ssm, rho_ssm, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)

Kdry, MUdry = stiffsand(K0, MU0, phi, phic, Cn, P=45)
vp_sti, vs_sti, rho_sti, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)

z1, z2, cutoff_sand, cutoff_shale = 2100, 2250, 0.3, 0.5

ss = (L.index >= z1) & (L.index <= z2) & (L.VSH <= cutoff_sand)
sh = (L.index >= z1) & (L.index <= z2) & (L.VSH >= cutoff_shale)

f, ax = plt.subplots(figsize=(5, 5))
ax.plot(L.PHIE[ss], L.VP[ss], **sty1)
ax.plot(phi, vp_ssm, '-k')
ax.plot(phi, vp_sti, ':k')
ax.set_xlim(0, 0.4), ax.set_ylim(2e3, 4e3)
ax.set_xlabel('porosity PHIE')
ax.set_ylabel('velocity VP [m/s]')
ax.set_title('Soft Sand and Stiff Sand models')

plt.show()
