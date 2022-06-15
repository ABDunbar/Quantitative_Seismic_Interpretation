# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

RHO_qz = 2.6;   K_qz = 37;  MU_qz = 44
RHO_sh = 2.8;   K_sh = 15;  MU_sh = 5
RHO_b = 1.1 ;   K_b = 2.8 
RHO_o = 0.8;    K_o = 0.9
RHO_g = 0.2;    K_g = 0.06              

Cn = 8
phic = 0.4
f = 1

phi=np.linspace(0.01, 0.4)


# define basic styles for plotting log curves (sty0), sand (sty1) and shale (sty2)
sty0 = {'lw':1, 'color':'k', 'ls':'-'}
sty1 = {'marker':'o', 'color':'g', 'ls':'none', 'ms':6, 'mec':'none', 'alpha':0.5}
sty2 = {'marker':'o', 'color':'r', 'ls':'none', 'ms':6, 'mec':'none', 'alpha':0.5}

def rplog_convert(df, pwave_sonic, shear_sonic):
    """
    Convert usec/ft (DT/DTS) to velocity (VP/VS)
    Create Impedance logs from velocity and density logs (IP/IS)
    Create VP/VS ratio log
    """
    try:
        df['VP'] = 304800 / df[pwave_sonic]
        df["IP"] = df.VP * df.RHOB
        df['VS'] = 304800 / df[shear_sonic]
        df['VPVS'] = df.VP / df.VS    
        df['IS'] = df.VS * df.RHOB
    except Exception as e:
        print(f"Error when creating log: {e}")
    
    return df

def density_porosity(input_density, matrix_density=2.65, fluid_density=1):
    """
    from Andy McDonald github repository on Petrophysics
    """
    denpor = (matrix_density - input_density) / (matrix_density - fluid_density)
    return denpor


def vshale_from_gr(df):
    """
    Creates Clavier, Larionov old, Larionov new, Steiber VSH
    """    
    GR_min = df.GR.min()
    GR_max = df.GR.max()
    df.loc[:, 'IGR'] = (df.GR - GR_min) / (GR_max - GR_min)
    df.loc[:, 'VSH_clavier'] = 1.7 - ((3.38 - (df.IGR + 0.7)**2)**0.5)
    df.loc[:, 'VSH_larionovO'] = 0.33 * (2**(2*df.IGR)-1)
    df.loc[:, 'VSH_steiber'] = df.IGR / (3 - 2*df.IGR)
    df.loc[:, 'VSH_larionovT'] = 0.083*(2**(3.7*df.IGR)-1)
    # Pick one to be "main" VSH
    df['VSH'] = df.VSH_larionovO
    return df


def plotlog(L, z1, z2, cutoff_sand, cutoff_shale): 
    # define filters to select sand (ss) and shale (sh)
    ss = (L.index>=z1) & (L.index<=z2) & (L.VSH<=cutoff_sand)
    sh = (L.index>=z1) & (L.index<=z2) & (L.VSH>=cutoff_shale)
    
    # plot figure    
    f = plt.subplots(figsize=(14, 6))
    ax0 = plt.subplot2grid((1,9), (0,0), colspan=1) # gr curve
    ax1 = plt.subplot2grid((1,9), (0,1), colspan=1) # ip curve
    ax2 = plt.subplot2grid((1,9), (0,2), colspan=1) # vp/vs curve
    ax3 = plt.subplot2grid((1,9), (0,3), colspan=3) # crossplot phi - vp
    ax4 = plt.subplot2grid((1,9), (0,6), colspan=3) # crossplot ip - vp/vs

    ax0.plot(L.VSH[ss], L.index[ss], **sty1)
    ax0.plot(L.VSH[sh], L.index[sh], **sty2)
    ax0.plot(L.VSH, L.index, **sty0)
    ax0.set_xlabel('VSH')
    ax0.locator_params(axis='x', nbins=2)

    ax1.plot(L.IP[ss], L.index[ss], **sty1)
    ax1.plot(L.IP[sh], L.index[sh], **sty2)
    ax1.plot(L.IP, L.index,  **sty0)
    ax1.set_xlabel('$I_\mathrm{P}$')
    ax1.set_xlim(6e3,14e3)                              # TODO: set automatically
    ax1.locator_params(axis='x', nbins=2)

    ax2.plot(L.VPVS[ss], L.index[ss], **sty1)
    ax2.plot(L.VPVS[sh], L.index[sh], **sty2)
    ax2.plot(L.VPVS, L.index, **sty0)
    ax2.set_xlabel('$V_\mathrm{P}/V_\mathrm{S}$')
    ax2.set_xlim(1.5,2.5)
    ax2.locator_params(axis='x', nbins=2)

    ax3.plot(L.PHIE[ss], L.VP[ss], **sty1)
    ax3.set_xlim(0,0.5),  ax3.set_ylim(2.5e3,5.5e3)
    ax3.set_xlabel('$V_\mathrm{P}$ vs $\phi_\mathrm{e}$')

    ax4.plot(L.VP*L.RHOB[ss], L.VP/L.VS[ss], **sty1)
    ax4.plot(L.VP*L.RHOB[sh], L.VP/L.VS[sh], **sty2)
    ax4.set_xlim(5e3,15e3),  ax4.set_ylim(1.5,2.5)
    ax4.set_xlabel('$V_\mathrm{P}/V_\mathrm{S}$ vs $I_\mathrm{P}$')

    for aa in [ax0,ax1,ax2]:
        aa.set_ylim(z2,z1)
#         aa.axhline(2153, color='k', ls='-')
#         aa.axhline(2183, color='k', ls='--')
    for aa in [ax0,ax1,ax2,ax3,ax4]:
        aa.tick_params(which='major', labelsize=8)
    for aa in [ax1,ax2]:
        aa.set_yticklabels([])
    plt.subplots_adjust(wspace=.8,left=0.05,right=0.95)
    
    #plt.savefig('/Users/matt/Dropbox/Agile/SEG/Tutorials/delMonte_Apr2017/Figure_1.png', dpi=300)
    plt.show()


def filt_unfilt_diff(unfilt, filt):

    print(f"LOG:\t\tUnfiltered\t:\tFiltered")
    print('='*48)
    print(f"GR min: \t{unfilt.GR.min():.2f} \t\t: \t{filt.GR.min():.2f}")
    print(f"GR max: \t{unfilt.GR.max():.2f} \t\t: \t{filt.GR.max():.2f}")
    print('-'*48)
    print(f"VSH min: \t{unfilt.VSH.min():.2f} \t\t: \t{filt.VSH.min():.2f}")
    print(f"VSH max: \t{unfilt.VSH.max():.2f} \t\t: \t{filt.VSH.max():.2f}")
    print('-'*48)
    print(f"PHIE min: \t{unfilt.PHIE.min():.2f} \t\t: \t{filt.PHIE.min():.2f}")
    print(f"PHIE max: \t{unfilt.PHIE.max():.2f} \t\t: \t{filt.PHIE.max():.2f}")
    print('-'*48)
    print(f"IP min: \t{unfilt.IP.min():.2f} \t: \t{filt.IP.min():.2f}")
    print(f"IP max: \t{unfilt.IP.max():.2f} \t: \t{filt.IP.max():.2f}")
    print('-'*48)
    print(f"VPVS min: \t{unfilt.VPVS.min():.2f} \t\t: \t{filt.VPVS.min():.2f}")
    print(f"VPVS max: \t{unfilt.VPVS.max():.2f} \t\t: \t{filt.VPVS.max():.2f}")
    print('-'*48)
    print(f"VP min: \t{unfilt.VP.min():.2f} \t: \t{filt.VP.min():.2f}")
    print(f"VP max: \t{unfilt.VP.max():.2f} \t: \t{filt.VP.max():.2f}")
    print('-'*48)
    print(f"VS min: \t{unfilt.VS.min():.2f} \t: \t{filt.VS.min():.2f}")
    print(f"VS max: \t{unfilt.VS.max():.2f} \t: \t{filt.VS.max():.2f}")
    print('-'*48)
    print(f"RHOB min: \t{unfilt.RHOB.min():.2f} \t\t: \t{filt.RHOB.min():.2f}")
    print(f"RHOB max: \t{unfilt.RHOB.max():.2f} \t\t: \t{filt.RHOB.max():.2f}")


def xvrh(f, M1, M2):
    '''
    Simple Voigt-Reuss-Hill bounds for 2-components mixture, (C) aadm 2017

    INPUT
    f: volumetric fraction of mineral 1
    M1: elastic modulus mineral 1
    M2: elastic modulus mineral 2

    OUTPUT
    M_Voigt: upper bound or Voigt average
    M_Reuss: lower bound or Reuss average
    M_VRH: Voigt-Reuss-Hill average
    '''
    M_Voigt = f*M1 + (1-f)*M2
    M_Reuss = 1/ ( f/M1 + (1-f)/M2 )
    M_VRH   = (M_Voigt+M_Reuss)/2
    return M_Voigt, M_Reuss, M_VRH


def vrh(volumes,k,mu):
    f = np.array(volumes).T
    k = np.resize(np.array(k),np.shape(f))
    mu = np.resize(np.array(mu),np.shape(f))

    k_u = np.sum(f*k, axis=1)
    k_l = 1. / np.sum(f/k, axis=1)
    mu_u = np.sum(f*mu, axis=1)
    mu_l = 1. / np.sum(f/mu, axis=1)
    k0 = (k_u+k_l) / 2.
    mu0 = (mu_u+mu_l) / 2.
    return k_u, k_l, mu_u, mu_l, k0, mu0

def vels(K_DRY, G_DRY, K0, D0, Kf, Df, phi):
    '''
    Calculates velocities and densities of saturated rock via Gassmann equation, (C) aadm 2015

    INPUT
    K_DRY,G_DRY: dry rock bulk & shear modulus in GPa
    K0, D0: mineral bulk modulus and density in GPa
    Kf, Df: fluid bulk modulus and density in GPa
    phi: porosity
    '''
    rho  = D0*(1-phi)+Df*phi
    K    = K_DRY + (1-K_DRY/K0)**2 / ( (phi/Kf) + ((1-phi)/K0) - (K_DRY/K0**2) )
    vp   = np.sqrt((K+4./3*G_DRY)/rho)*1e3
    vs   = np.sqrt(G_DRY/rho)*1e3
    return vp, vs, rho, K

def hertzmindlin(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
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
    '''
    P   /= 1e3 # converts pressure in same units as solid moduli (GPa)
    PR0  =(3*K0-2*G0)/(6*K0+2*G0) # poisson's ratio of mineral mixture
    K_HM = (P*(Cn**2*(1-phic)**2*G0**2) / (18*np.pi**2*(1-PR0)**2))**(1/3)
    G_HM = ((2+3*f-PR0*(1+3*f))/(5*(2-PR0))) * ((P*(3*Cn**2*(1-phic)**2*G0**2)/(2*np.pi**2*(1-PR0)**2)))**(1/3)
    return K_HM, G_HM

def softsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
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
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY =-4/3*G_HM + (((phi/phic)/(K_HM+4/3*G_HM)) + ((1-phi/phic)/(K0+4/3*G_HM)))**-1
    tmp   = G_HM/6*((9*K_HM+8*G_HM) / (K_HM+2*G_HM))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY

def stiffsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
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
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY  = -4/3*G0 + (((phi/phic)/(K_HM+4/3*G0)) + ((1-phi/phic)/(K0+4/3*G0)))**-1
    tmp    = G0/6*((9*K0+8*G0) / (K0+2*G0))
    G_DRY  = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY


def rpt(model='soft',vsh=0.0,fluid='gas',phic=0.4,Cn=8,P=10,f=1,display=True):
    phi=np.linspace(0.01,phic,10)
    sw=np.linspace(0,1,10)
    xx=np.empty((phi.size,sw.size))
    yy=np.empty((phi.size,sw.size))
    (K_hc, RHO_hc) = (K_g, RHO_g) if fluid == 'gas' else (K_o, RHO_o)

    _,_,K0 = vrh(vsh,K_sh,K_qz)
    _,_,MU0 = vrh(vsh,MU_sh,MU_qz)
    RHO0 = vsh*RHO_sh+(1-vsh)*RHO_qz
    if model=='soft':
        Kdry, MUdry = softsand(K0,MU0,phi,phic,Cn,P,f)
    elif model=='stiff':
        Kdry, MUdry = stiffsand(K0,MU0,phi,phic,Cn,P,f)

    for i,val in enumerate(sw):
        _,K_f,_= vrh(val,K_b,K_hc)
        RHO_f = val*RHO_b + (1-val)*RHO_hc
        vp,vs,rho,_= vels(Kdry,MUdry,K0,RHO0,K_f,RHO_f,phi)
        xx[:,i]=vp*rho
        yy[:,i]=vp/vs
    opt1={'backgroundcolor':'0.9'}
    opt2={'ha':'right','backgroundcolor':'0.9'}
    
    if display:
        plt.figure(figsize=(7,7))
        plt.plot(xx, yy, '-ok', alpha=0.3)
        plt.plot(xx.T, yy.T, '-ok', alpha=0.3)
        for i,val in enumerate(phi):
            plt.text(xx[i,-1]+150,yy[i,-1]+.02,'$\phi={:.02f}$'.format(val), **opt1)
        plt.text(xx[-1,0]-200,yy[-1,0]-0.015,'$S_\mathrm{{w}}={:.02f}$'.format(sw[0]), **opt2)
        plt.text(xx[-1,-1]-200,yy[-1,-1]-0.015,'$S_\mathrm{{w}}={:.02f}$'.format(sw[-1]), **opt2)
        plt.xlabel('IP'), plt.ylabel('VP/VS')
        plt.xlim(2e3,11e3)
        plt.ylim(1.6,2.8)

        #plt.title('RPT {} (N:G={}, fluid={})'.format(model.upper(),1-vsh, fluid))
        
        # plt.savefig('/Users/matt/Dropbox/Agile/SEG/Tutorials/delMonte_Apr2017/Figure_3_part1.png', dpi=300)
        plt.show()
    return xx,yy


def twolayer(vp0,vs0,rho0,vp1,vs1,rho1):
    from bruges.reflection import shuey2      # TODO:  install bruges.reflection
    from bruges.filters import ricker         # TODO:  install bruges.filters

    n_samples = 500
    interface=int(n_samples/2)
    ang=np.arange(31)
    wavelet=ricker(.25, 0.001, 25)
    # !!! MODIFICATION !!!
    # numpy.convolve(a, v, mode='full')
    # v(M,) array_like, Second one-dimensional input array.
    # wavelet return from ricker is >1 dimensional
    wavelet = wavelet[0]

    model_ip,model_vpvs,rc0,rc1 = ( np.zeros(n_samples) for _ in range(4) )
    model_z = np.arange(n_samples)
    model_ip[:interface]=vp0*rho0
    model_ip[interface:]=vp1*rho1
    model_vpvs[:interface]=np.true_divide(vp0,vs0)
    model_vpvs[interface:]=np.true_divide(vp1,vs1)

    avo=shuey2(vp0,vs0,rho0,vp1,vs1,rho1,ang)
    rc0[interface]=avo[0]
    rc1[interface]=avo[-1]
    synt0=np.convolve(rc0,wavelet,mode='same')
    synt1=np.convolve(rc1,wavelet,mode='same')
    clip=np.max(np.abs([synt0, synt1]))
    clip += clip*.2
    
    opz0={'color':'b', 'linewidth':4}
    opz1={'color':'k', 'linewidth':2}
    opz2={'linewidth':0, 'alpha':0.5}

    f = plt.subplots(figsize=(10, 4))
    ax0 = plt.subplot2grid((1,16), (0,0), colspan=2) # ip
    ax1 = plt.subplot2grid((1,16), (0,2), colspan=2) # vp/vs
    ax2 = plt.subplot2grid((1,16), (0,4), colspan=2) # synthetic @ 0 deg
    ax3 = plt.subplot2grid((1,16), (0,6), colspan=2) # synthetic @ 30 deg
    ax4 = plt.subplot2grid((1,16), (0,9), colspan=7) # avo curve

    ax0.plot(model_ip, model_z, **opz0)
    ax0.set_xlabel('IP')
    ax0.locator_params(axis='x', nbins=2)

    ax1.plot(model_vpvs, model_z, **opz0)
    ax1.set_xlabel('VP/VS')
    ax1.locator_params(axis='x', nbins=2)

    ax2.plot(synt0, model_z, **opz1)
    ax2.fill_betweenx(model_z, 0, synt0, where=synt0>0, facecolor='black', **opz2)
    ax2.set_xlim(-clip,clip)
    ax2.set_xlabel('angle={:.0f}'.format(ang[0]))
    ax2.locator_params(axis='x', nbins=2)

    ax3.plot(synt1, model_z, **opz1)
    ax3.fill_betweenx(model_z, 0, synt1, where=synt1>0, facecolor='black', **opz2)
    ax3.set_xlim(-clip,clip)
    ax3.set_xlabel('angle={:.0f}'.format(ang[-1]))
    ax3.locator_params(axis='x', nbins=2)

    ax4.plot(ang, avo, **opz0)
    ax4.axhline(0, color='k', lw=2)
    ax4.set_xlabel('angle of incidence')
    ax4.tick_params(which='major', labelsize=8)

    for aa in [ax0,ax1,ax2,ax3]:
        aa.set_ylim(350,150)
        aa.tick_params(which='major', labelsize=8)
        aa.set_yticklabels([])

    plt.subplots_adjust(wspace=.8,left=0.05,right=0.95)


def critpor(K0, G0, phi, phic=0.4):
    '''
    Critical porosity, Nur et al. (1991, 1995)
    written by aadm (2015) from Rock Physics Handbook, p.353

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    '''
    K_DRY  = K0 * (1-phi/phic)
    G_DRY  = G0 * (1-phi/phic)
    return K_DRY, G_DRY

def contactcement(K0, G0, phi, phic=0.4, Cn=8.6, Kc=37, Gc=45, scheme=2):
    '''
    Contact cement (cemented sand) model, Dvorkin-Nur (1996)
    written by aadm (2015) from Rock Physics Handbook, p.255

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    Kc, Gc: cement bulk & shear modulus in GPa
            (default 37, 45 i.e. quartz)
    scheme: 1=cement deposited at grain contacts
            2=uniform layer around grains (default)
    '''
    PR0=(3*K0-2*G0)/(6*K0+2*G0)
    PRc = (3*Kc-2*Gc)/(6*Kc+2*Gc)
    if scheme == 1: # scheme 1: cement deposited at grain contacts
        alpha = ((phic-phi)/(3*Cn*(1-phic))) ** (1/4)
    else: # scheme 2: cement evenly deposited on grain surface
        alpha = ((2*(phic-phi))/(3*(1-phic)))**(1/2)
    LambdaN = (2*Gc*(1-PR0)*(1-PRc)) / (np.pi*G0*(1-2*PRc))
    N1 = -0.024153*LambdaN**-1.3646
    N2 = 0.20405*LambdaN**-0.89008
    N3 = 0.00024649*LambdaN**-1.9864
    Sn = N1*alpha**2 + N2*alpha + N3
    LambdaT = Gc/(np.pi*G0)
    T1 = -10**-2*(2.26*PR0**2+2.07*PR0+2.3)*LambdaT**(0.079*PR0**2+0.1754*PR0-1.342)
    T2 = (0.0573*PR0**2+0.0937*PR0+0.202)*LambdaT**(0.0274*PR0**2+0.0529*PR0-0.8765)
    T3 = 10**-4*(9.654*PR0**2+4.945*PR0+3.1)*LambdaT**(0.01867*PR0**2+0.4011*PR0-1.8186)
    St = T1*alpha**2 + T2*alpha + T3
    K_DRY = 1/6*Cn*(1-phic)*(Kc+(4/3)*Gc)*Sn
    G_DRY = 3/5*K_DRY+3/20*Cn*(1-phic)*Gc*St
    return K_DRY, G_DRY


# Fluid Replacement Modelling

def xfrm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    """
    INPUT
    vp1, vs1, rho1: (vector) Measured Vp, Vs and density saturated with fluid 1
    rho_f1, k_f1:   (vector) Density and bulk modulus of fluid 1 (requires Sw)
    rho_f2, k_f2:   (scalar) Density and bulk modulus of fluid 2 (rho_o, k_o, rho_g, k_g, etc)
    k0:             (scalar) mineral bulk modulus - (k_u - k_l)/2
    phi:            (vector) porosity
    
    RETURN 
    vp2, vs2, rho2, k_s2: Vp, Vs, density and bulk modulus of rock with fluid 2. 
    
    Velocities are in m/s and densities in g/cm3.
    
    USAGE:
    vp1ox, vs1ox, rho1ox, k1ox = frm(vp1, vs1, rho1, RHO_WATER, K_WATER, RHO_GAS, K_GAS, K0, phiex1)
    """
    # convert Vp,Vs from m/s to km/s for calculation
    vp1  = vp1 / 1000.
    vs1  = vs1 / 1000.
    
    mu1  = rho1 * vs1**2.
    k_s1 = rho1 * vp1**2 - (4./3.)*mu1  # mu1 = rho1 * vs1**2
    
    # The dry rock bulk modulus
    kdry = (k_s1 * ((phi*k0)/k_f1 + 1 - phi) - k0) / ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)
    
    # Now we can apply Gassmann to get the new values
    k_s2 = kdry + (1- (kdry/k0))**2 / ( (phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2) )
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2  = mu1
    vp2  = np.sqrt(((k_s2 + (4./3)*mu2))/rho2)
    vs2  = np.sqrt((mu2/rho2))

    # return Vp,Vs as m/s
    return vp2*1000, vs2*1000, rho2, k_s2


def frm(vp_1, vs_1, rho_1, rho_f1, k_f1, rho_f2, k_f2, K0, phi_):
    """
    INPUT
    vp1, vs1, rho1: (vector) Measured Vp, Vs and density saturated with fluid 1
    rho_f1, k_f1:   (vector) Density and bulk modulus of fluid 1 (requires Sw)
    rho_f2, k_f2:   (scalar) Density and bulk modulus of fluid 2 (rho_o, k_o, rho_g, k_g, etc)
    k0:             (scalar) mineral bulk modulus - (k_u - k_l)/2
    phi:            (vector) porosity

    RETURN
    vp2, vs2, rho2, k_s2: Vp, Vs, density and bulk modulus of rock with fluid 2.

    Velocities are in m/s and densities in g/cm3.

    USAGE:
    vp1ox, vs1ox, rho1ox, k1ox = frm(vp1, vs1, rho1, RHO_WATER, K_WATER, RHO_GAS, K_GAS, K0, phiex1)
    """

    vp_frm = []
    vs_frm = []
    rho_frm = []

    for i in range(0, len(vp_1)):

        # convert Vp,Vs from m/s to km/s for calculation
        vp1 = vp_1[i] / 1000.
        vs1 = vs_1[i] / 1000.
        rho1 = rho_1[i]
        phi = phi_[i]
        k0 = K0[i]

        mu1 = rho1 * vs1 ** 2.
        rho2 = rho1 - phi * rho_f1 + phi * rho_f2
        rho_frm.append(rho2)
        mu2 = mu1
        vs2 = np.sqrt((mu2 / rho2))
        vs_frm.append(vs2*1000)

        k_s1 = rho1 * vp1 ** 2 - (4. / 3.) * mu1  # mu1 = rho1 * vs1**2    

        # The dry rock bulk modulus
        kdry = (k_s1 * ((phi*k0)/k_f1+1-phi)-k0) / ((phi*k0)/k_f1+(k_s1/k0)-1-phi)

        # kdry values were negative when k0=36 was too high -> use vrh()
        # Vp2 values were getting > 6000 when kdry was > 70 (why so high?)
        if (0 < kdry < 36):

            # Now we can apply Gassmann to get the new values
            k_s2 = kdry + (1 - (kdry / k0)) ** 2 / ((phi / k_f2) + ((1 - phi) / k0) - (kdry / k0 ** 2))
            
            if k_s2 > 0:
                
            # print(f"k_s2: {k_s2}")
                vp2 = np.sqrt(((k_s2 + (4. / 3.) * mu2)) / rho2)
                # vp_frm.append(np.round(vp2*1000, 0))
                vp_frm.append(int(vp2*1000))
            else: 
                vp2 = vp1
                vp_frm.append(vp2*1000)

        else:
            vp2 = vp1
            vp_frm.append(vp2*1000)
            
    # return as arrays
    vp_frm = np.array(vp_frm)
    vs_frm = np.array(vs_frm)
    rho_frm = np.array(rho_frm)
    
    return vp_frm, vs_frm, rho_frm, k_s2



def r0g(vp1, vs1, rho1, vp2, vs2, rho2):
    
    R0 = []
    G = []
    R_theta = []

    for i in range(len(vp1)):

        delta_rho = rho2[i] - rho1[i]
        delta_vp = vp2[i] - vp1[i]
        delta_vs = vs2[i] - vs1[i]

        vp = (vp2[i] + vp1[i]) / 2
        vs = (vs2[i] + vs1[i]) / 2
        rho = (rho2[i] + rho1[i]) / 2

        R0_temp = 1/2*((delta_vp / vp) + (delta_rho / rho))
        R0.append(R0_temp)

        G_temp = 1/2 * (delta_vp / vp) - 2 * vs**2/vp**2 * ((delta_rho / rho) + (2*delta_vs / vs))
        G.append(G_temp)

        R_theta_i = []
        for theta in range(41):
            R_theta_i.append(R0_temp + G_temp*(np.sin(math.radians(theta))**2))
        R_theta.append([R_theta_i])        

    median = []

    for j in range(41):
        angle = []
        for k in range(len(vp2)):
            angle.append(R_theta[k][0][j])
        median.append(statistics.median(angle))
        
    return R0, G, R_theta, median