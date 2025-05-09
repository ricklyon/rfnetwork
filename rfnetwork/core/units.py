import numpy as np
        
class const():
    """
    Physical Constants.
    
    eta0    : Wave impedance of free space
    c0      : Speed of light [m/s]
    c0_in   : Speed of light [in/s]
    e0      : Permittivity of free space [F/m]
    u0      : Permeability of free space [H/m]
    k       : Boltzmann's constant [J/K]
    r_e     : Radius of Earth [m]
    cu_l    : Thermal Conductivity of copper (W/m K)
    cu_sigma: Electrical conductivity of copper (S/m)
    eta0    : Plane wave impedance in free space
    t0      : room temperature [Kelvin]
    """
    eta0 = 376.73
    c0 = 299792458
    c0_in = 11802859050.705801
    e0 = 8.854187817e-12
    k =  1.380649e-23
    u0 = 4 * np.pi * 1e-7
    r_e = 6371
    cu_sigma = 5.76e7
    cu_l = 401
    eta0 = np.sqrt(u0/e0)
    t0 = 290


class conv():
    """ 
    Unit conversion functions. 
    Follows the same convention as matlab's units_ratio: to_from ()
    i.e. To convert from feet to meters: <meters> = conv.m_ft(<feet>)

    m       : meter
    km      : kilometer
    ft      : feet
    in      : inch
    nmi     : nautical mile
    mi      : mile

    v       : Volts
    w       : Watts
    z       : Impedance (complex)
    dbm     : dB (10log10) ref to 1mW
    db10    : dB (10log10)
    db20    : dB (20log10)
    lin     : ratio in linear scale
    vswr    : voltage standing wave ratio
    gamma   : reflection coeff

    aeff    : antenna effective aperture (linear)
    dir     : antenna maximum directivity (linear)
    e       : plane wave peak electric field (V/m)
    wm2     : average power density (W/m^2)
    
    c       : Celsius
    f       : Fahrenheit
    k       : Kelvin
    """
    # voltage, power
    v_w   = lambda W, R=50 : np.sqrt(2*R*W)
    w_v   = lambda V, R=50 : V**2 / R

    w_dbm = lambda dbm : (10**(dbm/10))*1e-3
    dbm_w = lambda W   : 10*np.log10(W/1e-3)

    v_dbm = lambda dbm, R=50 : np.sqrt(2*R*((10**(dbm/10))*1e-3))
    dbm_v = lambda v,   R=50 : 10*np.log10(((np.abs(v)**2)/(2*R))/1e-3)

    # ratios
    db20_lin  = lambda x   : 20*np.log10(np.abs(x))
    db10_lin  = lambda x   : 10*np.log10(np.abs(x))

    lin_db10  = lambda x   : 10**((np.abs(x) * np.sign(x))/10)
    lin_db20  = lambda x   : 10**((np.abs(x) * np.sign(x))/20)

    # impedance
    db_vswr = lambda vswr : 20*np.log10((vswr - 1) / (vswr + 1))
    vswr_db = lambda db: (1 + np.abs(10**(db/20))) / (1 - np.abs(10**(db/20)))

    gamma_vswr = lambda vswr: (vswr - 1) / (vswr + 1)
    vswr_gamma = lambda gamma: (1 + np.abs(gamma)) / (1 - np.abs(gamma))

    gamma_z = lambda z, refz=50: (z-refz)/(z+refz)
    z_gamma = lambda gamma, refz=50: refz*((1+gamma)/(1-gamma))

    # temperature
    c_f   = lambda f: (f - 32) * (5/9)
    f_c   = lambda c: (c * (9/5)) + 32
    k_c   = lambda c: c + 273.15
    c_k   = lambda k: k - 273.15

    # fields
    aeff_dir = lambda lmbda, d0=1: ((lmbda**2)*d0)/(4*np.pi)
    dir_aeff = lambda lmbda, aeff: ((4*np.pi)*aeff)/(lmbda**2)

    wm2_e = lambda efield : (1/(2*const.eta0)) * (np.abs(efield)**2)
    e_wm2 = lambda wm2 : np.sqrt(2*wm2*const.eta0)

    # simple ratios
    ratios =   dict(
        ft_m   = 3.28084,
        ft_km  = 3280.84,
        ft_nmi = 6076.12,
        ft_mi = 5280,
        m_mi = 1609.34,
        mi_nmi = 1.15078,
        km_nmi = 1.85200,
        km_mi  = 1.60934,
        in_m   = 39.3701,
        mil_mm = 39.3701,
        mm_in = 25.4
    )

# add conversion functions for simple ratios
for k,v in conv.ratios.items():
    ksp = k.split('_')
    ## use nested lambda to enforce scope
    setattr(conv, k, (lambda x : lambda v : x*v)(v))
    setattr(conv, ksp[1]+'_'+ksp[0], (lambda x : lambda v : x*v)(1/v))
    