import re
from pathlib import Path
import numpy as np
from scipy.special import ellipk
from scipy import signal
from .core.units import const


def dtft(xn: np.ndarray, frequency: np.ndarray, fs: float, downsample: bool = False) -> np.ndarray:
    """
    Compute the DTFT of the discrete time signal x[n] over a frequency range (cycles per second),

    Parameters
    ----------
    xn : np.ndarray
        time domain samples
    frequency : np.ndarray
        frequency points to evaluate the DTFT at
    fs : float 
        sampling frequency of xn
    downsample : bool, default: True
        allows downsampling prior to computing the DTFT
    
    Returns
    -------
    np.ndarray
        discrete time fourier transform of xn
    """

    # resampling frequency
    frs = np.max(frequency) * 20
    # number of samples in xn between each sample of frs
    downsample_factor = int(fs / frs)
    # determine whether downsample can be used. The sampling rate must be at least twice the downsampled rate
    downsample_apply = downsample and (len(xn) / downsample_factor) > 100 and fs > (frs * 2)

    # In most cases 1 / dt is much greater than the highest frequency of interest and the DTFT can be made much faster
    # by down-sampling. 
    if downsample_apply:
        # create lowpass filter that removes all frequency content above frs/2
        sos = signal.butter(20, frs/2.5, btype="lowpass", output="sos", fs = fs)
        # apply filter, then downsample the signal. The downsampling will leave aliases, but at much higher frequencies
        # than the DTFT will be evaluated.
        xn = signal.sosfiltfilt(sos, xn)[::downsample_factor]
        fs = frs

    # convert the continuous time frequency into a discrete frequency range. The discrete frequencies
    # are bounded by -0.5 to 0.5 if there is no aliasing.
    # To convert the continuous time frequency (cycles / sec), into the discrete frequency (cycles / sample). 
    # divide it by the sampling rate fs (samples / sec):
    # (cycles / sec) / (samples / sec) = (cycles / sample)
    fn = frequency / fs
    omega = 2 * np.pi * fn
    
    n = np.arange(len(xn))
    omega_mesh, n_mesh = np.meshgrid(omega, n)

    # broadcast input sequence across all omega
    # x_b = np.broadcast_to(xn[..., None], (len(n), len(omega)))
    # sum across all non-zero n
    Xw = np.sum(xn[..., None] * np.exp(-1j * omega_mesh * n_mesh), axis=0)

    # scale the DTFT so it's identical to the non-downsampled version
    if downsample_apply:
        Xw *= downsample_factor

    return Xw

def n_ports_from_snp(path: str) -> int:
    """
    Get the number of ports from a .snp file extension.
    """
    return int(re.match(r".[sS](\d+)[pP]", Path(path).suffix).group(1))


def ms_z_to_width(z: float, er: float, d: float) -> float:
    """
    Calculate approximate microstrip width for a given characteristic impedance.
    See equation on page 149 of [1].

    Parameters
    ----------
    z: float
        target characteristic impedance of microstrip line
    er: float
        relative permittivity of substrate
    h: float
        height of substrate
    
    Returns
    -------
    float
        line width in the same units as h.
        

    References
    ----------
    [1] Pozar, David M. Microwave Engineering. 4th ed., Wiley, 2011.
    """
    A = (z / 60) * np.sqrt((er+1)/2) + ((er-1)/(er+1)) *(0.23 + (0.11/er))
    B = (377 * np.pi) /( 2 * z * np.sqrt(er))

    # w/d > 2
    wd2 = (2 / np.pi) * ( B -1 - np.log(2*B -1) + (er-1)/(2*er) *( np.log(B-1) + 0.39 - (0.61 / er)))
    # w/d < 2
    wd12 = (8 * np.exp(A)) /( np.exp(2* A) -2)

    if wd2 > 2:
        return wd2 * d
    else:
        return wd12 * d


def coupled_sline_impedance(w: float, s: float, b: float, er: float):
    """
    Odd and even mode impedance for coupled stripline. Units of w, s and b are arbitrary.

    Assumes that thickness is zero.
    """
    # reference odd, even mode impedances.
    # page 174 in Matthaei, even and odd mode impedances of coupled strip line
    k_e = np.tanh((np.pi / 2) * (w / b)) * np.tanh((np.pi / 2) * (w + s) / b)
    kp_e = np.sqrt(1 - (k_e **2))

    k_o = np.tanh((np.pi / 2) * (w / b)) * (1 / np.tanh((np.pi / 2) * (w + s) / b))
    kp_o = np.sqrt(1 - (k_o **2))

    Z0_e = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_e) / ellipk(k_e))
    Z0_o = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_o) / ellipk(k_o))

    # mutual capacitance
    # Z0_e = 1 / vp*Ce
    # Z0_o = 1 / vp*Co
    # Co = Ca + 2 Cab

    return Z0_o, Z0_e



def coupled_sline_fringing_cap(w: float, s: float, b: float, er: float):
    """
    Odd and even mode fringing capacitance for edge coupled stripline, per unit length [m].
    See equations 5.05-24 and 5.05-25 (page 201) and Figure 5.05-13. Units of w, s, and b are arbitrary.

    Fringing capacitance is a weak function of the trace width. Assumes that both traces are the same width,
    but very little difference is seen in asymmetrical lines.

    Assumes that thickness is zero.
    """
    Z0_o, Z0_e = coupled_sline_impedance(w, s, b, er)

    # even ad odd mode capacitances, normalized by epsilon
    # Z0_e = 1 / vp*Ce, Table 5.05-1 (2) if Ca=Cb
    # Z0_o = 1 / vp*Co
    # Co = Ca + 2 Cab
    # Ce = Ca = Cb
    vp = const.c0 / np.sqrt(er)
    Ce = 1 / (Z0_e * vp * const.e0 * er)
    Co = 1 / (Z0_o * vp * const.e0 * er)

    # parallel plate capacitance, normalized by epsilon
    Cp = 2 * w / (b)
    # fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0.
    # or Balanis Advanced Engineering Electromagnetics, 2nd edition, equation 8-201
    Cf = 0.44
    # solve for even and odd fringing capacitances, equation 5.05-24 and 5.05-25
    Cf_e = (Ce / 2) - Cp - Cf
    Cf_o = (Co / 2) - Cp - Cf

    # return unnormalized capacitance in farads
    eps = er * const.e0
    return Cf_o * eps, Cf_e * eps


def ustrip_impedance(w: float, h: float, er: float):
    """
    Microstrip impedance and effective permittivity for thin microstrip lines
    """

    if (w / h) < 1:
        er_eff = ((er + 1) / 2) + ((er - 1) / 2) * ((1 + 12 * (h / w))**(-0.5) + 0.04 * (1 - (w / h)**2))
        zc = (const.eta0 / (2 * np.pi * np.sqrt(er_eff))) * np.log((8 * h / w) + 0.25 * (w / h))
    else:
        er_eff = ((er + 1) / 2) + ((er - 1) / 2) * ((1 + 12 * (h / w))**(-0.5))
        zc = (const.eta0 / (np.sqrt(er_eff))) * ((w / h) + 1.393  + 0.677 * np.log((w / h) + 1.444))**(-1)

    return zc, er_eff


def ustrip_fringing_cap(w, h, er):
    """
    Fringing capacitance from edge of uncoupled microstrip line, per unit length [m].
    This is a weak function of the width of the line.
    """
    # characteristic impedance and effective epsilon
    zc, er_eff  = ustrip_impedance(w, h, er)

    # parallel plate capacitance
    Cp = er * const.e0 * (w / h)
    # uncoupled fringe capacitance
    Cf = ((np.sqrt(er_eff) / (const.c0 * zc)) - Cp) / 2

    # fringing capacitance seems to be a bit high, reduced empirically 
    return Cf * 0.9


def coupled_ustrip_fringing_cap(w: float, s: float, h: float, er: float):
    """
    Odd and even mode fringing capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver

    Assumes that thickness is zero.
    """
    # uncoupled fringing capacitance
    Cf = ustrip_fringing_cap(w, h, er)

    # fringing capacitance in even mode
    A = np.exp(-0.1 * np.exp(2.33 - 2.53 * (w / h)))
    Cf_e = Cf / (1 + A * (h / s) * np.tanh(8 * s / h))

    # odd mode fringing capacitance through the dielectric
    coth = lambda x : np.cosh(x) / np.sinh(x)
    C_gd = (const.e0 * er / np.pi) * np.log(coth((np.pi / 4) * (s / h))) + 0.65 * Cf * (
        ((0.02 * np.sqrt(er)) / (s / h)) + 1 - (1 / (er **2 ))
    )
    # odd mode fringing capacitance through the air
    k = (s / h) / ((s / h) + 2 * (w / h))
    kp = np.sqrt(1 - (k**2))
    C_ga = const.e0 * ellipk(kp) / ellipk(k)

    # return unnormalized capacitance in farads, odd, even
    return (C_gd + C_ga), Cf_e


def coupled_ustrip_cap(w: float, s: float, h: float, er: float):
    """ Total odd and even mode capacitance per unit length """
    Cf = ustrip_fringing_cap(w, s, er)

    Cfo, Cfe = coupled_ustrip_fringing_cap(w, s, h, er)

    Cp = er * const.e0 * w / h

    # total odd and even mode capacitances
    Co = Cp + Cf + Cfo
    Ce = Cp + Cf + Cfe

    return Co, Ce


def coupled_ustrip_impedance(w: float, s: float, h: float, er: float):
    """
    Odd and even mode total capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver
    """

    # total even and odd mode coupled capacitance
    Co, Ce = coupled_ustrip_cap(w, s, h, er)
    # coupled capacitance in air
    Co_a, Ce_a = coupled_ustrip_cap(w, s, h, 1)

    Zo = 1 / (const.c0 * np.sqrt(Co * Co_a))
    Ze = 1 / (const.c0 * np.sqrt(Ce * Ce_a))

    return Zo, Ze


def lp_filter_prototype(n: int, ripple: float = 0.5):
    """
    Normalized element values for equal ripple low-pass filter prototypes. See Table 8.4 in [1].
    
    Parameters
    ----------
    n : int
        filter order (1-10). Odd n will be matched to 50 ohms, while even n requires an impedance match.
    ripple : float, default: 0.5
        pass-band ripple. Supported values are 0.5 and 3.0 dB.

    Returns
    -------
    list
        list of n+1 element values. The last value is the normalized resistive load.

    References
    ----------
    [1] Pozar, David M. Microwave Engineering. 4th ed., Wiley, 2011.

    """
    lp_table = {
        "0.0" : {
            1: [2.0000, 1.0000],
            2: [1.4142, 1.4142, 1.0000],
            3: [1.0000, 2.0000, 1.0000, 1.0000],
            4: [0.7654, 1.8478, 1.8478, 0.7654, 1.0000],
            5: [0.6180, 1.6180, 2.0000, 1.6180, 0.6180, 1.0000],
            6: [0.5176, 1.4142, 1.9318, 1.9318, 1.4142, 0.5176, 1.0000],
            7: [0.4450, 1.2470, 1.8019, 2.0000, 1.8019, 1.2470, 0.4450, 1.0000],
            8: [0.3902, 1.1111, 1.6629, 1.9615, 1.9615, 1.6629, 1.1111, 0.3902, 1.0000],
            9: [0.3473, 1.0000, 1.5321, 1.8794, 2.0000, 1.8794, 1.5321, 1.0000, 0.3473, 1.0000],
            10: [0.3129, 0.9080, 1.4142, 1.7820, 1.9754, 1.9754, 1.7820, 1.4142, 0.9080, 0.3129, 1.0000]
        },
        # 0.5dB Ripple.
        "0.5" : {
            1: [0.6986, 1.0000],
            2: [1.4029, 0.7071, 1.9841],
            3: [1.5963, 1.0967, 1.5963, 1.0000],
            4: [1.6703, 1.1926, 2.3661, 0.8419, 1.9841],
            5: [1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000],
            6: [1.7254, 1.2479, 2.6064, 1.3137, 2.4758, 0.8696, 1.9841],
            7: [1.7372, 1.2583, 2.6381, 1.3444, 2.6381, 1.2583, 1.7372, 1.0000],
            8: [1.7451, 1.2647, 2.6564, 1.3590, 2.6964, 1.3389, 2.5093, 0.8796, 1.9841],
            9: [1.7504, 1.2690, 2.6678, 1.3673, 2.7239, 1.3673, 2.6678, 1.2690, 1.7504, 1.0000],
            10: [1.7543, 1.2721, 2.6754, 1.3725, 2.7392, 1.3806, 2.7231, 1.3485, 2.5239, 0.8842, 1.9841]
        },
        # 3.0dB Ripple.
        "3.0" : {
            1: [1.9953, 1.0000],
            2: [3.1013, 0.5339, 5.8095],
            3: [3.3487, 0.7117, 3.3487, 1.0000],
            4: [3.4389, 0.7483, 4.3471, 0.5920, 5.8095],
            5: [3.4817, 0.7618, 4.5381, 0.7618, 3.4817, 1.0000],
            6: [3.5045, 0.7685, 4.6061, 0.7929, 4.4641, 0.6033, 5.8095],
            7: [3.5182, 0.7723, 4.6386, 0.8039, 4.6386, 0.7723, 3.518, 1.0000],
            8: [3.5277, 0.7745, 4.6575, 0.8089, 4.6990, 0.8018, 4.499, 0.6073, 5.8095],
            9: [3.5340, 0.7760, 4.6692, 0.8118, 4.7272, 0.8118, 4.669, 0.7760, 3.5340, 1.0000],
            10: [3.5384, 0.7771, 4.6768, 0.8136, 4.7425, 0.816, 4.726, 0.8051, 4.5142, 0.6091, 5.8095]
        }
    }

    return lp_table[f"{ripple:.1f}"][n]



def combline_sections_nb(g: list, f1: float, f2: float, er: float, h: float, wp: float = 1):
    """
    Normalized capacitances for edge coupled combline narrowband filter.

    Table 10.06-1, pg 617, Matthaei
    """
    eta0 = const.eta0

    n = len(g) - 2
    f0 = (f1 + f2) / 2
    w = (f2 - f1) / f0

    theta_1 = (np.pi / 2) * (1 - (w / 2))
    Y_a = (1 / 50)

    J0_Y = 1 / np.sqrt(g[0] * g[1] * wp)
    Jk_Y = [1 / (wp * np.sqrt(g[k] * g[k+1])) for k in range(1, n)]
    Jn_y = 1 / np.sqrt(g[n] * g[n+1] * wp)

    Jk_Y = [J0_Y] + Jk_Y + [Jn_y]

    Nk = [0] + [np.sqrt(Jk_Y[k]**2 + ((np.tan(theta_1)**2) / 4)) for k in range(1, n)]

    M1 = Y_a * (J0_Y * np.sqrt(h) + 1)
    Mn = Y_a * (Jn_y * np.sqrt(h) + 1)

    # self capacitances, normalized by epsilon
    C0 = (eta0 / np.sqrt(er)) * (2 * Y_a - M1)
    C1 = (eta0 / np.sqrt(er)) * (Y_a - M1 + h * Y_a * ((np.tan(theta_1) / 2) + (Jk_Y[0]) **2 + Nk[1] - (Jk_Y[1])))
    Ck = [(eta0 / np.sqrt(er)) * h * Y_a * (Nk[k-1] + Nk[k] - Jk_Y[k-1] - Jk_Y[k]) for k in range(2, n)]
    Cn = (eta0 / np.sqrt(er)) * (Y_a - Mn + h * Y_a * ((np.tan(theta_1) / 2) + (Jk_Y[-1])**2 + Nk[-1] - Jk_Y[-2]))
    Cn1 = (eta0 / np.sqrt(er)) * (2 * Y_a - Mn)

    Ck = np.array([C0] + [C1] + Ck + [Cn] +[Cn1])

    # mutual capacitance, normalized by epsilon
    Cm0 = (eta0 / np.sqrt(er)) * (M1 - Y_a)
    Cmk = [(eta0 / np.sqrt(er)) * (Y_a * h) * (Jk_Y[k]) for k in range(1, n)]
    CmN = (eta0 / np.sqrt(er)) * (Mn - Y_a)

    Cmk = np.array([Cm0] + Cmk + [CmN])

    return Ck, Cmk

def combline_sections_wb(g: list, f1: float, f2: float, er: float, h: float, wp: float = 1):
    """
    Normalized capacitances for edge coupled combline wideband filter.

    Table 10.07-1, pg 628, Matthaei
    """
    eta0 = const.eta0

    n = len(g) - 2
    f0 = (f1 + f2) / 2
    w = (f2 - f1) / f0

    theta_1 = (np.pi / 2) * (1 - (w / 2))
    Y_a = (1 / 50)

    # Table 10.07-1, pg 628
    Jk2_Y = [g[2] / ( g[0] * np.sqrt(g[k] * g[k+1]) ) for k in range(2, n-2)]
    Jn_Y = (1 / g[0]) * np.sqrt((g[2] * g[0]) / (g[n-2] * g[n+1]))
    Jk_Y = [0, 0] + Jk2_Y + [Jn_Y]

    Nk = [0, 0] + [np.sqrt((Jk_Y[k])**2 + ((wp * g[2] * np.tan(theta_1)) / (2 * g[0]))**2) for k in range(2, n-1)]

    Zn_Za = [(wp * g[k] * g[k+1] * np.tan(theta_1)) for k in range(n+1)]

    Y2_Ya = ((wp * g[2]) / (2 * g[0])) * np.tan(theta_1) + Nk[2] - (Jk_Y[2])
    Yk3_Ya = [Nk[k-1] + Nk[k] - Jk_Y[k-1] - Jk_Y[k] for k in range(3, n-1)]
    Yn_Ya = ((wp * ( 2 * g[0] * g[n-1] - g[2] * g[n+1]) * np.tan(theta_1)) / (2 * g[0] * g[n+1])) + Nk[n-2] - Jk_Y[n-2]
    Yk_Ya = [0, 0] + [Y2_Ya] + Yk3_Ya + [Yn_Ya]

    # self capacitance, normalized by epsilon
    C1 = (eta0 / np.sqrt(er)) * Y_a * (1 - np.sqrt(h)) / (Zn_Za[0])
    C2 = (eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[2]) - np.sqrt(h) * (C1)

    Ck3 = [(eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[k]) for k in range(3, n-1)]

    CN = (eta0 / np.sqrt(er)) * Y_a * (1 - np.sqrt(h)) / (Zn_Za[-1])
    CN_1 = (eta0 / np.sqrt(er)) * (Y_a * h) * (Yk_Ya[n-1]) - np.sqrt(h) * (CN)

    Ck = np.array([C1] + [C2] + Ck3 + [CN_1] + [CN])

    # mutual capacitance, normalized by epsilon
    Cm12 = (eta0 / np.sqrt(er)) * Y_a * (np.sqrt(h) / (Zn_Za[0]))
    Cmk2 = [(eta0 / np.sqrt(er)) * (Y_a * h) * (Jk_Y[k]) for k in range(2, n-1)]
    CmN = (eta0 / np.sqrt(er)) * Y_a * (np.sqrt(h) / (Zn_Za[-1]))

    Cmk = np.array([Cm12] + Cmk2 + [CmN])

    return Ck, Cmk


def round_to_multiple(value, multiple: float = 1, precision: int = 6):
    """ 
    Rounds value to nearest multiple. Multiple can be greater or less than 1.
        
    """
    return np.around(np.around(value / multiple) * multiple, precision)

