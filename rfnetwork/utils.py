import re
from pathlib import Path
import numpy as np
from scipy.special import ellipk
from scipy import signal
from np_struct import ldarray
from .core.units import const, conv


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

    # In most cases 1 / dt is much greater than the highest frequency of interest and the DTFT can be made faster
    # by down-sampling. 
    if downsample_apply:
        # create lowpass filter that removes all frequency content above frs/2
        sos = signal.butter(20, frs / 2.5, btype="lowpass", output="sos", fs = fs)
        # apply filter, then downsample the signal. 
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


def ifft(Xw: ldarray):
    """
    Computes the inverse fft on complex-valued data over a positive frequency range. Data should extend down to near 
    DC for accurate results.

    Parameters:
    -----------
    Xw : np.ndarray
        frequency domain data defined over the positive frequency range. 
        Data should extend down in frequency to the step size of frequency. For example, if the frequency 
        step is 10MHz, the lowest data point should be at 10MHz.

    Returns
    -------
    time : np.ndarray
        time vector in seconds
    xt : np.ndarray
        time domain data, with time as the first dimension followed by the remaining dimensions of Xw
    """

    data = Xw.transpose(("frequency", ...))
    frequency = Xw.coords["frequency"]
    
    # frequency step size
    f0, f1 = frequency[:2]
    fstep = f1 - f0

    # count out many data points are missing between 0Hz and the first data point (assumes data does
    # not contain a 0hz term)
    missing_fnum = int((f0 - fstep) / fstep) + 1

    missing_data = np.zeros(missing_fnum, dtype=np.complex128)
    missing_f = np.arange(0, f0, fstep)

    # create zero frequency term by extrapolating lowest value
    b_dims = [None] * len(data.shape[1:])
    missing_data = np.broadcast_to(missing_data[:, *b_dims], (missing_fnum,) + data.shape[1:])
    data_full = np.concatenate((missing_data, data), axis=0)

    # full frequency array
    data_f = np.concatenate((missing_f, frequency))

    # mirror the data around the y-axis to generate negative frequency terms for full spectrum
    # do not flip 0 freq term
    data_flip = np.flip(np.conjugate(data_full[1:]), axis=0)

    # numpy ifft requires the values sorted in frequency as:
    # [0, ..., np.pi, -np.pi, ...]
    ifft_spectrum = np.concatenate((data_full, data_flip), axis=0)

    # number of samples
    n = len(ifft_spectrum)

    # the frequency points are f = k * (fs /N). fs is samples per second, dividing this by N (samples), gives units s^-1
    # where N is the number of samples and k is 0, 1... N
    # this means the sample rate is fs = f_1 * N
    fs = f0 * n
    ds = 1 / fs

    # casual time is t >= 0. the length n includes samples < 0, so the length of the casual time is half of n
    t_casual = np.arange(0, (ds * (n / 2)), ds)

    # check that the frequency map from fftfreq matches the data freq
    fft_freq = np.fft.fftfreq(n, d=ds)
    assert np.max(fft_freq[: len(data_f)] - data_f) < 1e-3

    # compute inverse FFT
    xt = np.fft.ifft(ifft_spectrum, axis=0).real
    # center the spectrum at 0s, move negative time samples to the beginning
    xt = np.fft.fftshift(xt, axes=0)
    # time vector, with 0s in the middle
    time = np.concatenate((-np.flip(t_casual)[:-1], t_casual))
    
    # data coordinates excluding frequency
    d_coords = {k: data.coords[k] for k in data.coords.keys() if k != "frequency"}
                
    return ldarray(
        xt, coords = dict(time=time, **d_coords)
    )

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


def coupled_sline_impedance(w: float, s: float, b: float, er: float, t: float = 0):
    """
    Odd and even mode impedance for coupled stripline. Units of w, s and b are arbitrary.
    """

    # this version in Modern RF and Microwave Filter Design section 3.7.1, Protap Pramanick
    # seems to be more accurate for the even mode than Matthaei
    Ae = (np.log(2) + np.log(1 + np.tanh(np.pi * s / (2 * b)))) / (2 * np.pi * np.log(2))
    Ao = (np.log(2) + np.log(1 + ( 1 / np.tanh(np.pi * s / (2 * b))))) / (2 * np.pi * np.log(2))

    Cf = 2 * np.log((2 * b - t) / (b - t)) - (t / b) * ((t * (2 * b - t)) / (b - t)**2)

    Z0_e = (30 * np.pi * (b - t)) / (np.sqrt(er) * (w + Ae * b * Cf))
    Z0_o = (30 * np.pi * (b - t)) / (np.sqrt(er) * (w + Ao * b * Cf))

    # reference odd, even mode impedances.
    # page 174 in Matthaei, even and odd mode impedances of coupled strip line
    # k_e = np.tanh((np.pi / 2) * (w / b)) * np.tanh((np.pi / 2) * (w + s) / b)
    # kp_e = np.sqrt(1 - (k_e **2))

    # k_o = np.tanh((np.pi / 2) * (w / b)) * (1 / np.tanh((np.pi / 2) * (w + s) / b))
    # kp_o = np.sqrt(1 - (k_o **2))

    # Z0_e = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_e) / ellipk(k_e))
    # Z0_o = ((30 * np.pi) / (np.sqrt(er))) * (ellipk(kp_o) / ellipk(k_o))

    # mutual capacitance
    # Z0_e = 1 / vp*Ce
    # Z0_o = 1 / vp*Co
    # Co = Ca + 2 Cab

    return Z0_o, Z0_e



def coupled_sline_fringing_cap(w: float, s: float, b: float, er: float):
    """
    Odd and even mode fringing capacitance for edge coupled stripline, per unit length [m].
    See equations 5.05-24 and 5.05-25 (page 201) and Figure 5.05-13, Matthaei. Units of w, s, and b are arbitrary.

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
    # Cf = (2 / np.pi) * np.log((1 / (1 - t/b) + 1)) - (t / (np.pi * b)) * np.log((1 / (1 - t/b)**2 - 1))
    Cf = 0.44
    # solve for even and odd fringing capacitances, equation 5.05-24 and 5.05-25
    Cf_e = (Ce / 2) - Cp - Cf
    Cf_o = (Co / 2) - Cp - Cf

    # return unnormalized capacitance in farads
    eps = er * const.e0
    return Cf_o * eps, Cf_e * eps


def ustrip_impedance(w: float, h: float, er: float, frequency: np.ndarray = 1e9, t: float = 0.0013) -> tuple:
    """
    Microstrip impedance and effective permittivity for thin microstrip lines

    Parameters
    ----------
    w : float, optional
        Width of microstrip trace in inches.
    h : float
        Substrate thickness in inches
    er : float | list
        Dielectric constant of substrate. Supports single values or a list for each frequency. 
    frequency : list, optional
        Array of frequencies in Hz.
    t : float, optional
        Copper thickness in inches, defaults to 1oz copper (1.3 mils).

    Returns
    -------
    Z0 : float | np.ndarray
        characteristic line impedance at each frequency

    eps_eff : float | np.ndarray
        effective permittivity at each frequency

    References
    ----------
    [1] Design-Data-for-Microstrip-Transmission-Lines-on-RT-duroid-Laminates, Rogers Corporation, 2013
    """

    f_isscalar = np.isscalar(frequency)
    frequency = np.atleast_1d(frequency)
    f_ghz = frequency / 1e9

    # B is in mm
    B = conv.mm_mil(h * 1e3)
    T = t / h
    U = w / h
    eta = 376.73

    U1 = U + (T * np.log(1 + ((4 * np.e) / T) * (np.tanh((6.517 * U) ** (1 / 2))) ** 2)) / np.pi
    Ur = U + ((U1 - U) * (1 + 1 / (np.cosh((er - 1) ** 0.5)))) / 2

    Au = 1 + (np.log((Ur**4 + (Ur**2 / 2704)) / (Ur**4 + 0.432))) / 49 + np.log((Ur / 18.1) ** 3 + 1) / 18.7
    Ber = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053

    Y = (er + 1) / 2 + ((er - 1) / 2) * (1 + (10 / Ur)) ** (-Au * Ber)

    def Z01(x):
        return (
            eta * np.log((6 + (2 * np.pi - 6) * np.exp(-((30.666 / x) ** 0.7528))) / x + ((4 / (x**2)) + 1) ** 0.5)
        ) / (2 * np.pi)

    Z0 = Z01(Ur) / (Y**0.5)
    eff = Y * (Z01(U1) / Z01(Ur)) ** 2

    P1 = 0.27488 + U * (0.6315 + 0.525 * (0.0157 * f_ghz * B + 1) ** -20) - 0.06583 * np.e ** (-8.7513 * U)
    P2 = 0.33622 * (1 - np.e ** (-0.03442 * er))
    P3 = 0.0363 * np.e ** (-4.6 * U) * (1 - np.e ** (-(((f_ghz * B) / 38.7) ** 4.97)))
    P4 = 2.751 * (1 - np.e ** (-((er / 15.916) ** 8))) + 1

    P = P1 * P2 * (f_ghz * B * (0.1844 + P3 * P4)) ** 1.5763

    eff_f = er - ((er - eff) / (1 + P))
    Z0_f = Z0 * (eff / eff_f) ** 0.5 * ((eff_f - 1) / (eff - 1))

    # return values to scalar values if frequency was a scalar
    if f_isscalar:
        Z0_f, eff_f = Z0_f.item(), eff_f.item()

    return Z0_f, eff_f


def coupled_ustrip_impedance(
    w: float, h: float, s: float, er: float, frequency: np.ndarray = 1e9, t: float = 0.0013
) -> tuple:
    """
    Coupled microstrip line impedance for odd and even modes. See section 3.8.2 in [1], but be aware there are typos.
    Another good reference with the same equations: https://qucs.github.io/tech/node77.html.

    Parameters
    ----------
    w : float, optional
        Width of microstrip traces in inches.
    h : float
        Substrate thickness in inches
    s : float
        trace spacing in inches
    er : float
        Dielectric constant of substrate.
    frequency : float | list, default: 1 GHz
        frequencies in Hz.
    t : float, optional
        Copper thickness in inches, defaults to 1oz copper (1.3 mils).

    Returns
    -------
    Zo : float | np.ndarray
        Odd mode impedance

    Ze : float | np.ndarray
        Even mode impedance

    References
    ----------
    [1] P. Pramanick and P. Bhartia, Modern RF and Microwave Filter Design. Norwood, MA, USA: Artech House, 2016. 
    
    """

    u = w / h
    g = s / h

    ZL, eps_eff = ustrip_impedance(w, h, er, frequency, t)

    v = (u * (20 + g**2) / (10 + g**2)) + g * np.exp(-g)
    a_e = 1 + (1 / 49) * np.log((v**4 + (v / 52)**2) / (v**4 + 0.432)) + (1 / 18.7) * np.log(1 + (v / 18.1)**3)
    b_e = 0.564 * ((er - 0.9) / (er + 3))**0.053

    a0 = 0.7287 * (eps_eff - (er + 1) / 2) * ( 1 - np.exp(-0.179 * u))
    b0 = 0.747 * er / (0.15 + er)
    c0 = b0 - (b0 - 0.207) * np.exp(-0.414 * u)
    d0 = 0.593 + 0.694 * np.exp(-0.562 * u)

    # even and odd mode effective permittivity
    eps_e = (er + 1) / 2 + (er - 1) / 2 * (1 + (10 / v)) ** (-a_e * b_e)
    eps_o = (((er + 1) / 2) + a0 - eps_eff) * np.exp(-c0 * (g ** d0)) + eps_eff

    # static even mode impedance
    Q1 = 0.8695 * u **(0.194)
    Q2 = 1 + 0.7519* g + 0.189 * g ** 2.31
    Q3 = 0.1975 + (16.6 + (8.4 / g) **6)**(-0.387) + (np.log((g**10) / (1 + (g / 3.4)**10)) / 241)
    Q4 = (2 * Q1) / (Q2 * ((u **-Q3) * np.exp(-g) + (2 - np.exp(-g) * u ** (-Q3))))
    Ze = (ZL * np.sqrt(eps_eff / eps_e)) / (1 - Q4 * ( ZL / const.eta0) * np.sqrt(eps_eff))

    # static odd mode impedance
    Q5 = 1.794 + 1.14 * np.log(1 + 0.638 / (g + 0.517 * g ** 2.43))
    Q6 = 0.2305 + (np.log(g**10 / (1 + (g / 5.8) ** 10)) / 281.3) + (np.log(1 + 0.598 * g ** (1.154)) / 5.1)
    Q7 = (10 + 190 * g**2) / (1 + 82.3 * g **3)
    Q8 = np.exp(-6.5 - 0.95 * np.log(g) - (g / 0.15) ** 5)
    Q9 = np.log(Q7) * (Q8 + 1 / 16.5)
    Q10 = (Q2 * Q4 - Q5 * np.exp(np.log(u) * Q6 * u ** (-Q9))) / Q2
    Zo = (ZL * np.sqrt(eps_eff / eps_o)) / (1 - Q10 * ( ZL / const.eta0) * np.sqrt(eps_eff))

    # TODO: implement the impedance equations with frequency dispersion.

    return Zo, Ze


def ustrip_fringing_cap(w: float, h: float, er: float):
    """
    Fringing capacitance from edge of uncoupled microstrip line, per unit length [m].
    This is a weak function of the width of the line.

    References
    ----------
    [1] Microstrip Filters for RF/Microwave Applications. Jia-Sheng Hong, M. J. Lancaster
    """
    # characteristic impedance and effective epsilon
    zc, er_eff  = ustrip_impedance(w, h, er)

    # parallel plate capacitance
    Cp = er * const.e0 * (w / h)
    # uncoupled fringe capacitance
    Cf = ((np.sqrt(er_eff) / (const.c0 * zc)) - Cp) / 2

    return Cf


def coupled_ustrip_fringing_cap(w: float, s: float, h: float, er: float):
    """
    Odd and even mode fringing capacitance for edge coupled microstrip line, per unit length [m].
    See microstrip_coupled_capacitance.pdf in docs/solver

    Assumes that thickness is zero.

    References
    ----------
    [1] Microstrip Filters for RF/Microwave Applications. Jia-Sheng Hong, M. J. Lancaster
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

def butterworth_prototype(n: int):
    """
    Prototype values for maximally flat low-pass filter. See section 3.2.1 in [1]. Values include
    the normalized source and load resistance.

    Parameters
    ----------
    n : int
        filter order.

    References
    ----------
    [1] Microstrip Filters for RF/Microwave Applications. Jia-Sheng Hong, M. J. Lancaster

    """

    g_i = [float(2 * np.sin(((2 * i - 1) * np.pi) / (2 * n))) for i in range(1, n + 1)]

    return [1] + g_i + [1]

def chebyshev_prototype(n: int, ripple: float):
    """
    Prototype values for chebyshev low-pass filter. See section 3.2.1 in [1]. Values include
    the normalized source and load resistance.

    Parameters
    ----------
    n : int
        filter order.
    ripple : float
        passband ripple in dB

    References
    ----------
    [1] Microstrip Filters for RF/Microwave Applications. Jia-Sheng Hong, M. J. Lancaster

    """
    beta = np.log(1 / np.tanh(ripple / 17.37))
    gamma = np.sinh(beta / (2 * n))

    g = [1] * (n + 2)

    g[1] = (2 / gamma) * np.sin(np.pi / (2 * n))

    for i in range(2, n + 1):
        g[i] = (1 / g[i - 1]) * (
            4 * np.sin(((2 * i - 1) * np.pi) / (2 * n)) * np.sin(((2 * i - 3) * np.pi) / (2 * n))
        ) / (gamma **2 + np.sin(((i - 1) * np.pi) / n) ** 2)

    g[n + 1] = 1 if (n % 2) else (1 / np.tanh(beta / 4))**2

    return [float(i) for i in g]



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

