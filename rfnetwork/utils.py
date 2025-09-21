import re
from pathlib import Path
import numpy as np

def dtft(xn: np.ndarray, omega: np.ndarray):
    """
    Compute the DTFT of the discrete time signal x[n] over omega (radians per sample).
    """
    n = np.arange(len(xn))
    omega_mesh, n_mesh = np.meshgrid(omega, n)

    # broadcast input sequence across all omega
    x_b = np.broadcast_to(xn[..., None], (len(n), len(omega)))
    # sum across all non-zero n
    Xw = np.sum(x_b * np.exp(-1j* omega_mesh * n_mesh), axis=0)

    return Xw

def dtft_f(xn: np.ndarray, f: np.ndarray, fs: float):
    """
    Compute the DTFT of the discrete time signal x[n] over a frequency range. (cycles per second)
    """
    # convert the continuous time frequency into a discrete frequency range. The discrete frequencies
    # are bounded by -0.5 to 0.5 if there is no aliasing.
    # To convert the continuous time frequency (cycles / sec), into the discrete frequency (cycles / sample). 
    # divide it by the sampling rate fs (samples / sec):
    # (cycles / sec) / (samples / sec) = (cycles / sample)
    fn = f / fs
    return dtft(xn, 2 * np.pi * fn)

def n_ports_from_snp(path: str | Path) -> int:
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
