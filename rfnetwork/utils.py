import re
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve
from scipy.special import ellipk
from .core.units import const

def dtft(xn: np.ndarray, f: np.ndarray, fs: float):
    """
    Compute the DTFT of the discrete time signal x[n] over a frequency range. (cycles per second)
    """
    # convert the continuous time frequency into a discrete frequency range. The discrete frequencies
    # are bounded by -0.5 to 0.5 if there is no aliasing.
    # To convert the continuous time frequency (cycles / sec), into the discrete frequency (cycles / sample). 
    # divide it by the sampling rate fs (samples / sec):
    # (cycles / sec) / (samples / sec) = (cycles / sample)
    fn = f / fs
    omega = 2 * np.pi * fn
    
    n = np.arange(len(xn))
    omega_mesh, n_mesh = np.meshgrid(omega, n)

    # broadcast input sequence across all omega
    x_b = np.broadcast_to(xn[..., None], (len(n), len(omega)))
    # sum across all non-zero n
    Xw = np.sum(x_b * np.exp(-1j * omega_mesh * n_mesh), axis=0)

    return Xw

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
    # fringing capacitance on the outer edges (not between the two lines), figure 5.05-10b, for t=0
    Cf = 0.44
    # solve for even and odd fringing capacitances, equation 5.05-24 and 5.05-25
    Cf_e = (Ce / 2) - Cp - Cf
    Cf_o = (Co / 2) - Cp - Cf

    # return unnormalized capacitance in farads
    eps = er * const.e0
    return Cf_o * eps, Cf_e * eps

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

def blend_cell_widths(
    a: float, b: float, d: float, r_max: float = 1.5, n_min: int = 1, tol: float = 0.0001, dtype_=np.float32
):
    """
    Return a list of cell widths that divide a cell with width d into cells that do not exceed the given
    growth rate r_max. a is the width of the previous cell, b is the width of the next cell.
    """

    # swap a and b if a is larger than b. Always start with the smaller cell and increase up to b
    flip_a_b = False
    if (b < a):
        a, b = b, a
        flip_a_b = True

    # if no grading is needed, split the cell into equal sections
    if ((b / a) < r_max):
        # get number of cells that divides this cell into equal sections as close to a as possible
        n = d / a
        n = np.clip(int(n + 1 if n % 1 > 0.5 else n), n_min, None)
        
        return np.array([d / n] * n, dtype=np.float32)
    
    # if d is less than or equal to a, no grading is possible, return d as the cell width
    if np.abs(d - a) < tol:
        return np.array([d])

    # number of cells can be no larger than d/a (with all cells are of width a)
    n_max = int(d / a) + 1

    for n in range(n_min, n_max+1):
        # growth rate of cells must make the last cell at least 2/3 of b
        # a * m^n > (2/3) b
        m_min = ((1 / r_max) * (b/a))**(1/n)

        # if the minimum growth rate is higher than 1.5, the mesh cannot be resolved with this number of cells,
        # move to the next n
        if m_min > r_max:
            continue
        
        # last cell can be no larger than b, and growth rate can be no larger than r_max
        # a * m^n < b
        m_max = np.clip((b / a)**(1/n), None, r_max)

        cells_m_min = a * np.array([m_min**r for r in range(1, n+1)])
        cells_m_max = a * np.array([m_max**r for r in range(1, n+1)])
        
        # growth rate must satisfy a(m + m^2 + m^3 + ... m^n) = d
        # find the min and max widths that can be created by dividing this section into n cells
        d_min = np.sum(cells_m_min)
        d_max = np.sum(cells_m_max)

        # print(n, d, d_min, d_max)

        # skip this n if the min/max width it can generate fall outside the target width d
        if d_min > d or d_max < d:
            continue

        # at this point, we know that n can generate the target width d. Find the growth rate that does this.
        def compute_d(m):
            return a * np.sum([m**r for r in range(1, n+1)]) - d

        m0 = np.sqrt(m_min * m_max) # initial guess for growth rate m
        m = fsolve(compute_d, x0=m0, args=())
        
        cells_m = a * np.array([m**r for r in range(1, n+1)], dtype=np.float32).flatten()

        if np.abs(np.sum(cells_m) - d) < tol:
            return np.flip(cells_m).astype(dtype_) if flip_a_b else cells_m
        else:
            raise RuntimeError("Mesh did not converge.")
        
    # if no n is able to produce a cell with total width d using a growth rate less than r_max,
    # split the cell equally as a last resort
    return np.array([d / n_min] * n_min)