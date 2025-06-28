import re
from pathlib import Path
import numpy as np

def get_pnum_from_snp(path: str | Path) -> int:
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