import numpy as np
from pathlib import Path
import re
from typing import Tuple, Union
from np_struct import ldarray

def read_snp(filepath: Union[Path, str]) -> Tuple[ldarray, ldarray, list]:
    """
    Reads a touchstone file and returns the s-matrix data, as well as the noise parameters if they are found in the 
    file.

    The frequency vectors for the s-matrix and the noise parameters may be different.
    If noise parameters are not found in the file, None is returned. 

    Parameters
    ----------
    filepath: (Path | str)
        path to .sNp touchstone file, where N is the number of ports.

    Returns
    -------
    s-matrix : ldarray
        labeled array with dimensions (frequency, b, a)
    noise_parameters : ldarray
        Fx3 labeled array with dimensions (frequency, noise_param). 
        The noise param dimension has three values,
        - "nf_min" : minimum noise figure (linear)
        - "gamma_opt" : complex valued source reflection coefficient for optimum noise figure.
        - "rn" : noise resistance, normalized to reference impedance
    comments: list
        list of comment strings found in the file
    """
    # list of comments found in the file
    comments = []
    # 1D list of data values, as strings. Both s-params and noise params are stored here, concatenated together
    rawdata = []
    # option row parameters
    options = []
    # option row indices for each field 
    FREQ = 0
    TYPE = 1
    FORMAT = 2
    REF_Z = 4
    # mappings to convert frequencies in file to hz
    freq_unit = {'hz': 1e0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}

    # get the number of ports (N) from the file suffix
    suffix = Path(filepath).suffix
    match = re.compile(r'\.s(\d+)p', re.IGNORECASE).match(suffix)

    if match is None or len(match.groups()) < 1:
        raise ValueError("Invalid extension for touchstone file. Expected *.sNp, where N is the number of ports.")
    N = int(match.group(1))

    # read the comments and format header
    with open(filepath, 'r', encoding='utf-8', errors="ignore") as f:

        for line in f:
            line = line.strip()

            if not len(line):
                continue
            
            # parse the option line
            elif (line[0] == '#'):
                # sometimes noise data has a separate option line, the standard appears to be that there shouldn't be,
                # so ignore a second option line. Noise parameters are always interpreted as dB values.
                # https://cdn.macom.com/applicationnotes/AN3009.pdf
                if not len(options):
                    options = [x.lower() for x in line[1:].split()]

            # comment lines are preceded with "!"
            elif (line[0] == '!'):
                comments.append(line[1:])
            # split data rows by whitespace and append the string to a list
            else:
                rawdata += line.split()

    # convert string values to floats and cast as a numpy array. This is a flattened (single row) of data values
    data_flt = np.asarray(rawdata, dtype=np.float64)

    # each frequency has N**2 values associated it with it, and each value is represented in the touchstone
    # file with two data fields (i.e. real/imag). Get the frequency vector by skipping over the data fields
    frequency = data_flt[::2*int(N**2) + 1]

    # if noise data is present, the frequency vector will jump to a lower value after the s-parameter data. 
    # Get the index where this occurs.
    f_diff = np.diff(frequency) < 0

    # parse noise parameter data if present
    if np.any(f_diff):
        f_boundary = np.argmax(f_diff) + 1 
        # separate the list for the s-parameter frequency
        s_freq = frequency[:f_boundary]
        # separate the noise data from the data list, convert the f_boundary index to an index for the data values
        s_boundary = f_boundary * (2*int(N**2) + 1)
        sdata_flt = data_flt[:s_boundary]
        ndata_flt = data_flt[s_boundary:]

        # reshape the flattened noise data to a matrix with a row per frequency, there are always five fields
        # per frequency: (frequency, min NF, |Gamma Opt|, ang(Gamma Opt), Rn)
        # The min NF is always in dB, and the Gamma Opt angle is always in degrees, these do not follow the option
        # line. Rn is normalized to REF_Z
        n_freq = ndata_flt[::5]
        ndata = np.reshape(ndata_flt, (len(n_freq), 5))

        # convert the min nf to linear
        nf_min = 10 ** (ndata[:, 1] / 10) 
        # convert gamma opt to a complex number
        gmma_opt = ndata[:, 2] * np.exp(1j * np.deg2rad(ndata[:, 3]))
        # get noise resistance (normalized to 50 ohms)
        rn = ndata[:, 4]

        # compile all the noise terms into a complex matrix
        ndata = np.column_stack([nf_min, gmma_opt, rn])

        # convert frequency to hz
        n_freq = n_freq*freq_unit[options[FREQ]]
    
    else:
        s_freq, sdata_flt = frequency, data_flt
        n_freq, ndata = None, None

    # reshape the flattened sdata so each frequency has it's own row, then remove the frequency column so we're left
    # with just the sdata
    sdata_m = np.reshape(sdata_flt, (len(s_freq), (2*int(N**2) + 1)))[:, 1:]
    # reshape the data again to form MxNx2 matrix (two values per element in s-matrix)
    sdata_m = np.reshape(sdata_m, (-1, N, N, 2))

    # combine the two data parts based on the format option into the first index of the last dimension
    part1, part2 = sdata_m[..., 0], sdata_m[..., 1]
    if options[FORMAT] =='ma':
        sdata = part1 * np.exp(1j * np.deg2rad(part2))
    elif options[FORMAT] =='ri':
        sdata = part1 + 1j * part2
    else: #options[FORMAT] =='db'
        sdata = (10** (part1 / 20))*np.exp(1j*np.deg2rad(part2))	

    # convert frequency to hz
    s_freq = s_freq*freq_unit[options[FREQ]]

    ##S21 precedes S12 for 2-port touchstone files, swap them
    if (N == 2):
        sdata = np.transpose(sdata, (0,2,1))

    # return as a labeled numpy array
    sdata_ld = ldarray(sdata, coords=dict(frequency=s_freq, b=np.arange(1,N+1), a=np.arange(1,N+1)))

    if n_freq is not None:
        ndata_ld = ldarray(ndata, coords=dict(frequency=n_freq, noise_param=["nf_min", "gamma_opt", "rn"]))
    else:
        ndata_ld = None

    return sdata_ld, ndata_ld, comments

