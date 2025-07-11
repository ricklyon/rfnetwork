import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .core.units import conv
from . core import core
import mpl_markers as mplm
from np_struct import ldarray
from typing import List, Tuple

# conversion functions for supported formats
fmt_func = {
    "mag": np.abs,
    "db": conv.db20_lin,
    "ang": lambda x: np.angle(x, deg=True),
    "ang_unwrap": lambda x: np.rad2deg(np.unwrap(np.angle(x))),
    "vswr": conv.vswr_gamma,
    "real": np.real,
    "imag": np.imag,
    "realz": lambda x: np.real(conv.z_gamma(x, refz=50)),
    "imagz": lambda x: np.imag(conv.z_gamma(x, refz=50)),
    "smith": np.imag,
}

# labels for supported formats
fmt_label = {
    "db": "[dB]",
    "ang": "Phase [deg]",
    "ang_unwrap": "Phase [deg]",
    "vswr": "VSWR",
    "realz": "Real Impedance [Ohms]",
    "imagz": "Imag Impedance [Ohms]",
    "freq": "Frequency [GHz]",
    "smith": "",
    "group_delay": "Group Delay [ns]",
    "time_delay": "Time Delay [ns]",
    "nf": "Noise Figure [dB]",
    "frequency": "Frequency [Mhz]"
}

# prefix for paths in the legend
fmt_prefix = {
    "db": "S",
    "ang": "S",
    "ang_unwrap": "S",
    "vswr": "VSWR",
    "realz": "Z",
    "imagz": "Z",
    "smith": "S",
    "group_delay": "$\\tau$",
    "time_delay": "$\\tau$",
    "nf": "NF",
}


def smithchart_marker(ax: plt.Axes, frequency: np.ndarray, fc: float):
    """ place a smith chart marker by frequency rather than x/y position """
    
    f_idx = np.argmin(np.abs(frequency - fc))
    return mplm.line_marker(
        idx=f_idx, axes=ax, xline=False, yformatter=lambda x, y, idx: f"{frequency[idx]/1e6:.0f}MHz"
    )

def smith_circles(values: list, line_type: str, n_points = 5001, gamma_clip: float = 1) -> np.ndarray:
    """
    Generate smith chart impedance/admittance circles

    Parameters
    ----------
    values : list
        normalized values of circles 
    line_type : {'r', 'x', 'g', 's', 'gamma', 'vswr'}
        Line type. Support values are,
        - "r": real
        - "x": reactance, conductance ("g"), susceptance ("s")
        - "g": conductance,
        - "s": susceptance,
        - "gamma": gamma circle with constant radius (0-1) from origin
        - "vswr": VSWR circle
    n_points : float, default: 5001
        resolution of circles
    gamma_clip : float, default: 1
        maximum gamma value before circles are clipped. Allows active smith charts if >1.
    
    Returns
    -------
    circles : np.ndarray
        complex-valued circle data. Plot onto an axis with
        ``ax.plot(np.real(circles), np.imag(circles))``
    """

    values = np.atleast_1d(values)

    old_settings = np.geterr()
    np.seterr(all='ignore')

    # circle center (x0, y0), and the circle radius for each line type
    line_xyr = dict(
        r = (values / (1 + values), np.zeros_like(values), 1 / (1+values)),
        x = (np.ones_like(values), 1 / values, 1 / values),
        g = (-(values / (1 + values)), np.zeros_like(values), 1 / (1+values)),
        s = (-np.ones_like(values), - 1 / values, 1 / values),
        gamma = (np.zeros_like(values), np.zeros_like(values), values), 
        vswr = (np.zeros_like(values), np.zeros_like(values), (values - 1) / (values + 1))
    )
    
    # generate lines
    x0, y0, r = line_xyr[line_type]
    ang = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
    circles = r[..., None] * np.exp(1j * ang)[None] + (x0 + 1j * y0)[..., None]
    # clip circles beyond gamma=1 (or whatever the user specified)
    circles = np.where(np.abs(circles) > gamma_clip * (1 + 1e-6), np.nan, circles)

    np.seterr(**old_settings)

    # the transpose can be directly plotted: ax.plot(np.real(circles), np.imag(circles))
    return circles.T

def stability_circles(sdata: ldarray):
    """
    Get source and load stability circles.
    For the typical case where S(11) and S(22) are less than 1, the unstable region is enclosed by the circle if it 
    does not contain the origin, otherwise the unstable region is outside the circle.

    Parameters
    ----------
    sdata : np.ndarray
        s-matrix data at a single frequency
    
    Returns
    -------
    source_data : np.ndarray
        complex-valued source stability data. Plot onto an axis with
        ``ax.plot(np.real(circles), np.imag(circles))``

    load_data : np.ndarray
        complex-valued circle data. Plot onto an axis with
        ``ax.plot(np.real(circles), np.imag(circles))``
    """

    # remove unitary frequency data
    sdata = sdata.squeeze()

    s11 = sdata[0, 0]
    s22 = sdata[1, 1]
    s21 = sdata[1, 0]
    s12 = sdata[0, 1]

    def stability_circle_output(s11, s12, s21, s22):
        del_ = s11*s22 - (s12*s21)
        r = np.abs((s12*s21)/(np.abs(s22)**2 - np.abs(del_)**2))
        c = np.conj(s22 - (del_*np.conj(s11))) / (np.abs(s22)**2 - np.abs(del_)**2)
        return r, c

    def stability_circle_input(s11, s12, s21, s22):
        del_ = s11*s22 - (s12*s21)
        r = np.abs((s12*s21)/(np.abs(s11)**2 - np.abs(del_)**2))
        c = np.conj(s11 - (del_*np.conj(s22))) / (np.abs(s11)**2 - np.abs(del_)**2)
        return r, c


    rl, cl = stability_circle_output(s11, s12, s21, s22)
    rs, cs = stability_circle_input(s11, s12, s21, s22)

    xdat = np.linspace(0, 2*np.pi, 1000, endpoint=True)
    s_circ = rs*np.exp(1j * xdat) + cs
    l_circ = rl*np.exp(1j * xdat) + cl

    return s_circ, l_circ


def draw_smithchart(
    axes: plt.Axes, 
    impedance_values: list = None, 
    admittance_values: list = None, 
    impedance_kwargs: dict = dict(linewidth=0.25, color="k"),
    admittance_kwargs: dict = dict(color="m", alpha=0.4, linewidth=0.2)
):
    """
    Draws Smith chart lines onto the axes

    Parameters
    ----------
    axes: plt.Axes
        Axes object
    impedance_values : list, optional
        normalized impedance line values, applies to both resistance and reactance lines.
    admittance_values : list, optional
        normalized admittance line values, applies to both conductance and susceptance lines.
    """

    if impedance_values is None:
        impedance_values = [10 , 5, 2, 1, 0.5, 0.2, 0]

    if admittance_values is None:
        admittance_values = [10 , 5, 2, 1, 0.5, 0.2, 0]

    axes.set_aspect('equal')
    axes.set_axis_off()
    axes.axis([-1.1, 1.1, -1.1, 1.1])

    if isinstance(impedance_kwargs, dict) and len(impedance_values):

        r = smith_circles(impedance_values, "r")
        xp = smith_circles(impedance_values, "x")
        xn = smith_circles(-np.array(impedance_values), "x")

        sm_lines = axes.plot(r.real, r.imag, **impedance_kwargs)
        sm_lines += axes.plot(xp.real, xp.imag, **impedance_kwargs)
        sm_lines += axes.plot(xn.real, xn.imag, **impedance_kwargs)
        # draw x axis line
        sm_lines += axes.plot([-1, 1], [0, 0], **impedance_kwargs)

    if isinstance(admittance_kwargs, dict) and len(admittance_values):
        g = smith_circles(admittance_values, "g")
        sp = smith_circles(admittance_values, "s")
        sn = smith_circles(-np.array(admittance_values), "s")

        sm_lines += axes.plot(g.real, g.imag, **admittance_kwargs)
        sm_lines += axes.plot(sp.real, sp.imag, **admittance_kwargs)
        sm_lines += axes.plot(sn.real, sn.imag, **admittance_kwargs)

    # prevent markers from attaching to smith chart lines
    mplm.disable_lines(sm_lines, axes=axes)

    # add a flag that this axes has been formatted for a smith chart already
    axes._smithchart = True

def plot(
    axes: plt.Axes,
    sdata: ldarray,
    *paths : Tuple[int],
    ndata: ldarray = None,
    fmt: str = "db",
    freq_unit: str = "ghz",
    ref: tuple = None,
    label: str = "",
    label_mode: str = "prefix",
    lines: List[Line2D] = None,
    **kwargs
) -> List[Line2D]:
    """
    Plots s-matrix or noise figure data over frequency.

    Parameters
    ----------
    axes : plt.Axes
        matplotlib axes object
    sdata : ldarray
        Labeled numpy array of s-matrix data with dimensions, (frequency, b, a). 
    *paths : tuple | int
        Port paths to plot. Each path can be an integer or a tuple of port numbers. For example, 21 is equivalent
        to (2, 1) and plots S21. If no arguments are given every possible path will be plotted.
    ndata : ldarray
        Labeled numpy array of noise correlation matrix with dimensions (frequency, b, a). Used only if fmt is "nf".
    fmt : str, default: "db"
        data format for y-axis data. Accepts the following values
        - "mag": Magnitude
        - "db" : 20log of magnitude 
        - "ang" Phase angle
        - "ang_unwrap": Unwrapped phase angle
        - "vswr" : Voltage standing wave ratio
        - "real" : Real part of the complex s-matrix data
        - "imag" : Imaginary part of the complex s-matrix data
        - "realz" : Real part of the port input impedance
        - "imagz" : Imaginary part of the port input impedance
        - "nf" : Noise figure
    freq_unit : {"Hz", "kHz", "MHz", "GHz"}, default: "GHz"
        Unit for frequency axis. 
    ref : tuple | int
        Normalizes all plotted paths to this path. For example, to plot the phase difference between S21 and S31,
        ``.plot(21, ref=31, fmt="ang")``. Supports a list of tuples or integers the same length as the number of paths,
        where each path is normalized to a different reference path.
    label : str | list, default: ""
        Legend labels for plotted lines. Supports either a string to add a common label for all lines,
        or a list of strings for each line. 
    label_mode : {"prefix", "suffix", "override"}, default: "prefix"
        Controls the placement of the line labels relative to the default label of "S(b,a)". By default, 
        labels are a "prefix" to the default label. "override" replaces the default label.
    lines : list[Line2D], optional
        Line2D objects for each path. If provided, updates the existing lines instead of drawing new ones on the plot.
    **kwargs
        parameters passed into :meth:`matplotlib.axes.plot`.

    Returns
    -------
    lines : list[Line2D]
        list of line objects that were created for each path. If ``lines`` parameter was used, returned lines
        are the same as the ``lines`` parameter.

    """


    n_ports = sdata.shape[-1]
    # plot all possible paths if not given
    if not len(paths) and fmt != 'smith':
        paths = [(i, j) for i in range(1, n_ports + 1) for j in range(1, n_ports + 1)]
    elif not len(paths):
        # show only the input return paths if plotting a smith chart
        paths = [(i, i) for i in range(1, n_ports + 1)]

    # generate a list of tuples of port numbers for each path
    if isinstance(paths[0], int):
        # break integers into sets of 2-tuples: 21 -> (2,1)
        paths = [((p // 10), (p % 10)) for p in paths]

    # break reference path into list of tuples
    if ref is not None:

        # check that noise figure is not plotted when reference is provided
        if fmt in ["nf"]:
            raise ValueError("Plotting against a reference path is not supported for noise figure plots.")
        
        # broadcast single valued reference paths to the length of paths
        if not isinstance(ref, list):
            ref = [ref] * len(paths)

        if isinstance(ref[0], int):
            # break integers into sets of 2-tuples: 21 -> (2,1)
            ref = [((r // 10), (r % 10)) for r in ref]

        if len(ref) != len(paths):
            raise ValueError(f"Reference paths must be the same length as paths. Got {len(ref)}, expected {len(paths)}")

    # create frequency vector and x-axis label
    f_multiplier = dict(hz=1, khz=1e3, mhz=1e6, ghz=1e9)[freq_unit.lower()]
    f_label = dict(hz="Hz", khz="kHz", mhz="MHz", ghz="GHz")[freq_unit.lower()]
    xdata = sdata.coords["frequency"] / f_multiplier

    # broadcast label(s) across all paths
    labels = np.broadcast_to(label, len(paths)).copy()
    # assign custom legend labels to either a suffix or prefix, or to completely replace the default label.
    prefix_label = [""] * len(paths)
    suffix_label = [""] * len(paths)
    override_label = None
    
    if label_mode == "override":
        override_label = labels
    elif label_mode == "prefix":
        prefix_label = labels
    elif label_mode == "suffix":
        suffix_label = labels

    # set up axes
    if fmt == "smith" and not hasattr(axes, "_smithchart"):
        draw_smithchart(axes)
    elif lines is None:
        axes.set_xlabel(f"Frequency [{f_label}]")
        axes.set_ylabel(fmt_label.get(fmt))
        axes.grid(True)
        axes.margins(x=0)

    lines = [] if lines is None else lines

    # plot over each path given, keep track of the index for tuning calls where each line in a list is updated
    for i, (p1, p2) in enumerate(paths):
        
        # pull the path data out
        path_data = sdata[{"b": p1, "a": p2}]

        # divide by the reference data
        if ref is not None:
            r1, r2 = ref[i]
            path_data /= sdata[{"b": r1, "a": r2}]

        # apply formatting to data (convert to dB, ang, etc...)
        if fmt == "nf":
            ydata = conv.db10_lin(core.noise_figure_from_ndata(sdata, ndata, (p1, p2)))
        else:
            ydata = fmt_func[fmt](path_data)

        # transform xdata to the real part of the ydata if plotting for a smithchart
        xdata_s = fmt_func["real"](path_data) if fmt == "smith" else xdata

        # if line is already created and passed in, update the existing line.
        if len(lines) > i:
            lines[i].set_data(xdata_s, ydata)

        # plot like normal if lines were not provided
        else:
            # create default label of "S(b,a)", i.e. S(2,1).
            if override_label is None:
                # escape underscores in port names
                p1_name = r"\mathrm{" + str(p1).replace("_", "\\_") + "}"
                p2_name = r"\mathrm{" + str(p2).replace("_", "\\_") + "}"

                label = prefix_label[i]
                # build legend label for line, e.g S(2,1)
                label += r"{}$({{ {},{} }})$".format(fmt_prefix[fmt], p1_name, p2_name)

                # add a "divide by" path to the label if a reference is given
                if ref is not None:
                    r1_name = r"\mathrm{" + str(r1).replace("_", "\\_") + "}"
                    r2_name = r"\mathrm{" + str(r2).replace("_", "\\_") + "}"
                    label += r" / {}$({{ {},{} }})$".format(fmt_prefix[fmt], r1_name, r2_name)

                label += suffix_label[i]
            # use custom label and skip default
            else:
                label = override_label[i]

            # create line and add to list of newly created line objects
            lines += axes.plot(xdata_s, ydata, label=label, **kwargs)

    return lines
