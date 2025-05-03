import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .core.units import conv
from . core import core
import mpl_markers as mplm

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

def smith_circles(values: list, line_type: str, npoints = 5001, gamma_clip: float = 1):
    """
    Generate smith chart impedance/admittance circles

    line_type: {'r', 'x', 'g', 's', 'gamma', 'vswr'}
        real, reactance, conductance, susceptance
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
    np.seterr(**old_settings)

    # generate lines
    x0, y0, r = line_xyr[line_type]
    ang = np.linspace(0, 2 * np.pi, npoints, endpoint=True)
    circles = r[..., None] * np.exp(1j * ang)[None] + (x0 + 1j * y0)[..., None]
    # clip circles beyond gamma=1 (or whatever the user specified)
    circles = np.where(np.abs(circles) > gamma_clip * (1 + 1e-6), np.nan, circles)

    # the transpose can be directly plotted: ax.plot(np.real(circles), np.imag(circles))
    return circles.T


def format_smithchart(ax: plt.Axes, values=None, line_kwargs=dict(linewidth=0.25, color="k")):

    if values is None:
        values = [10 , 5, 2, 1, 0.5, 0.2, 0]

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.axis([-1.1, 1.1, -1.1, 1.1])

    r = smith_circles(values, "r")
    xp = smith_circles(values, "x")
    xn = smith_circles(-np.array(values), "x")

    sm_lines = ax.plot(r.real, r.imag, **line_kwargs)
    sm_lines += ax.plot(xp.real, xp.imag, **line_kwargs)
    sm_lines += ax.plot(xn.real, xn.imag, **line_kwargs)

    # draw xy axis
    sm_lines += ax.plot([-1, 1], [0, 0], **line_kwargs)
    sm_lines += ax.plot([0, 0], [-1, 1], **line_kwargs)

    mplm.disable_lines(sm_lines, axes=ax)


def plot(
    data,
    *paths,
    fmt="db",
    freq_unit="ghz",
    axes=None,
    label=None,
    smithchart_kwargs={},
    **line_kwargs
) -> plt.Axes:
    """
    Plots s-matrix data for each path over the primary axis. Matplotlib supports plotting over 2 dimensions, so
    the Sparam can only have 1 dimension above the frequency axis.

    Parameters:
    ------------
    paths (tuple, list):
        list of port paths to plot. (e.g. 21 would plot S21). Each path can be an integer or a tuple of port numbers
        or Port objects. If no arguments are given every possible path will be plotted.
    fmt (string):
        format for y axis data. Accepts the following values. Defaults to db.
            mag
            db
            ang
            ang_unwrap
            vswr
            real
            imag
            realz
            imagz

    refpath (tuple, int):
        Normalizes all paths to this path. For example, to plot the phase difference between S21 and S31:
            .plot(21, refpath=31, fmt='ang')
    axes (Axes):
        Uses the given axes to plot on, will create new Axes if not given.
    xaxis (string):
        Dimension label to use as the independent variable, defaults to 'freq'.
    figsize (tuple):
        figure size to use if axes is not provided. Defaults to (10, 5)
    excitation (np.ndarray):
        voltages incident at each port. If provided, the traces will be active measurements and the paths
        will be referenced to the input voltages.

    Returns:
    ---------
    axes

    """

    if axes is None:
        # create axes if one was not given
        figsize = (5,5) if fmt == "smith" else (7,5)
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(1, 1)

    if fmt == "smith":
        # draw lines on smithchart
        format_smithchart(axes, **smithchart_kwargs)

    sdata = data["s"]
    pnum = sdata.shape[-1]
    # plot all possible paths if not given
    if not len(paths) and fmt != 'smith':
        paths = [(i, j) for i in range(1, pnum + 1) for j in range(1, pnum + 1)]
    elif not len(paths):
        # show only the input return paths if plotting a smith chart
        paths = [(i, i) for i in range(1, pnum + 1)]

    # generate a list of tuples of port numbers for each path
    if isinstance(paths[0], int):
        # break integers into sets of 2-tuples: 21 -> (2,1)
        paths = [((p // 10), (p % 10)) for p in paths]
        path_names = [(str(p1), str(p2)) for p1, p2 in paths]
    else:
        paths = list(paths)
        path_names = [(str(p1), str(p2)) for p1, p2 in paths]

    # get xaxis vector
    f_multiplier = dict(hz=1, khz=1e3, mhz=1e6, ghz=1e9)[freq_unit]
    f_label = dict(hz="Hz", khz="kHz", mhz="MHz", ghz="GHz")[freq_unit]
    xdata = sdata.coords["frequency"] / f_multiplier

    # default line label
    global_label = "" if label is None else label + " "

    # apply labels and grid
    axes.set_xlabel(f"Frequency [{f_label}]")
    axes.set_ylabel(fmt_label.get(fmt))
    axes.grid(True)

    # plot over each path given, keep track of the index for tuning calls where each line in a list is updated
    for i, (p1, p2) in enumerate(paths):
        
        # pull the path data out
        path_data = sdata[{"b": p1, "a": p2}]

        # apply formatting to data (convert to dB, ang, etc...)
        if fmt == "nf":
            ydata = conv.db10_lin(core.noise_figure_from_ndata(data["s"], data["n"], (p1, p2)))
        else:
            ydata = fmt_func[fmt](path_data)

        # transform xdata to the real part of the ydata if plotting for a smithchart
        xdata_s = fmt_func["real"](path_data) if fmt == "smith" else xdata

        # create the label for this line
        p1_name, p2_name = path_names[i]
        label = r"{}{}$_{{ {},{} }}$".format(global_label, fmt_prefix[fmt], p1_name, p2_name)

        # plot like normal if lines were not provided
        axes.plot(xdata_s, ydata, label=label, **line_kwargs)

    axes.margins(x=0)
    axes.legend()

    return axes
