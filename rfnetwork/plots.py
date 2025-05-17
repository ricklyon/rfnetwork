import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .core.units import conv
from . core import core
import mpl_markers as mplm
from np_struct import ldarray

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
    
    # generate lines
    x0, y0, r = line_xyr[line_type]
    ang = np.linspace(0, 2 * np.pi, npoints, endpoint=True)
    circles = r[..., None] * np.exp(1j * ang)[None] + (x0 + 1j * y0)[..., None]
    # clip circles beyond gamma=1 (or whatever the user specified)
    circles = np.where(np.abs(circles) > gamma_clip * (1 + 1e-6), np.nan, circles)

    np.seterr(**old_settings)

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

    g = smith_circles(values, "g")
    sp = smith_circles(values, "s")
    sn = smith_circles(-np.array(values), "s")

    sm_lines = ax.plot(r.real, r.imag, **line_kwargs)
    sm_lines += ax.plot(xp.real, xp.imag, **line_kwargs)
    sm_lines += ax.plot(xn.real, xn.imag, **line_kwargs)

    admittance_kwargs = dict(color="m", alpha=0.4, linewidth=0.2)

    sm_lines += ax.plot(g.real, g.imag, **admittance_kwargs)
    sm_lines += ax.plot(sp.real, sp.imag, **admittance_kwargs)
    sm_lines += ax.plot(sn.real, sn.imag, **admittance_kwargs)

    # draw x axis line
    sm_lines += ax.plot([-1, 1], [0, 0], **line_kwargs)

    mplm.disable_lines(sm_lines, axes=ax)

def plot_stability_circles(sdata: ldarray, f0: float, axes = None, load_kwargs=dict(), source_kwargs=dict()):
    """
    Plot source and load stability circles at f0. 
    """
    if axes is None:
        # create axes if one was not given
        fig = plt.figure(figsize=(7, 7))
        axes = fig.subplots(1, 1)
        format_smithchart(axes)

    s11 = sdata.sel(frequency=f0, b=1, a=1)
    s22 = sdata.sel(frequency=f0, b=2, a=2)
    s21 = sdata.sel(frequency=f0, b=2, a=1)
    s12 = sdata.sel(frequency=f0, b=1, a=2)

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

    l_line = axes.plot(l_circ.real, l_circ.imag, label=r"$\Gamma_L$ Stability", **load_kwargs)
    s_line = axes.plot(s_circ.real, s_circ.imag, label=r"$\Gamma_S$ Stability", **source_kwargs)

    return axes, s_line[0], l_line[0]

def plot(
    sdata,
    *paths,
    fmt="db",
    ndata=None,
    freq_unit="ghz",
    ref_path=None,
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
        figsize = (7,7) if fmt == "smith" else (7,5)
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(1, 1)

        if fmt == "smith":
            # draw lines on smithchart
            format_smithchart(axes, **smithchart_kwargs)

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

    # break reference path into tuple
    if ref_path is not None:
        ref_path = ((ref_path // 10), (ref_path % 10)) if isinstance(ref_path, int) else ref_path

        # check that noise figure is not plotted when reference is provided
        if fmt in ["nf"]:
            raise ValueError("Plotting against a reference path is not supported for noise figure plots.")

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

    lines = []

    # plot over each path given, keep track of the index for tuning calls where each line in a list is updated
    for i, (p1, p2) in enumerate(paths):
        
        # pull the path data out
        path_data = sdata[{"b": p1, "a": p2}]

        # divide by the reference data
        if ref_path is not None:
            r1, r2 = ref_path
            path_data /= sdata[{"b": r1, "a": r2}]

        # apply formatting to data (convert to dB, ang, etc...)
        if fmt == "nf":
            ydata = conv.db10_lin(core.noise_figure_from_ndata(sdata, ndata, (p1, p2)))
        else:
            ydata = fmt_func[fmt](path_data)

        # transform xdata to the real part of the ydata if plotting for a smithchart
        xdata_s = fmt_func["real"](path_data) if fmt == "smith" else xdata

        # escape underscores in port names
        p1_name = r"\mathrm{" + str(p1).replace("_", "\\_") + "}"
        p2_name = r"\mathrm{" + str(p2).replace("_", "\\_") + "}"
        # build legend label for line, e.g S(2,1)
        label = r"{}{}$({{ {},{} }})$".format(global_label, fmt_prefix[fmt], p1_name, p2_name)

        # add a "divide by" path to the label if a reference is given
        if ref_path is not None:
            r1_name = r"\mathrm{" + str(r1).replace("_", "\\_") + "}"
            r2_name = r"\mathrm{" + str(r2).replace("_", "\\_") + "}"
            label += r" / {}$({{ {},{} }})$".format(fmt_prefix[fmt], r1_name, r2_name)

        # plot like normal if lines were not provided
        lines += axes.plot(xdata_s, ydata, label=label, **line_kwargs)

    axes.margins(x=0)
    axes.legend()

    return axes, lines
