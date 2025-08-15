import numpy as np
from np_struct import ldarray
from pathlib import Path
from scipy.interpolate import CubicSpline
from copy import deepcopy

from abc import abstractmethod
from typing import Tuple, List, Union

import mpl_markers as mplm
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (QApplication)
import sys

from matplotlib.lines import Line2D

from . import touchstone, utils
from . core import core
from . import plots
from . tuning import TunerGroup

class Component(object):
    """
    Base class for Network Components
    """
    def __init__(
        self, 
        shunt: bool = False, 
        passive: bool = True, 
        n_ports: int = 2, 
        state: dict = dict(), 
    ):
        """
        Parameters
        ----------
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.
        passive : bool, default: True
            if True, ``evaluate`` calls ``evaluate_sdata`` instead of ``evaluate_data`` and noise correlation
            matrix is computed passively.
        n_port : int, default: 2
            number of component ports
        state : dict, optional
            dictionary of state variables specific to the sub-class (i.e. line width, switch state etc...).
            The state values can be read with the ``state`` property and changed later with `set_state()`. Attempting
            to set the state of variables that were not included in the initial dictionary will raise an error.

        """
        self._shunt = shunt
        self._passive = passive
        self._n_ports = n_ports
        self._state = {k: None for k in state.keys()}
        self._tune = dict(frequency=None, args=[], axes=[])

        self.set_state(**state)

    @property
    def frequency(self):
        """
        Generic frequency vector for base components
        """
        frequency = np.arange(10e6, 10.01e9, 10e6)
        return frequency
    
    @property
    def n_ports(self) -> int:
        """
        Number of ports of the component
        """
        return self._n_ports

    @property
    def state(self) -> dict:
        """
        Dictionary of state values
        """
        return self._state

    def set_state(self, **kwargs):
        """
        Change the state variables. Keys are required to have been included in the initial state dictionary passed in
        to the Component constructor.
        """
        for k, v in kwargs.items():
            if k not in self.state.keys():
                raise KeyError(f"Invalid state key, {k}.")
            
            self._state[k] = deepcopy(v)
        
    def __or__(self, other):
        """ 
        Allows port to be indexed with the syntax: component|2 
        """
        return (self, int(other))
    
    def equals(self, other) -> bool:
        """
        Returns True if the s-matrix data from other is equivalent to this object.

        Parameters
        ----------
        other : Component
            Component object to check equality against.
        """
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        
        if self._passive == other._passive != self._shunt == other._shunt:
            return False

        # check that state keys are identical
        state_keys = [k for k in self.state.keys() if k in other.state.keys()]

        if len(state_keys) != len(other.state.keys()) or len(state_keys) != len(self.state.keys()):
            return False
        
        # check state values, they may be numpy arrays. lists or single values
        return all([np.all(self.state[k] == other.state[k]) for k in state_keys])
    
    @abstractmethod
    def evaluate_data(self, frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns s-matrix and noise correlation matrix of the component.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data for, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        ndata : np.ndarray
            MxNxN noise correlation matrix where M is the number of frequency values and N is the number of ports. 
        """
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        """
        Returns s-matrix of the component.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data for, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where F is the number of frequency values and N is the number of ports. 
        """

        raise NotImplementedError()
    
    def __call__(self, **kwargs):
        """
        Returns a copy of the component that is optionally configured to a new state.
        """
        nobj = deepcopy(self)
        nobj.set_state(**kwargs)
        return nobj
    
    def plot(
        self, 
        *paths : Tuple[int], 
        frequency: np.ndarray = None, 
        fmt: str = "db",
        axes: plt.Axes = None, 
        tune : bool = False,
        freq_unit: str = "ghz",
        ref: tuple = None,
        label: str = "",
        label_mode: str = "prefix",
        lines: List[Line2D] = None,
        **kwargs
    ) -> List[Line2D]:
        """
        Plots s-matrix or noise figure data over frequency

        Parameters
        ----------
        *paths : tuple | int
            Port paths to plot. Each path can be an integer or a tuple of port numbers. For example, 21 is equivalent
            to (2, 1) and plots S21. If no arguments are given every possible path will be plotted.
        frequency : np.ndarray, optional
            frequencies [Hz] to plot data over. If not provided, attempts to find a default frequency vector that 
            minimizes extrapolation of component data.
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
        axes : matplotlib.Axes, optional
            Axes object to plot data on. If not provided, an axes is created with the default figure size. 
        tune : bool, optional
            If true, adds the plot as a tuning plot.
        freq_unit : {"Hz", "kHz", "MHz", "GHz"}, default: "GHz"
            Unit for frequency axis. 
        ref : tuple | int
            Normalizes all plotted paths to this path. For example, to plot the phase difference between S21 and S31,
            ``.plot(21, ref=31, fmt="ang")``. Supports a list of tuples or integers the same length as the number of 
            paths,
            where each path is normalized to a different reference path.
        label : str | list, default: ""
            Legend labels for plotted lines. Supports either a string to add a common label for all lines,
            or a list of strings for each line. 
        label_mode : {"prefix", "suffix", "override"}, default: "prefix"
            Controls the placement of the line labels relative to the default label of "S(b,a)". By default, 
            labels are a "prefix" to the default label. "override" replaces the default label.
        lines : list[Line2D], optional
            Line2D objects for each path. If provided, updates the existing lines instead of drawing new ones on the 
            plot.
        **kwargs
            parameters passed into :meth:`matplotlib.axes.plot`.

        Returns
        -------
        lines : list[Line2D]
            list of line objects that were created for each path. If ``lines`` parameter was used, returned lines
            are the same as the ``lines`` parameter.
        """
        data = self.evaluate(frequency, noise=fmt in ["nf"])

        if axes is None:
            _, axes = plt.subplots()

        # access noise correlation matrix only if data format is set to noise figure    
        ndata = data["n"] if fmt in ["nf"] else None
        lines = plots.plot(
            axes, 
            data["s"], 
            *paths, 
            ndata=ndata, 
            fmt=fmt, 
            freq_unit=freq_unit, 
            ref=ref, 
            label=label, 
            label_mode=label_mode,
            **kwargs
        )
        axes.legend()

        # save plot parameters to update the lines later if configured as a tuning plot.
        if tune:
            if self._tune["frequency"] is None:
                self._tune["frequency"] = frequency

            # check that frequency is the same for all tune plots
            elif (frequency.shape != self._tune["frequency"].shape) or not np.all(self._tune["frequency"] == frequency):
                raise RuntimeError("Frequency vectors must be identical for all tuning plots.")

            # save plot arguments so the plot method can invoked again in the tuning callback
            self._tune["args"] += [(axes, paths, fmt, lines, freq_unit, ref)]
            # avoid duplicates in axes list
            if axes not in self._tune["axes"]:
                self._tune["axes"] += [axes]
    
        return lines
    
    def tune(self, tuners: List[dict]):
        """
        Open a tuning window.

        Parameters
        ----------
        tuners : list
            configuration dictionaries for tuned variables. Each dictionary requires the following key-value pairs,
            - "variable" : Name of a state variable.
            - "lower" : Floating point lower bound of tuner.
            - "upper" : Floating point upper bound of tuner.
            - "label" : Label for tuner.
            - "component" : If tuning a network, the component designator must be specified for each tuned variable. 
            Supports nested Networks by using a "." between sub-networks. For example "m_in.c1" would point to the 
            "c1" component of the "m_in" sub-network.

        Examples
        --------

        >>> import rfnetwork as rfn
        >>> c = rfn.elements.Capacitor_pF(10)
        >>> c.tune(dict(variable="value", lower=1, upper=30, label="C1 [pF]"))
        """
        
        # cast single dictionaries as a list
        if isinstance(tuners, dict):
            tuners = [tuners]

        # add callback functions and initial values to config dictionaries
        for config in tuners:

            # drill down to a sub-component if this is a network
            if "component" in config:
                component = self
                for c in config["component"].split("."):
                    component = component[c]
            else:
                component = self

            config["callback"] = component.set_state
            config["initial"] = component.state[config["variable"]]

        for ax in self._tune["axes"]:
            mplm.init_axes(ax)

        # bring the plot window up
        plt.show(block=False)

        window = TunerGroup(tuners, self._update_tune_plots)
        window.setWindowTitle("Tuning")

        # start the tuner app
        qapp = QApplication.instance()
        if qapp is None:
            qapp = QApplication(sys.argv)

        window.show()
        qapp.exec()

    def reset_tune_plots(self):
        """
        Clears all plots added as tuning plots.
        """
        self._tune = dict(frequency=None, args=[], axes=[])

    def _update_tune_plots(self):
        """
        Callback for tuner events that redraws the tuning plots.
        """

        if self._tune["frequency"] is None:
            return
        
        # call evaluate once for all plots, assumes frequency is the same
        nf = any([tune_param[0] in ["nf"] for tune_param in self._tune["args"]])

        data = self.evaluate(self._tune["frequency"], noise=nf)
        ndata = data["n"] if nf else None

        # update the data lines
        for (axes, paths, fmt, lines, freq_unit, ref) in self._tune["args"]:
            plots.plot(axes, data["s"], *paths, fmt=fmt, ndata=ndata, lines=lines, freq_unit=freq_unit, ref=ref)
        
        # blit the axes
        for ax in self._tune["axes"]:
            mplm.draw_all(ax)
    
    def evaluate(self, frequency: np.ndarray = None, noise: bool = False) -> dict:
        """
        Computes the component s-matrix and (optionally) the noise correlation matrix.

        Parameters
        ----------
        frequency : np.ndarray, optional
            Frequency vector to compute data over. If not provided, attempts to find a default frequency vector that 
            minimizes extrapolation of component data.
        noise : bool, default: False
            If True, computes the noise correlation matrix of the component.

        Returns
        -------
        dict
            dictionary with the s-matrix data in the "s" key, and the noise correlation matrix in "n".
            Matrices are labeled numpy arrays with dimensions (frequency, b, a).
        """

        if frequency is None:
            frequency = self.frequency

        frequency = np.atleast_1d(frequency)
        
        # compute noise data only if not passive
        if self._passive or not noise:
            sdata = self.evaluate_sdata(frequency)
            ndata = None

        else:
            sdata, ndata = self.evaluate_data(frequency)
        
        # ground port two of a 2-port device if shunted. 
        if self._shunt:
            # raises an error if not a 2-port
            assert sdata.shape[-1] == 2, "must be 2-port"

            short = np.full((len(frequency), 1, 1), -1, dtype="complex128")
            _, shunt_data = core.connect(dict(s=sdata), dict(s=short), (2, 1))

            # connect port 1 to a junction so the open port can be connected to 2 other components
            node = core.junction_sdata(frequency, 3)
            _, shunt_data = core.connect(shunt_data, dict(s=node), (1, 1))
            sdata = shunt_data["s"]

            # invalidate noise data, shunting active components is not supported.
            ndata = None
                
        if noise and ndata is None:
            assert core.is_passive(sdata), "Noise data not found for active component."
            # get the noise correlation matrix
            ndata = core.get_passive_ndata(sdata)

        # cast data as labeled arrays
        ret_data = dict()
        n_ports = sdata.shape[-1]
        port_a = np.arange(1, n_ports +1)

        if not isinstance(sdata, ldarray):
            sdata = ldarray(sdata, coords=dict(frequency=frequency, b=port_a, a=port_a))

        ret_data["s"] = sdata

        if noise:
            ret_data["n"] = ldarray(ndata, coords=dict(frequency=frequency, b=port_a, a=port_a))

        return ret_data


class Component_SnP(Component):
    """
    Component defined from a touchstone file (.snp)
    """

    def __init__(
        self, 
        file: Union[str, dict], 
        shunt: bool = False, 
        passive: bool = False, 
        state: dict = dict()
    ):
        """
        Parameters
        ----------
        file : str | dict
            dictionary of file paths. Keys are used as variable names that can be used to select files in ``set_state``,
            values are file paths. Supports a single string value which is assigned as the "default" key. 
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.
        passive : bool, default: True
            if True, ``evaluate`` calls ``evaluate_sdata`` instead of ``evaluate_data`` and noise correlation
            matrix is computed passively.
        state : dict, optional
            dictionary of state variables specific to the sub-class (i.e. line width, switch state etc...).
            The state values can be read with the ``state`` property and changed later with `set_state()`. Attempting
            to set the state of variables that were not included in the initial dictionary will raise an error.
        """
        if isinstance(file, (str, Path)):
            self.file = dict(default=Path(file))

        # convert any string paths to Path
        elif isinstance(file, dict):
            self.file = {**{k: Path(v) for k,v in file.items()}}

        # check that all files have the same number of ports
        n_ports = np.array([utils.n_ports_from_snp(v) for v in self.file.values()])
        n_port = n_ports[0]

        # check that all files exist
        if any([not v.exists() or not v.is_file() for v in self.file.values()]):
            raise ValueError(f"File does not exist: {list(self.file.values())}")

        if not np.all(n_port == n_ports):
            raise ValueError("Files must all have the same number of ports.")
        
        self._comments = []
        self._data_cache = dict()

        if not len(state):
            state = dict(file=tuple(self.file.keys())[0])

        if shunt and not np.all(n_port == 2):
            raise ValueError("Only 2-port components can be shunted.")
        
        # component might be passive, but don't assume that it is by default, check for passivity in evaluate_data
        super().__init__(shunt=shunt, passive=passive, n_ports=n_port, state=state)

    @property
    def frequency(self):
        """
        Return the frequency values of the touchstone file for the current state.
        """
        sdata, _ = self._get_file_data()
        return sdata.coords["frequency"]

    def _get_file_data(self) -> Tuple[ldarray, ldarray]:
        """
        Read the touchstone data for the current state.

        Returns
        -------
        sdata : ldarray
            labeled numpy array of s-matrix data with dimensions (frequency, b, a).
        np_data : 
            labeled numpy array of noise parameter data (if found in the touchstone file) with dimensions
            (frequency, noise_param). "nf_min", "gamma_opt", "rn"
        """

        # reading touchstones is slow, only read file if state hasn't been hit before
        if self.state["file"] not in self._data_cache.keys():
            filepath = self.file[self.state["file"]]
            sdata, np_data, self._comments = touchstone.read_snp(filepath)
            # store into cache
            self._data_cache[self.state["file"]] = [sdata, np_data]

        # otherwise load from cache
        else:
            sdata, np_data = self._data_cache[self.state["file"]]

        return sdata, np_data

    def equals(self, other):
        """
        Returns True if the s-matrix data from other is equivalent to this object.

        Parameters
        ----------
        other : Component
            Component object to check equality against this object.
        """
        if not super().equals(other):
            return False
        
        # even if the file keys are the same, they might point to different paths
        return self.file[self.state["file"]] == other.file[other.state["file"]]
        

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        """
        Returns s-matrix matrix of the component.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data over, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        """
        # get touchstone file data
        sdata, np_data = self._get_file_data()

        # interpolate the s-parameters at the desired frequency points
        if frequency is not None:
            if len(sdata.coords["frequency"]) > 1:
                sp1 = CubicSpline(sdata.coords["frequency"], sdata, axis=-3)
                sdata = sp1(frequency)
            else:
                sdata = sdata.sel(frequency=frequency)

        return sdata
        

    def evaluate_data(self, frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns s-matrix and noise correlation matrix of the component.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data over, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        ndata : np.ndarray
            MxNxN noise correlation matrix where M is the number of frequency values and N is the number of ports. 
        """
        sdata = self.evaluate_sdata(frequency)

        if frequency is None:
            frequency = data.coords["frequency"]

        # get noise parameters from cache
        _, np_data = self._data_cache[self.state["file"]]

        if np_data is not None:
            # interpolate the noise parameters at the sdata frequency points, allow extrapolation
            sp2 = CubicSpline(np_data.coords["frequency"], np_data, axis=-2)
            np_data = sp2(frequency)

            ndata = core.noise_params_to_ndata(np_data, sdata)

        elif core.is_passive(sdata):
            ndata = core.get_passive_ndata(sdata)

        else:
            raise RuntimeError("Noise parameters not available for active component.")
        
        return sdata, ndata


class Component_Data(Component):
    """
    Component defined from a user-defined data.
    """

    def __init__(self, data: ldarray, state: dict = dict()):

        self._sdata = data
        nports = data.shape[-2]
        super().__init__(nports, passive=False, state=state)

    @property
    def frequency(self):
        """
        Return the frequency values of the touchstone file for the current state.
        """
        return self._sdata.coords["frequency"]
    
    def equals(self, other):
        """
        Returns True if the s-matrix data from other is equivalent to this object.

        Parameters
        ----------
        other : Component
            Component object to check equality against
        """
        return False
    
    def evaluate_sdata(self, frequency) -> np.ndarray:
        """
        Returns s-matrix matrix of the component.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data over, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        """
        sdata = self._sdata.sel(**self.state)

        # interpolate the s-parameters at the desired frequency points
        if len(sdata.coords["frequency"]) > 1:
            sp1 = CubicSpline(sdata.coords["frequency"], sdata, axis=-3)
            sdata = sp1(frequency)
        else:
            sdata = sdata.sel(frequency=frequency)

        return sdata