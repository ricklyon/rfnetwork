import numpy as np
from np_struct import ldarray
from pathlib import Path
from scipy.interpolate import CubicSpline
from copy import deepcopy as dcopy

from abc import abstractmethod
from typing import Tuple

from . import touchstone, const, utils
from . core import core
from . import plots

class Component(object):
    
    def __init__(self, shunt: bool = False, passive: bool = True, pnum: int = 2):
        """
        Parameters
        ----------
        shunt : bool, default: False
            if True, port 2 of the component is grounded. Component must be a passive 2-port device.
        passive : bool, default: True
            if True, ``evaluate`` calls ``evaluate_sdata`` instead of ``evaluate_data`` and noise correlation
            matrix is computed passively.
        """
        self._shunt = shunt
        self._passive = passive
        self._pnum = pnum

    @property
    def pnum(self):
        return self._pnum
    
    def __or__(self, other):
        """ Allows port to be indexed with the syntax: block|2 """
        return (self, int(other))
    
    def equals(self, other):
        """
        Returns True if other is equivalent to this object. Subclasses must override this method with their own
        checks and call ``super().equals(other)``.
        """
        self_class = self.__class__.__name__
        other_class = other.__class__.__name__
        return (self_class == other_class) and (self._shunt == other._shunt) and (self._passive == other._passive)
    
    @abstractmethod
    def evaluate_data(self, frequency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns s-matrix and noise correlation matrix
        """
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        """
        Returns s-matrix data of the component.
        """
        raise NotImplementedError()
    
    def plot(self, frequency = None, *paths, fmt: str = "db", **kwargs):
        """
        
        """
        data = self.evaluate(frequency, noise=fmt in ["nf"])
        ndata = data["n"] if fmt in ["nf"] else None
        return plots.plot(data["s"], *paths, fmt=fmt, ndata=ndata, **kwargs)
    
    def plot_stability_circles(self, f0: float, **kwargs):
        """
        
        """
        data = self.evaluate(f0)["s"]
        return plots.plot_stability_circles(data, f0, **kwargs)
    
    def evaluate(self, frequency: np.ndarray = None, noise: bool = False) -> dict:
        """
        Returns a dictionary with keys: "n" (noise correlation matrix) and "s" (s-matrix). 
        """

        frequency = np.atleast_1d(frequency) if frequency is not None else None
        
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
            _, shunt_data = core.connect(dict(s=sdata), dict(s=short), (2, 1), noise=False)

            # connect port 1 to a junction so the open port can be connected to 2 other components
            node = core.junction_sdata(frequency, 3)
            _, shunt_data = core.connect(shunt_data, dict(s=node), (1, 1), noise=False)
            sdata = shunt_data["s"]

            # invalidate noise data, shunting active components is not supported.
            ndata = None
                
        if noise and ndata is None:
            assert core.is_passive(sdata), "Noise data not found for active component."
            # get the noise correlation matrix
            ndata = core.get_passive_ndata(sdata)
            
        # attempt to use the frequency vector already in the sdata if frequency is not provided.
        if frequency is None:
            if hasattr(sdata, "coords") and sdata.coords is not None:
                frequency = sdata.coords["frequency"]
            else:
                raise RuntimeError("Frequency data not found in sdata.")

        # cast data as labeled arrays
        ret_data = dict()
        pnum = sdata.shape[-1]
        port_a = np.arange(1, pnum +1)

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

    def __init__(self, file: str | dict, state: str = None, shunt: bool = False, passive: bool = False):

        if isinstance(file, (str, Path)):
            self.file = dict(default=Path(file))

        elif isinstance(file, dict):
            self.file = {**{k: Path(v) for k,v in file.items()}}

        # check that all files have the same number of ports
        pnums = np.array([utils.get_pnum_from_snp(v) for v in self.file.values()])
        pnum = pnums[0]

        # check that all files exist
        if any([not v.exists() or not v.is_file() for v in self.file.values()]):
            raise ValueError(f"File does not exist: {list(self.file.values())}")

        if not np.all(pnums == pnum):
            raise ValueError("Files must all have the same number of ports.")
        
        self._state = state if state is not None else tuple(self.file.keys())[0]
        self._comments = []
        self._data_cache = dict()

        if shunt and not np.all(pnums == 2):
            raise ValueError("Only 2-port components can be shunted.")
        
        # component might be passive, but don't assume that it is by default, check for passivity in evaluate_data
        super().__init__(shunt=shunt, passive=passive, pnum=pnum)

    def __call__(self, state: str = None):
        # simple syntax for duplicating components in Network declarations, optionally with a different state.
        # for example: 
        # c1 = ComponentFromFile(<path>)
        # c2 = sw()
        nobj = dcopy(self)
        nobj.set_state(state)

        return nobj
    
    def equals(self, other):
        
        if not super().equals(other):
            return False
        
        files_match = self.file[self.state] == other.file[other.state]
        return self.state == other.state and files_match
    
    def evaluate_sdata(self, frequency: np.ndarray = None) -> np.ndarray:

        # reading touchstones is slow, only read file if state hasn't been hit before
        if self.state not in self._data_cache.keys():
            filepath = self.file[self.state]
            sdata, np_data, self._comments = touchstone.read_snp(filepath)
            # store into cache
            self._data_cache[self.state] = [sdata, np_data]

        # otherwise load from cache
        else:
            sdata, np_data = self._data_cache[self.state]

        # interpolate the s-parameters at the desired frequency points
        if frequency is not None:
            if len(sdata.coords["frequency"]) > 1:
                sp1 = CubicSpline(sdata.coords["frequency"], sdata, axis=-3)
                sdata = sp1(frequency)
            else:
                sdata = sdata.sel(frequency=frequency)

        return sdata
        

    def evaluate_data(self, frequency: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:

        sdata = self.evaluate_sdata(frequency)

        if frequency is None:
            frequency = data.coords["frequency"]

        # get noise parameters from cache
        _, np_data = self._data_cache[self.state]

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

    def set_state(self, state: str = None):
        if state == self._state or state is None:
            return

        if state not in self.file.keys():
            raise KeyError(f"Unrecognized state: {state}")
        
        self._state = state

    @property
    def state(self):
        return self._state

class Component_Data(Component):
    """
    Component defined from a user-defined or imported data.
    """

    def __init__(self, data: ldarray, shunt: bool = False, passive: bool = False):

        self._sdata = data
        pnum = data.shape[-2]
        super().__init__(shunt=shunt, passive=passive, pnum=pnum)

    def equals(self, other):
        return False

    def __call__(self):
        # simple syntax for duplicating components in Network declarations
        return dcopy(self)
    
    def evaluate_sdata(self, frequency) -> np.ndarray:
        
        sdata = self._sdata

        # interpolate the s-parameters at the desired frequency points
        if len(sdata.coords["frequency"]) > 1:
            sp1 = CubicSpline(sdata.coords["frequency"], sdata, axis=-3)
            sdata = sp1(frequency)
        else:
            sdata = sdata.sel(frequency=frequency)

        return sdata