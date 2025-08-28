
import numpy as np
from typing import Union, List, Tuple

from .component import Component
from .network import DynamicNetwork, Network
from .core.units import conv, const
from .core import core
from . import utils
from np_struct import ldarray
from copy import deepcopy as dcopy
from scipy.interpolate import CubicSpline


__all__ = (
    "Line",
    "MSLine",
    "Stripline",
    "Hybrid180",
    "Hybrid90"
    "LumpedElement",
    "Resistor",
    "Capacitor",
    "Inductor",
    "Short",
    "Open",
    "Load",
    "Attenuator",
    "PiAttenuator",
    "LowPassFilter",
    "HighPassFilter",
    "BandPassFilter",
    "BandStopFilter"
)


class Load(Component):
    """
    1-port load with variable impedance.
    """

    def __init__(self, value: float = 50):
        """
        Parameters
        ----------
        value: float, default: 50
            real or complex impedance [ohms]. Defaults to 50 ohms
        """
        super().__init__(n_ports=1, state=dict(value=value))
        assert value > 0, "Load must be passive"
    
    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return np.full((len(frequency), 1, 1), conv.gamma_z(self.value), dtype="complex128")

class Open(Component):
    """
    1-port open-circuit load
    """
    def __init__(self):
        super().__init__(n_ports=1)

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return np.full((len(frequency), 1, 1), 1, dtype="complex128")


class Short(Component):
    """
    1-port short-circuit load
    """
    def __init__(self):
        super().__init__(n_ports=1)

    def evaluate_sdata(self, frequency: np.ndarray):
        return np.full((len(frequency), 1, 1), -1, dtype="complex128")


class Hybrid90(Component):
    """
    Ideal directional coupler with 90 degree phase delay between the thru and coupled paths.

    - Port 1: Input
    - Port 2: Thru
    - Port 3: Coupled
    - Port 4: Isolated
    """

    def __init__(self):
        super().__init__(n_ports=4)

    def evaluate_sdata(self, frequency: np.ndarray):
        sdata = (-1 / np.sqrt(2)) * np.array([
            [0, 1j, 1, 0],
            [1j, 0, 0, 1],
            [1, 0, 0, 1j],
            [0, 1, 1j, 0]
        ], dtype="complex128")
        return np.broadcast_to(sdata[None], (len(frequency), 4, 4)).copy()


class Hybrid180(Component):
    """
    Ideal directional coupler with 180 degree phase delay between the thru and coupled paths.

    - Port 1: Input
    - Port 2: Thru
    - Port 3: Coupled
    - Port 4: Isolated
    """

    def __init__(self):
        super().__init__(n_ports=4)

    def evaluate_sdata(self, frequency: np.ndarray):
        sdata = (-1j / np.sqrt(2)) * np.array([
            [0, 1, -1, 0],
            [1, 0, 0, -1],
            [-1, 0, 0, 1],
            [0, -1, 1, 0]
        ], dtype="complex128")

        return np.broadcast_to(sdata[None], (len(frequency), 4, 4)).copy()


class LumpedElement(Component):
    """
    Base Class for 2-port series and shunt lumped elements
    """
    MULTIPLIER = 1

    def __init__(self, value: float = 0, shunt: bool = False):
        """
        Parameters
        ----------
        value : float
            Resistance (Ohms), capacitance (F), or inductance (H) of lumped component.
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.
        """
        super().__init__(shunt=shunt, state=dict(value=value))

    def evaluate_sdata_from_z(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the s-matrix given the series impedance of the lumped element.
        """
        # initialize new sdata matrix
        shape = (len(z), 2, 2)
        sdata = np.zeros(shape, dtype="complex128")

        # generate new s-matrix data for series element.
        z0 = 50
        s11 = z / (2 * z0 + z)
        s21 = (1 + s11) * (z0 / (z + z0))

        # component is reciprocal and s-matrix is symmetric
        sdata[:, 0, 0] = s11
        sdata[:, 1, 1] = s11
        sdata[:, 0, 1] = s21
        sdata[:, 1, 0] = s21

        return sdata
             

class Resistor(LumpedElement):
    """
    Ideal Resistor network element.
    """

    def evaluate_sdata(self, frequency: np.ndarray):

        z = np.full(frequency.shape, (self.state["value"] * self.MULTIPLIER))
        return super().evaluate_sdata_from_z(z)


class Capacitor(LumpedElement):
    """
    Ideal Capacitor network element.
    """

    def evaluate_sdata(self, frequency: np.ndarray):

        # z = 1/jwC
        z = 1 / (1j * 2 * np.pi * frequency * (self.state["value"] * self.MULTIPLIER))
        return super().evaluate_sdata_from_z(z)


class Inductor(LumpedElement):
    """
    Ideal Inductor network element.
    """
    
    def evaluate_sdata(self, frequency: np.ndarray):

        # z = jwL
        z = 1j * 2 * np.pi * frequency * (self.state["value"] * self.MULTIPLIER)
        return super().evaluate_sdata_from_z(z)


class Capacitor_pF(Capacitor):
    """
    Ideal Capacitor network element, in pF units.
    """

    MULTIPLIER = 1e-12
    

class Inductor_nH(LumpedElement):
    """
    Ideal Inductor network element, in nH units.
    """
    MULTIPLIER = 1e-9


class Attenuator(Component):
    def __init__(self, attenuation_db: float):
        """
        Parameters
        ----------
        attenuation_db: float
            attenuation in dB
        """
        super().__init__(passive=True, state=dict(value=-np.abs(attenuation_db)))

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:

        s21 = core.conv.lin_db20(-np.abs(self.state["value"]))
        sdata = np.array([[1e-6, s21], [s21, 1e-6]], dtype="complex128")
        return np.broadcast_to(sdata, (len(frequency), 2, 2)).copy()




class PiAttenuator(Network):
    """
    Resistive attenuator in pi configuration
    """
    r1 = Resistor(shunt=True)
    r2 = Resistor()
    r3 = Resistor(shunt=True)

    cascades = [("P1", r1, r2, r3, "P2")]

    def __init__(self, attenuation_db: float, r0: float = 50):
        """
        Parameters
        ----------
        attenuation_db: float
            attenuation in dB
        """
        A = conv.lin_db20(-np.abs(attenuation_db))
        state = dict(
            r1=r0 * (A + 1) / (1 - A), 
            r2=r0 * (1 - A**2) / (2 * A),
            r3=r0 * (A + 1) / (1 - A)
        )
        super().__init__(passive=True, state=state)



class LC_Series(Network):
    """
    Inductor and Capacitor in series.
    """
    l1 = Inductor()
    c2 = Capacitor()
    cascades = [["P1"] + [l1, c2] + ["P2"]]

    def __init__(self, L: float, C: float, shunt: bool = False):
        """
        Parameters
        ----------
        L : float
            inductor value [H]
        C : float
            capacitor value [F]
        """
        state=dict(l1=L, c2=C)
        super().__init__(shunt=shunt, state=state, passive=True)


class LC_Parallel(Network):
    """
    Inductor and Capacitor in parallel.
    """

    l1 = Inductor()
    c2 = Capacitor()
    nodes = [
        ("P1", l1|1, c2|1),
        ("P2", l1|2, c2|2),
    ]

    def __init__(self, L: float, C: float, shunt: bool = False):
        """
        Parameters
        ----------
        L : float
            inductor value [H]
        C : float
            capacitor value [F]
        """
        state=dict(l1=L, c2=C)
        super().__init__(shunt=shunt, state=state, passive=True)


class Line(Component):
    """
    Base class for generic transmission lines. 
    """

    def __init__(
        self, 
        z0: Union[float, List] = 50, 
        er: Union[float, List] = 1, 
        loss: Union[float, List] = 0, 
        frequency: List = 0, 
        length: float = None, 
        shunt: bool = False, 
        state: dict = dict()
    ):
        """
        Parameters
        ----------
        
        z0 : float | list, default: 50
            Characteristic impedance of line.
        er : float | list, default: 1
            Dielectric constant of substrate.
        loss : float | list, default: 0
            Total loss, (dielectric + conductor) losses, in dB per inch.
        frequency : list, optional
            Array of frequencies in Hz for z0, er, and loss if they are provided as lists.
        length : float, optional
            Length of line segment in inches.
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.

        Examples
        --------
        >>> import rfnetwork as rfn
        >>> ln = rfn.elements.Line( 
        ...    z0=55,
        ...    er=[3.2, 3.5],
        ...    loss=[0.5, 0.7],
        ...    frequency=[10e9, 12e9],
        ...    length=1.5,
        ... )
        >>> ln.state
        {'length': 1.5, 'z0': 55, 'er': [3.2, 3.5], 'loss': [0.5, 0.7]}

        If setting the length with the electrical length, keep in mind that it is relative to a wavelength in the
        medium, not in free space. 
        
        >>> ln1 = ln(360, f0=10e9)
        >>> ln1.state["length"]
        0.6597...

        Whereas a wavelength in free space is,
        
        >>> rfn.const.c0_in / 10e9
        1.1802...

        """

        self._frequency = np.atleast_1d(frequency)
        super().__init__(shunt=shunt, state=dict(length=length, z0=z0, er=er, loss=loss, **state))
    
    def __call__(self, length: float = None, f0: float = None):
        """
        Create a new instance with the specified line length.

        Parameters
        ----------
        length : float
            length of line instance in inches. If f0 is given, interpreted as electrical length in degrees.
        f0 : float, optional
            Frequency for the electrical length, if length is given in degrees.
     
        """
        # assign electrical length if given
        if f0 is not None:
            # get er at center frequency
            er = self.get_properties(f0).sel(value="er").item()
            # wavelength in inches at center frequency
            vp_in = const.c0_in / np.sqrt(er)
            lmbda_in = vp_in / f0
            # electrical length is e_len = (beta * len) = (2*np.pi)len/lmbda.
            length = (np.deg2rad(length) * lmbda_in) / (2 * np.pi)

        # create new instance with length
        lineseg = dcopy(self)

        if length is not None:
            lineseg.set_state(length=float(length))

        return lineseg

    def get_wavelength(self, frequency, units="in") -> ldarray:
        """
        Returns the propagation wavelength [inches] of the medium at the specified frequency [Hz].

        Parameters
        ----------
        frequency: float | np.ndarray
            Frequencies [Hz] at which to compute wavelength.
        units: {"in", "m"}, default: "in"
            units of the returned wavelength.

        Returns
        -------
        ldarray:
            Mx1 array where M is the number of frequencies.

        Examples
        --------
        >>> import rfnetwork as rfn
        >>> ln_er1 = rfn.elements.Line(er=1)
        >>> ln_er3 = rfn.elements.Line(er=3)

        >>> ln_er1.get_wavelength(10e9)
        ldarray([1.18028591])...
    
        >>> ln_er3.get_wavelength(10e9)
        ldarray([0.68143839])...

        """

        # generate the properties at the frequency first so the er is correct, even if we haven't
        # added to a network yet and don't have a frequency list.
        properties = self.get_properties(frequency)

        # get the list of dielectric constants
        er = properties.sel(value="er")

        # calculate lambda in inches
        c0 = const.c0_in if units == "in" else const.c0
        lmbda = c0 / (frequency * np.sqrt(er))

        return lmbda

    def get_delay(self, frequency: Union[float, np.ndarray]) -> Tuple:
        """
        Returns the phase delay [deg] and time delay [ps] at the specified frequency.

        Parameters
        ----------
        frequency: float | np.ndarray
            Frequencies [Hz] at which to compute delay.

        Returns
        -------
        phase_delay : ldarray
            phase delay in degrees
        time_delay : ldarray
            time delay in degrees
        """
        if self.state["length"] is None:
            raise RuntimeError("Line must have length assigned to compute delay")

        lmbda = self.get_wavelength(frequency)

        pdelay = np.rad2deg((2 * np.pi * self.state["length"]) / lmbda)

        vp = lmbda * frequency
        tdelay = self.state["length"] / vp

        return pdelay, tdelay * 1e12

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        """
        Computes the lossy s-matrix of a transmission line.
        """

        if self.state["length"] is None:
            raise RuntimeError("Line length must be specified before evaluating sdata")
        
        frequency = np.atleast_1d(frequency)

        properties = self.get_properties(frequency)

        sdata = np.zeros(shape=(len(frequency), 2, 2), dtype="complex128")

        length = self.state["length"]
        z0 = properties.sel(value="z0")
        er = properties.sel(value="er")
        db_loss = properties.sel(value="loss")

        zref = 50

        lmbda_0 = const.c0_in / frequency
        lmbda = lmbda_0 / np.sqrt(er)

        alpha = db_loss / (20 * np.log10(np.e))
        beta = (2 * np.pi) / lmbda
        propg = alpha + 1j * beta
        gmmaL = (zref - z0) / (zref + z0)

        Zin = z0 * ((1 + gmmaL * np.e ** (-2 * propg * length)) / (1 - gmmaL * np.e ** (-2 * propg * length)))
        gmmaS = (Zin - zref) / (Zin + zref)

        s21 = (2 * zref * (1 + gmmaS)) / ((zref + z0) * (np.e ** (propg * length) + gmmaL * np.e ** (-propg * length)))
        s11 = gmmaS

        sdata[:, 0, 0] = s11
        sdata[:, 1, 1] = s11
        sdata[:, 0, 1] = s21
        sdata[:, 1, 0] = s21

        return sdata

    def get_properties(self, frequency: float | np.ndarray) -> ldarray:
        """
        Returns characteristic impedance and dielectric constant over frequency.

        Parameters
        ----------
        frequency : float | np.ndarray
            frequency value(s) to evaluate properties at.

        Returns
        -------
        ldarray
            Mx3 array, where M is the number of frequency points. Columns are z0, er, and loss per inch [dB].

        """

        frequency = np.atleast_1d(frequency)
        # create a labeled array with rows as frequency and columns as data type (zo, loss, etc...).
        # parameters will be interpolated to match the freq passed into evaluate
        coords = dict(frequency=self._frequency, value=["z0", "er", "loss"])

        # initialize properties array.
        properties = ldarray(coords=coords)

        # populate each column, if values are single numbers numpy will broadcast the value to the correct length.
        for i, p in enumerate([self.state["z0"], self.state["er"], self.state["loss"]]):
            # ensure all properties are positive numbers (no negative permittivity materials or amplifier lines)
            properties[..., i] = np.abs(p)

        if len(self._frequency) < 2:
            # broadcast single-point properties across the provided frequency vector
            shape = (len(frequency), len(properties.coords["value"]))
            properties = np.broadcast_to(properties, shape).copy()
        else:
            # interpolate the line properties over frequency
            sp1 = CubicSpline(self._frequency, properties, axis=0)
            properties = sp1(frequency)

        # cast result as ldarray
        ncoords = dict(frequency=frequency, value=["z0", "er", "loss"])
        return ldarray(properties, coords=ncoords)



class MSLine(Line):
    """
    Microstrip Line
    """

    def __init__(
        self, 
        h: float, 
        er: Union[float, List], 
        w: float = None, 
        z0: float = 0, 
        t: float = 0.0013, 
        df: Union[float, List] = 0, 
        loss: Union[float, List] = 0,
        frequency: List = 0,
        length: float = None,
        shunt: bool = False
    ):
        """
        Parameters
        ----------
        h : float
            Substrate thickness in inches
        er : float | list
            Dielectric constant of substrate.
        w : float, optional
            Width of microstrip trace in inches. Specify w or z0, not both.
        z0 : float, optional
            Target characteristic impedance of line. Width will be set automatically if provided.
        t : float, optional
            Copper thickness in inches, defaults to 1oz copper (1.3 mils). Does not affect loss.
        df : float | list, optional
            Dissipation factor of substrate material, or dielectric loss tangent.
        loss : float | list, optional
            Total loss, (dielectric + conductor) losses, in dB per inch. Ignored if df is provided.
        frequency : list, optional
            Array of frequencies in Hz for er, df, and loss if they are provided as lists.
        length : float, optional
            Length of line segment in inches.
        shunt : bool, default: False
            if True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.

        Examples
        --------

        To create multiple instances of this line with different lengths, 

        >>> import rfnetwork as rfn
        >>> ms = rfn.elements.MSLine(
        ...    w=0.020,
        ...    h=0.010,
        ...    t=0.0013,
        ...    er=3.0,
        ...    loss=0.5,
        ... )

        >>> # create 0.5in segment
        >>> ms1 = ms(0.5)  
        >>> # create 1.0in segment
        >>> ms2 = ms(1.0)  
        >>> ms1.state
        {'length': 0.5, ...

        If f0 is provided, the length is interpreted as the electrical length in degrees.

        >>> ms1 = ms(90, f0=10e9)  # create a 0.5in segment
        >>> ms1.state["length"]  # state holds the length in degrees
        0.2950...

        """
        # get initial guess for width if z0 was given but not width
        if z0 and (w is None):
            w = utils.ms_z_to_width(z0, er, h)

        state = dict(w=w, h=h, t=t, df=df)
        super().__init__(
            z0=z0, er=er, loss=loss, frequency=frequency, length=length, shunt=shunt, state=state
        )

    def __call__(self, length: float = None, w: float = None, f0: float = None):
        """
        Create a new instance with the specified line length, and optionally a new width.

        Parameters
        ----------
        length : float
            length of line instance in inches. If f0 is given, interpreted as electrical length in degrees.
        w : float, optional
            line width in inches
        f0 : float, optional
            Frequency for the electrical length, if length is given in degrees.
     
        """
        lineseg = super().__call__(length=length, f0=f0)

        if w is not None:
            lineseg.set_state(w = float(w))

        return lineseg
    
    def get_properties(self, frequency: Union[float, np.ndarray]) -> ldarray:
        """
        Returns characteristic impedance and effective dielectric constant over frequency.
        Equations are the same as those Roger's MWI tool [1]. See page 149 of [2] for loss equations.

        Parameters
        ----------
        frequency : float | np.ndarray
            frequency value(s) to evaluate properties at.

        Returns
        -------
        ldarray
            Mx3 array, where M is the number of frequency points. Columns are z0, effective er, and loss per inch [dB].

        References
        ----------
        [1] Design-Data-for-Microstrip-Transmission-Lines-on-RT-duroid-Laminates, Rogers Corporation, 2013
        [2] Pozar, David M. Microwave Engineering. 4th ed., Wiley, 2011.

        """
        frequency = np.atleast_1d(frequency)
        f_ghz = frequency / 1e9

        # get the interpolated list of line properties
        properties = super().get_properties(frequency)

        er = properties[{"value": "er"}]
        # B is in mm
        B = conv.mm_mil(self.state["h"]* 1e3)
        T = self.state["t"] / self.state["h"]
        U = self.state["w"]/ self.state["h"]
        eta = 376.73

        U1 = U + (T * np.log(1 + ((4 * np.e) / T) * (np.tanh((6.517 * U) ** (1 / 2))) ** 2)) / np.pi
        Ur = U + ((U1 - U) * (1 + 1 / (np.cosh((er - 1) ** 0.5)))) / 2

        Au = 1 + (np.log((Ur**4 + (Ur**2 / 2704)) / (Ur**4 + 0.432))) / 49 + np.log((Ur / 18.1) ** 3 + 1) / 18.7
        Ber = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053

        Y = (er + 1) / 2 + ((er - 1) / 2) * (1 + (10 / Ur)) ** (-Au * Ber)

        def Z01(x):
            return (
                eta
                * np.log((6 + (2 * np.pi - 6) * np.e ** (-((30.666 / x) ** 0.7528))) / x + ((4 / (x**2)) + 1) ** 0.5)
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

        # update the properties values
        properties[{"value": "z0"}] = Z0_f
        properties[{"value": "er"}] = eff_f
        
        # generate dielectric loss from the loss tangent, as well as conductor losses
        if np.any(np.array(self.state["df"]) > 0):

            # interpolate loss tangent
            if len(self._frequency) < 2:
                # broadcast single-point properties across the provided frequency vector
                df = np.broadcast_to(self.state["df"], len(frequency))
            else:
                # interpolate the line properties over frequency
                sp1 = CubicSpline(self._frequency, self.state["df"])
                df = sp1(frequency)

            lmbda0_in = const.c0_in / (f_ghz * 1e9)
            k0 = (2 * np.pi) / lmbda0_in
            # calculate dielectric loss per inch
            ad_in = ((k0 * er) * (eff_f - 1) * df) / (2 * np.sqrt(eff_f) * (er - 1))

            # calculate copper resistivity using conductivity in seimens per meter
            Rs = np.sqrt((2 * np.pi * f_ghz * 1e9 * const.u0) / (2 * const.cu_sigma))

            # calculate losses due to copper conductor in np/m.
            # use width in meters
            w_m = conv.m_in(self.state["w"])
            a_c = Rs / (Z0_f * w_m)
            # convert loss to per inch
            ac_in = a_c / 39.3701

            # convert total loss to positive valued loss in db per inch
            properties[{"value": "loss"}] = -conv.db20_lin(np.exp(-(ad_in + ac_in)))

        return properties


class Stripline(Line):
    """
    Stripline with balanced substrate.
    """

    def __init__(
        self, 
        w: float, 
        b: float, 
        er: Union[float, List],
        t: float = 0.0013, 
        df: Union[float, List] = 0, 
        loss: Union[float, List] = 0,
        frequency: List = 0,
        length: float = None,
        shunt: bool = False
    ):
        """
        Parameters
        ----------
        w : float
            width of stripline trace in inches
        b : float
            substrate thickness in inches (with line in the center).
        t : float, optional
            copper thickness in inches, defaults to 1oz copper (1.3 mils). Does not affect loss.
        df : float | list, optional
            Dissipation factor of substrate material, or dielectric loss tangent.
        loss : float | list, optional
            Total loss, (dielectric + conductor) losses, in dB per inch. Ignored if df is provided.
        frequency : list, optional
            Array of frequencies in Hz for er, df, and loss if they are provided as lists.
        length : float, optional
            Length of line segment in inches.
        shunt : bool, default: False
            if True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.

        Examples
        --------

        >>> import rfnetwork as rfn
        >>> sl = rfn.elements.Stripline(
        ...    w=0.008,
        ...    b=0.010,
        ...    er=3.0,
        ...    loss=0.5,
        ...    length=0.5
        ... )

        Evaluate s21 to get the loss of the line,

        >>> s21 = sl.evaluate(10e9)["s"].sel(b=2, a=1)
        >>> rfn.conv.db20_lin(s21)
        ldarray([-0.32187157])...

        """
        
        state = dict(w=w, b=b, t=t, df=df)
        super().__init__(z0=0, er=er, loss=loss, frequency=frequency, length=length, state=state, shunt=shunt)

    def get_properties(self, frequency: float | np.ndarray) -> ldarray:
        """
        Returns characteristic impedance and dielectric constant over frequency.
        See section 3.8 in [1].

        Parameters
        ----------
        frequency : float | np.ndarray
            frequency value(s) to evaluate properties at.

        Returns
        -------
        ldarray
            Mx3 array, where M is the number of frequency points. Columns are z0, er, and loss per inch [dB].
        
        References
        ----------
        [1] Pozar, David M. Microwave Engineering. 4th ed., Wiley, 2011.

        """

        # use properties freq if a single value frequency is not provided.

        frequency = np.atleast_1d(frequency)
        f_ghz = frequency / 1e9

        # get the interpolated list of line properties
        properties = super().get_properties(frequency)

        er = properties[{"value": "er"}]

        # effective width
        w_over_b = (self.state["w"] / self.state["b"]) 
        we_b = w_over_b if w_over_b >= 0.35 else w_over_b - (0.35 - w_over_b)**2

        # characteristic impedance
        n = np.sqrt(er)
        z0 = ((30*np.pi) / n) * (1 / (we_b + 0.441))

        # update the properties values
        properties[{"value": "z0"}] = z0

        # generate dielectric loss from the loss tangent, as well as conductor losses
        if np.any(np.array(self.state["df"]) > 0):

            # interpolate loss tangent
            if len(self._frequency) < 2:
                # broadcast single-point properties across the provided frequency vector
                df = np.broadcast_to(self.state["df"], len(frequency))
            else:
                # interpolate the line properties over frequency
                sp1 = CubicSpline(self._frequency, self.state["df"])
                df = sp1(frequency)

            # calculate copper resistivity using conductivity in seimens per meter
            Rs = np.sqrt((2 * np.pi * f_ghz * 1e9 * const.u0) / (2 * const.cu_sigma))

            # 'A' parameter. We can stay in inches here since everything is a ratio
            w = self.state["w"]
            b = self.state["b"]
            t = self.state["t"]
            A = 1 + ((2*w)/(b-t)) + (1/np.pi)*((b+t)/(b-t)) * np.log((2*b -t)/t)

            # 'B' parameter
            B = 1 + (b / (0.5*w + 0.7*t)) * (0.5 + ((0.414*t)/w) + (1/(2*np.pi))*np.log((4*np.pi*w)/t))

            # conductor attenuation constant in inches^-1
            a_c_gt = (0.16 * Rs * B) / (z0*b)
            a_c_lt = (2.7e-3 * Rs * er * z0 * A) / (30*np.pi*(b-t))

            a_c = np.where(n*z0 < 120, a_c_lt, a_c_gt)

            # equation 3.30 in pozar
            k_in = (2*np.pi*f_ghz*1e9) * (n / const.c0_in)
            a_d = (k_in * df) /2

            # convert total loss to positive valued loss in db per inch
            properties[{"value": "loss"}] = -conv.db20_lin(np.exp(-(a_d+ a_c)))

        return properties


class LowPassFilter(DynamicNetwork):
    """ 
    Lumped component low-pass filter 
    """
    def __init__(self, fc: float, n: int, r0: float = 50, ripple: float = 0.5):
        """
        Parameters
        ----------
        fc : float
            3 dB cutoff frequency [Hz]
        n : int
            Filter order (1-10). Odd n will be matched to 50 ohms, while even n requires an impedance match.
        r0 : float, default: 50
            Port impedance. Default is 50 ohms.
        ripple : float, default: 0.5
            Pass-band ripple in dB. Supported values are 0.5 and 3.0 dB.
        """
        # get n normalized element values, drop the last value which is the resistive load.
        proto_vals = utils.lp_filter_prototype(n, ripple=ripple)[:-1]

        components = dict()
        wc = 2 * np.pi * fc

        # build components for each element, scaling the impedance and frequency from the normalized protoype values.
        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0: # start with series inductor
                components[f"L{i+1}"] = Inductor((g_k * r0) / wc)
            else:
                components[f"C{i+1}"] = Capacitor(g_k / (r0 * wc), shunt=True)
        
        cascades = [["P1"] + list(components.values()) + ["P2"]]

        super().__init__(components, cascades=cascades, passive=True)


class HighPassFilter(DynamicNetwork):
    """ 
    Lumped component high-pass filter 
    """
    
    def __init__(self, fc: float, n: int, r0: float = 50, ripple: float = 0.5):
        """
        Parameters
        ----------
        fc : float
            3 dB cutoff frequency [Hz]
        n : int
            Filter order (1-10). Odd n will be matched to 50 ohms, while even n requires an impedance match.
        r0 : float, default: 50
            Port impedance. Default is 50 ohms.
        ripple : float, default: 0.5
            Pass-band ripple. Supported values are 0.5 and 3.0 dB.
        """
        # get n+1 normalized element values. drop the last value which is the resistive load.
        proto_vals = utils.lp_filter_prototype(n, ripple=ripple)[:-1]

        components = dict()
        wc = 2 * np.pi * fc

        # build components for each element, inductors are transformed to capacitors in the HP conversion, 
        # and vice versa
        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0: # start with series capacitor
                components[f"C{i+1}"] = Capacitor(1 / (r0 * wc * g_k))
            else:
                components[f"L{i+1}"] = Inductor(r0 / (g_k * wc), shunt=True)

        cascades = [["P1"] + list(components.values()) + ["P2"]]

        super().__init__(components, cascades=cascades, passive=True)


class BandPassFilter(DynamicNetwork):
    """
    Lumped component band-pass filter
    """

    def __init__(self, fc1: float, fc2: float, n: int, r0: float = 50, ripple: float = 0.5):
        """
        Parameters
        ----------
        fc1 : float
            Lower 3 dB cutoff frequency [Hz]
        fc2 : float
            Upper 3 db cutoff frequency [Hz]
        n : int
            Filter order (1-10). Odd n will be matched to 50 ohms, while even n requires an impedance match.
        r0 : float, default: 50
            Port impedance. Default is 50 ohms.
        ripple : float, default: 0.5
            Pass-band ripple. Supported values are 0.5 and 3.0 dB.
        """

        # band pass 
        components = dict()

        r0 = 50
        wc1, wc2 = 2 * np.pi * fc1, 2 * np.pi * fc2
        w0 = np.sqrt(wc1 * wc2) # geometric mean
        fb = (wc2 - wc1) / w0 # fractional bandwidth

        proto_vals = utils.lp_filter_prototype(n, ripple=ripple)[:-1]

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0:  # start with series element
                L = (g_k * r0) / (w0 * fb)
                C = fb / (r0 * w0 * g_k)
                components[f"S{i+1}"] = LC_Series(L=L, C=C)
            else:
                L = (fb * r0) / (w0 * g_k)
                C = g_k / (w0 * fb * r0)
                components[f"P{i+1}"] = LC_Parallel(L=L, C=C, shunt=True)

        cascades = [["P1"] + list(components.values()) + ["P2"]]
        super().__init__(components, cascades=cascades, passive=True)


class BandStopFilter(DynamicNetwork):
    """
    Lumped component band-stop filter
    """

    def __init__(self, fc1: float, fc2: float, n: int, r0: float = 50, ripple: float = 0.5):
        """
        Parameters
        ----------
        fc1 : float
            Lower 3 dB cutoff frequency [Hz]
        fc2 : float
            Upper 3 db cutoff frequency [Hz]
        n : int
            Filter order (1-10). Odd n will be matched to 50 ohms, while even n requires an impedance match.
        r0 : float, default: 50
            Port impedance. Default is 50 ohms.
        ripple : float, default: 0.5
            Pass-band ripple. Supported values are 0.5 and 3.0 dB.
        """

        # band pass 
        components = dict()

        r0 = 50
        wc1, wc2 = 2 * np.pi * fc1, 2 * np.pi * fc2
        w0 = np.sqrt(wc1 * wc2) # geometric mean
        fb = (wc2 - wc1) / w0 # fractional bandwidth

        proto_vals = utils.lp_filter_prototype(n, ripple=ripple)[:-1]

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0:  # start with series element
                L = (g_k * r0 * fb) / (w0)
                C = 1 / (r0 * w0 * g_k * fb)
                components[f"S{i+1}"] = LC_Parallel(L=L, C=C)
            else:
                L = r0 / (w0 * g_k * fb)
                C = (g_k * fb) / (w0 * r0)
                components[f"P{i+1}"] = LC_Series(L=L, C=C, shunt=True)

        cascades = [["P1"] + list(components.values()) + ["P2"]]
        super().__init__(components, cascades=cascades, passive=True)