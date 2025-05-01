
import numpy as np

from .component import Component
from .core.units import conv, const
from .core import core
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
    "Junction"
    "Attenuator"
)

class Junction(Component):
    """
    N-port lossless junction
    """

    def __init__(self, N):
        """
        Parameters:
        ----------
        N (int):
            number of junction ports
        """
        super().__init__(pnum=N)
        assert N > 0, "N must be positive"
        self.N = int(N)

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return core.junction_sdata(frequency, self.N)
    

class Load(Component):
    """
    1-port Load with variable impedance.
    """

    def __init__(self, value=50):
        """
        Parameters:
        ----------
        value (float, complex):
            real or complex impedance [ohms]. Defaults to 50 ohms
        """
        super().__init__(pnum=1)
        assert value > 0, "Load must be passive"
        self.value = value

    def equals(self, other):
        return super().equals(other) and (other.value == self.value)

    def set_state(self, state):
        self.value = state

    @property
    def state(self):
        return self.value
    
    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return np.full((len(frequency), 1, 1), conv.gamma_z(self.value), dtype="complex128")

class Open(Component):
    """
    1-port open-circuit load
    """
    def __init__(self):
        super().__init__(pnum=1)

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return np.full((len(frequency), 1, 1), 1, dtype="complex128")


class Short(Component):
    """
    1-port short-circuit load
    """
    def __init__(self):
        super().__init__(pnum=1)

    def evaluate_sdata(self, frequency: np.ndarray):
        return np.full((len(frequency), 1, 1), -1, dtype="complex128")


class Hybrid90(Component):
    """
    Ideal directional coupler with 90 degree phase delay between the thru and coupled paths.

    Port 1: Input
    Port 2: Thru
    Port 3: Coupled
    Port 4: Isolated
    """

    def __init__(self):
        super().__init__(pnum=4)

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

    Port 1: Input
    Port 2: Thru
    Port 3: Coupled
    Port 4: Isolated
    """

    def __init__(self):
        super().__init__(pnum=4)

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

    def __init__(self, value: float = 0, shunt: bool = False):
        """
        Parameters
        ----------
        value : float
            Resistance (Ohms), capacitance (F), or inductance (H) of component.
        """
        self.value = value
        super().__init__(passive=True, shunt=shunt)
        
    def equals(self, other):
        return super().equals(other) and (other.value == self.value)

    def set_state(self, value):
        self.value = value

    @property
    def state(self):
        return self.value
    
    def evaluate_sdata_from_z(self, z: np.ndarray):

        # initialize new sdata matrix
        shape = (len(z), 2, 2)
        sdata = np.zeros(shape, dtype="complex128")

        # generate new s-matrix data for series element.
        z0 = 50
        s11 = z / (2 * z0 + z)
        s21 = (1 + s11) * (z0 / (z + z0))

        # there's probably a more pythonic way to do this:
        sdata[:, 0, 0] = s11
        sdata[:, 1, 1] = s11
        sdata[:, 0, 1] = s21
        sdata[:, 1, 0] = s21

        return sdata
             


class Resistor(LumpedElement):
    """
    Simple Resistor network element.
    """

    def evaluate_sdata(self, frequency: np.ndarray):

        z = np.full(frequency.shape, self.value)
        return super().evaluate_sdata_from_z(z)


class Capacitor(LumpedElement):
    """
    Simple Capacitor network element.
    """

    def evaluate_sdata(self, frequency: np.ndarray):

        # z = 1/jwC
        z = 1 / (1j * 2 * np.pi * frequency * self.value)
        return super().evaluate_sdata_from_z(z)


class Inductor(LumpedElement):
    """
    Simple Inductor network element.

    """
    
    def evaluate_sdata(self, frequency: np.ndarray):

        # z = jwL
        z = 1j * 2 * np.pi * frequency * self.value
        return super().evaluate_sdata_from_z(z)


class Attenuator(Component):
    def __init__(self, attenuation_db=0):
        """
        Parameters:
        ----------
        attenuation_db: float
            positive value attenuation in dB
        """

        self.s21 = core.conv.lin_db20(-np.abs(attenuation_db))
        super().__init__(passive=True)

    def evaluate_sdata(self, frequency: np.ndarray):

        sdata = np.array([[1e-6, self.s21], [self.s21, 1e-6]], dtype="complex128")
        return np.broadcast_to(sdata, (len(frequency), 2, 2)).copy()

class Line(Component):
    """
    Base class for generic transmission lines. Use this to model stripline or GCPW lines.
    """

    def __init__(self, z0=50, er=1, loss=0, frequency=0, length=None, shunt=False):
        """
        Parameters:
        -----------
            z0 (float, list):
                Real characteristic impedance. Assumes line is low loss so only real component of z0 should be given.
            er (float, list, optional):
                Dielectric constant of media. If modeling a mixed media line like microstrip,
                use the effective dielectric constant.
            loss (float, list, optional):
                total loss, (dielectric + conductor) losses, in dB per inch. If not provided 0 db/inch loss is assumed.
            frequency (float, list, optional):
                array of frequencies in Hz matching the parameter lengths above, If the parameters are not arrays,
                freq is ignored since the parameters will be
                constant across all frequencies of the parent Network.
            length (float, optional):
                length in inches

        Any or all parameters can be supplied as a single value, or as a list of values over a frequency range.
        If any of the parameters are provided as a list, a frequency list must be provided that maps
        each value to a frequency.

        Example:
        ---------

        exLine1 = Line( z0=    50,
                        er=     [3.2,  3.5],
                        loss=   [0.5,  0.6],
                        frequency=   [10e9, 12e9] )

        exLine2 = Line(z0=50, er=2.56)

        """

        super().__init__(shunt=shunt)

        self.z0 = z0
        self.er = er
        self.loss = loss
        self._frequency = np.atleast_1d(frequency)
        self.properties = None

        self.length = length

    def equals(self, other):
        if not super().equals(other):
            return False
        
        fields = ("z0", "er", "loss", "length", "_frequency")
        return all([np.all(getattr(self, f) == getattr(other, f)) for f in fields])
    
    def __call__(self, length=None, e_len=None, fc=None):
        # call is used to create a new Line instance with the specified length.
        # this allows multiple Line objects to be created with different lengths from the same line object
        if self.length is not None:
            raise RuntimeError("line already has length of {} in.".format(self.length))

        # assign electrical length if given
        if e_len is not None:
            # wavelength in inches at center frequency:
            lmbda_c = const.c0_in / fc
            # electrical length is e_len = (beta * len) = (2*np.pi)len/lmbda.
            length = (np.deg2rad(e_len) * lmbda_c) / (2 * np.pi)

        # create new instance with length
        lineseg = dcopy(self)
        lineseg.length = float(length)

        return lineseg

    def get_wavelength(self, frequency, units="in"):
        """
        Returns the propagation wavelength [inches] of the medium at the specified frequency (Hz).

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

    def get_delay(self, frequency):
        """
        Returns the phase delay [deg] and time delay [ns] at the specified frequency.

        """
        if self.length is None:
            raise RuntimeError("Line must have length assigned to compute delay")

        lmbda = self.get_wavelength(frequency)

        pdelay = np.rad2deg((2 * np.pi * self.length) / lmbda)

        vp = lmbda * frequency
        tdelay = self.length / vp

        return pdelay, tdelay * 1e9

    def evaluate_sdata(self, frequency):
        """
        Computes the lossy s-matrix of a transmission line.
        """

        if self.length is None:
            raise RuntimeError("Line length must be specified before evaluating sdata")
        
        frequency = np.atleast_1d(frequency)

        properties = self.get_properties(frequency)

        sdata = np.zeros(shape=(len(frequency), 2, 2), dtype="complex128")

        length = self.length
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

    def get_properties(self, frequency):

        frequency = np.atleast_1d(frequency)
        # create a labeled array with rows as frequency and columns as data type (zo, loss, etc...).
        # parameters will be interpolated to match the freq passed into evaluate
        dim = dict(frequency=self._frequency, value=["z0", "er", "loss"])

        # initialize properties array.
        properties = ldarray(dim=dim)

        # populate each column, if values are single numbers numpy will broadcast the value to the correct length.
        for i, p in enumerate([self.z0, self.er, self.loss]):
            # ensure all properties are positive numbers (no negative permittivity materials or amplifier lines)
            properties[..., i] = np.abs(p)

        if len(self._frequency) < 2:
            # broadcast single-point properties across the provided frequency vector
            shape = (len(frequency), len(properties.dim["value"]))
            properties = np.broadcast_to(properties, shape).copy()
        else:
            # interpolate the line properties over frequency
            sp1 = CubicSpline(self._frequency, properties, axis=0)
            properties = sp1(frequency)

        # cast result as ldarray
        ndim = dict(frequency=frequency, value=["z0", "er", "loss"])
        return ldarray(properties, dim=ndim)



class MSLine(Line):
    """
    Models microstrip lines based on closed form equations used in Rogers MWI tool.

    """

    def __init__(
        self, 
        h: float, 
        er: float, 
        w: float=None, 
        z0: float = 0, 
        t: float = 0.0013, 
        loss_tan: float = None, 
        loss: float = 0,
        frequency = 0,
        length = None,
        shunt = False,
    ):
        """
        Parameters:
        ------------
            h (float):
                substrate thickness in inches
            er (float, list):
                dielectric constant of substrate, or design dk. Not the effective dielectric constant.
            w (float, optional):
                width of microstrip trace in inches. Specify w or z0, not both.
            z0 (float, optional):
                target characteristic impedance of line. Width will be set automatically.
            t (float, optional):
                copper thickness in inches, defaults to 1oz copper (1.3 mils). Does not affect loss (yet).
            loss_tan (float, optional):
                dielectric loss tangent as a single value over frequency range. If both loss_tan and loss are provided,
                only loss_tan is used and loss is ignored.

            loss (float, list, optional):
                total loss, (dielectric + conductor) losses, in dB per inch.
            frequency (float, list, optional):
                array of frequencies in Hz for the loss and/ore er lists above.

        er, and loss can be supplied as a single value, or as a list of values over a frequency range. If either are
        provided as a list, a frequency list must be provided that maps each value to a frequency.

        Example:
        -----------
        msline = MSLine(w =   .020,
                        h =   .010,
                        t =   .0013,
                        er=   [3.0, 3.1],
                        loss= [0.5, 0.6],
                        frequency= [10e9, 12e9],
                        )
        """
        self.w = w
        self.h = h
        self.t = t
        self.loss_tan = loss_tan
        # get initial guess for width if z0 was given but not width
        if z0 and (w is None):
            self.w = self.calculate_approx_width(z0, er, h)

        super().__init__(z0=z0, er=er, loss=loss, frequency=frequency, length=length, shunt=shunt)

    def equals(self, other):
        if not super().equals(other):
            return False
        
        fields = ("w", "h", "t", "loss_tan")
        return all([np.all(getattr(self, f) == getattr(other, f)) for f in fields])

    def get_properties(self, frequency):
        """
        Returns a list of characteristic impedance and effective dielectric constants over frequency.

        Reference:
        ---------

            [1] Design-Data-for-Microstrip-Transmission-Lines-on-RT-duroid-Laminates, Rogers Corporation, 2013

            (see ../docs/rogers_transmission_line_eq.pdf)

            [2] Pozar page 149 for loss equations.

        """
        frequency = np.atleast_1d(frequency)
        f_ghz = frequency / 1e9

        # get the interpolated list of line properties
        properties = super().get_properties(frequency)

        er = properties[{"value": "er"}]
        # B is in mm
        B = conv.mm_mil(self.h * 1e3)
        T = self.t / self.h
        U = self.w / self.h
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
        if self.loss_tan is not None:

            # interpolate loss tangent
            if len(self._frequency) < 2:
                # broadcast single-point properties across the provided frequency vector
                loss_tan = np.broadcast_to(self.loss_tan, len(frequency))
            else:
                # interpolate the line properties over frequency
                sp1 = CubicSpline(self._frequency, self.loss_tan)
                loss_tan = sp1(frequency)

            lmbda0_in = const.c0_in / (f_ghz * 1e9)
            k0 = (2 * np.pi) / lmbda0_in
            # calculate dielectric loss per inch
            ad_in = ((k0 * er) * (eff_f - 1) * loss_tan) / (2 * np.sqrt(eff_f) * (er - 1))

            # calculate copper resistivity using conductivity in seimens per meter
            Rs = np.sqrt((2 * np.pi * f_ghz * 1e9 * const.u0) / (2 * const.cu_sigma))

            # calculate losses due to copper conductor in np/m.
            # use width in meters
            w_m = conv.m_in(self.w)
            a_c = Rs / (Z0_f * w_m)
            # convert loss to per inch
            ac_in = a_c / 39.3701

            # convert total loss to positive valued loss in db per inch
            properties[{"value": "loss"}] = -conv.db20_lin(np.exp(-(ad_in + ac_in)))

        return properties

    def calculate_approx_width(self, z, er, d):
        """
        [2] Pozar page 149
        """
        A = (z / 60) * np.sqrt((er+1)/2) + ((er-1)/(er+1)) *(0.23 + (0.11/er))
        B = (377 * np.pi) /( 2 * z * np.sqrt(er))

        # w/d > 2
        wd2 = (2 / np.pi) * ( B -1 - np.log(2*B -1) + (er-1)/(2*er) *( np.log(B-1) + 0.39 - (0.61 / er)))
        # w/d < 2
        wd12 = (8 * np.exp(A)) /( np.exp(2* A) -2)

        if wd2 > 2:
            return wd2*d
        else:
            return wd12*d


class Stripline(Line):
    """
    Models stripline lines.

    """

    def __init__(
            self, 
            w: float, 
            b: float, 
            er: float,
            t: float = 0.0006, 
            loss_tan: float = None, 
            loss: float = 0,
            frequency = 0,
            length = None
        ):
        """
        Parameters:
        ------------
            w (float):
                width of stripline trace in inches
            b (float):
                substrate thickness in inches between the ground planes (with line in the center)
            t (float, optional):
                copper thickness in inches, defaults to 1oz copper (1.3 mils). Does not affect loss (yet).
            er (float, list):
                dielectric constant of substrate, or design dk. Not the effective dielectric constant.
            loss_tan (float, optional):
                dielectric loss tangent as a single value over frequency range. If both loss_tan and loss are provided,
                only loss_tan is used and loss is ignored.

            loss (float, list, optional):
                total loss, (dielectric + conductor) losses, in dB per inch.
            frequency (float, list, optional):
                array of frequencies in Hz for the loss and/ore er lists above.

        er, and loss can be supplied as a single value, or as a list of values over a frequency range. If either are
        provided as a list, a frequency list must be provided that maps each value to a frequency.
        """
        
        self.w = w
        self.b = b
        self.t = t
        self.loss_tan = loss_tan
        super().__init__(z0=0, er=er, loss=loss, frequency=frequency, length=length)

    def equals(self, other):
        if not super().equals(other):
            return False
        
        fields = ("w", "b", "t", "loss_tan")
        return all([np.all(getattr(self, f) == getattr(other, f)) for f in fields])
    
    def get_properties(self, frequency):
        """
        Returns a list of characteristic impedance and effective dielectric constants over frequency.

        Reference:
        ---------

            [1] Pozar section 3.8

        """

        # use properties freq if a single value frequency is not provided.

        frequency = np.atleast_1d(frequency)
        f_ghz = frequency / 1e9

        # get the interpolated list of line properties
        properties = super().get_properties(frequency)

        er = properties[{"value": "er"}]

        # effective width
        w_over_b = (self.w / self.b) 
        we_b = w_over_b if w_over_b >= 0.35 else w_over_b - (0.35 - w_over_b)**2

        # characteristic impedance
        n = np.sqrt(er)
        z0 = ((30*np.pi) / n) * (1 / (we_b + 0.441))

        # update the properties values
        self.properties[{"value": "z0"}] = z0

        # generate dielectric loss from the loss tangent, as well as conductor losses
        if self.loss_tan is not None:

            # interpolate loss tangent
            if len(self._frequency) < 2:
                # broadcast single-point properties across the provided frequency vector
                loss_tan = np.broadcast_to(self.loss_tan, len(frequency))
            else:
                # interpolate the line properties over frequency
                sp1 = CubicSpline(self._frequency, self.loss_tan)
                loss_tan = sp1(frequency)

            # calculate copper resistivity using conductivity in seimens per meter
            Rs = np.sqrt((2 * np.pi * f_ghz * 1e9 * const.u0) / (2 * const.cu_sigma))

            # 'A' parameter. We can stay in inches here since everything is a ratio
            w = self.w
            b = self.b
            t = self.t
            A = 1 + ((2*w)/(b-t)) + (1/np.pi)*((b+t)/(b-t)) * np.log((2*b -t)/t)

            # 'B' parameter
            B = 1 + (b / (0.5*w + 0.7*t)) * (0.5 + ((0.414*t)/w) + (1/(2*np.pi))*np.log((4*np.pi*w)/t))

            # conductor attenuation constant in inches^-1
            a_c_gt = (0.16 * Rs * B) / (z0*b)
            a_c_lt = (2.7e-3 * Rs * er * z0 * A) / (30*np.pi*(b-t))

            a_c = np.where(n*z0 < 120, a_c_lt, a_c_gt)

            # equation 3.30 in pozar
            k_in = (2*np.pi*f_ghz*1e9) * (n / const.c0_in)
            a_d = (k_in * loss_tan) /2

            # convert total loss to positive valued loss in db per inch
            self.properties[{"value": "loss"}] = -conv.db20_lin(np.exp(-(a_d+ a_c)))

