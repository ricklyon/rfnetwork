import rfnetwork as rfn
import numpy as np
import matplotlib.pyplot as plt

# 0.5dB Ripple. Table 8.4 Pozar
lp_filter_0p5_ripple = {
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
}

# 3.0dB Ripple.
lp_filter_3p0_ripple = {
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

def lp_filter(fc: float, n: int) -> rfn.Network:
    """
    Returns an order n (1-10) ideal lumped element low pass filter network with cutoff frequency fc, and
    0.5dB pass band ripple.

    Odd n will be matched to 50 ohms. Even n requires an impedance match.
    """
    proto_vals = lp_filter_0p5_ripple[n][:-1]
    r0 = 50
    wc = 2 * np.pi * fc
    
    def lp_filter_components():
        components = dict()

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0: # start with series inductor
                components[f"L{i+1}"] = rfn.elements.Inductor((g_k * r0) / wc)
            else:
                components[f"C{i+1}"] = rfn.elements.Capacitor(g_k / (r0 * wc), shunt=True)
        
        return components

    class Filter_LP(rfn.Network):
        components = lp_filter_components()
        cascades = [["P1"] + list(components.values()) + ["P2"]]

    return Filter_LP(passive=True)

def hp_filter(fc: float, n: int) -> rfn.Network:
    """
    Returns an order n (1-10) ideal lumped element high pass filter network with cutoff frequency fc and order n (1-10), and
    0.5dB pass band ripple.

    Odd n will be matched to 50 ohms. Even n requires an impedance match.
    """
    proto_vals = lp_filter_0p5_ripple[n][:-1]
    r0 = 50
    wc = 2 * np.pi * fc

    def hp_filter_components():
        components = dict()

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0: # start with series capacitor
                components[f"C{i+1}"] = rfn.elements.Capacitor(1 / (r0 * wc * g_k))
            else:
                components[f"L{i+1}"] = rfn.elements.Inductor(r0 / (g_k * wc), shunt=True)

        return components

    class Filter_HP(rfn.Network):
        components = hp_filter_components()
        cascades = [["P1"] + list(components.values()) + ["P2"]]

    return Filter_HP(passive=True)

def bp_filter(fc1: float, fc2: float, n: int, mode: str = "bandpass") -> rfn.Network:
    """
    Returns an order n (1-10) lumped element band pass filter network with cutoff frequencies [fc1, fc2], and
    0.5dB pass band ripple.

    Odd n will be matched to 50 ohms. Even n requires an impedance match.

    If mode is "bandstop", returns a bandstop filter with the same bandwidth. All other values of mode will return
    a bandpass filter.

    See Table 8.6 in Pozar
    """
    proto_vals = lp_filter_0p5_ripple[n][:-1]
    r0 = 50
    wc1, wc2 = 2 * np.pi * fc1, 2 * np.pi * fc2
    w0 = np.sqrt(wc1 * wc2) # geometric mean
    fb = (wc2 - wc1) / w0 # fractional bandwidth

    class Filter_Series_Section(rfn.Network):
        l1 = rfn.elements.Inductor()
        c2 = rfn.elements.Capacitor()
        cascades = [["P1"] + [l1, c2] + ["P2"]]

    class Filter_Parallel_Section(rfn.Network):
        l1 = rfn.elements.Inductor()
        c2 = rfn.elements.Capacitor()
        nodes = [
            ("P1", l1|1, c2|1),
            ("P2", l1|2, c2|2),
        ]

    def bp_filter_components():
        # band pass 
        components = dict()

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0:  # start with series element
                L = (g_k * r0) / (w0 * fb)
                C = fb / (r0 * w0 * g_k)
                components[f"S{i+1}"] = Filter_Series_Section(l1=L, c2=C, passive=True)
            else:
                L = (fb * r0) / (w0 * g_k)
                C = g_k / (w0 * fb * r0)
                components[f"P{i+1}"] = Filter_Parallel_Section(l1=L, c2=C, shunt=True, passive=True)

        return components
    
    def bs_filter_components():
        # band stop
        components = dict()

        for i, g_k in enumerate(proto_vals):
            if i % 2 == 0:  # start with series element
                L = (g_k * r0 * fb) / (w0)
                C = 1 / (r0 * w0 * g_k * fb)
                components[f"S{i+1}"] = Filter_Parallel_Section(l1=L, c2=C, passive=True)
            else:
                L = r0 / (w0 * g_k * fb)
                C = (g_k * fb) / (w0 * r0)
                components[f"P{i+1}"] = Filter_Series_Section(l1=L, c2=C, shunt=True, passive=True)

        return components

    class Filter_BP(rfn.Network):
        components = bs_filter_components() if mode == "bandstop" else bp_filter_components()
        cascades = [["P1"] + list(components.values()) + ["P2"]]

    return Filter_BP(passive=True)

if __name__ == "__main__":
    # compare with example 8.4 in Pozar
    f0 = 1e9
    fc1 = 1e9-50e6
    fc2 = 1e9+50e6
    n = 3
    f = bp_filter(fc1, fc2, 3)

    frequency = np.arange(0.5e9,  1.5e9, 1e6)
    f.evaluate(frequency)

    ax = rfn.plots.plot(f.sdata, 21)
    ax.set_ylim([-50, 0])