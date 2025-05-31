import rfnetwork as rfn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpl_markers as mplm
import sys
from time import time

from PySide6.QtWidgets import (QApplication, QWidget, QLineEdit, QSlider, QGridLayout, QLabel, QVBoxLayout)

np.set_printoptions(suppress=True, threshold=12)

DATA_DIR = Path(__file__).parent / "data/PD55008E_S_parameter"

# frequency range for plots
frequency = np.arange(300, 500, 10) * 1e6 
f0 = 440e6 # design frequency

def smithchart_marker(ax, fc: float, **properties):
    """ place a smith chart marker by frequency rather than x/y position """
    
    f_idx = np.argmin(np.abs(frequency - fc))
    return mplm.line_marker(
        idx=f_idx, axes=ax, xline=False, yformatter=lambda x, y, idx: f"{frequency[idx]/1e6:.0f}MHz", **properties
)


pa_8w = rfn.Component_SnP(
    file={
        150: DATA_DIR / "PD55008E_150mA.s2p", 
        800: DATA_DIR / "PD55008E_800mA.s2p", 
        1500: DATA_DIR / "PD55008E_1500mA.s2p"
    }
)

# 50 ohm microstrip model, substrate is from the amplifier evaluation board.
ms50 = rfn.elements.MSLine(
    h=0.030, 
    er=2.55, 
    w=0.08,
    loss_tan=0.017,
)


class pa_input(rfn.Network):
    """
    Amplifier input matching network
    """
    c2 = rfn.elements.Capacitor(12e-12, shunt=True)
    ms3 = ms50(1.1)
    c1 = rfn.elements.Capacitor(40e-12, shunt=True)
    ms2 = ms50(0.4)
    r1 = rfn.elements.Resistor(2)

    # Port 2 will connect to port 1 of the amplifer
    cascades = [
        ("P1", c2, ms3, c1, ms2, r1, "P2"),
    ]

    probes=True

class pa_output(rfn.Network):
    """
    PA output matching network
    """
    ms1 = ms50(0.4) # length is in inches
    c1 = rfn.elements.Capacitor(65e-12, shunt=True)
    ms2 = ms50(1.1)
    c2 = rfn.elements.Capacitor(15e-12, shunt=True)

    # Port 1 will connect to the amplifier output
    cascades = [
        ("P1", ms1, c1, ms2, c2, "P2"),
    ]
    
    # individual probes can be assigned here, but setting to True
    # creates a probe at every internal node of the network.
    probes=True



class pa_match(rfn.Network):
    """
    Matched amplifier circuit
    """
    m_in = pa_input()
    u1 = pa_8w(state=150)
    m_out = pa_output()

    cascades = [
        ("P1", m_in, u1, m_out, "P2"),
    ]

    probes=True

n = pa_match()

lines = dict()

def plot(init=False):

    n._cache = n.evaluate(frequency)

    if init:
        lines1 = None
        lines2 = None
        ln_s11 = None
        ln_s22 = None
    else:
        lines1 = lines["ax1"]
        lines2 = lines["ax2"]
        ln_s11 = lines["s11"]
        ln_s22 = lines["s22"]

    ax1, lines1 = n.plot_probe(
        frequency,
        ("u1|1", "m_in|2"),
        ("m_in.r1|1", "m_in.ms2|2"),
        ("m_in.ms2|1", "m_in.c1|2"),
        ("m_in.c1|1", "m_in.ms3|2"),
        ("m_in.ms3|1", "m_in.c2|2"),
        input_port=1, fmt="smith", return_lines=True, lines=lines1
    )

    ax1, ln_s11 = n.plot(frequency, 11, axes=ax1, fmt="smith", return_lines=True, lines=ln_s11)

    if init:
        mplm.init_axes(ax1)
        smithchart_marker(ax1, f0, ylabel=False, lines=lines1)
        smithchart_marker(ax1, f0, lines=ln_s11)
        lines["ax1"] = lines1
        lines["s11"] = ln_s11
    else:
        mplm.draw_all(ax1)


    ax2, lines2 = n.plot_probe(
        frequency,
        ("u1|2", "m_out|1"),
        ("m_out.ms1|2", "m_out.c1|1"),
        ("m_out.c1|2", "m_out.ms2|1"),
        ("m_out.ms2|2", "m_out.c2|1"),
        input_port=2, fmt="smith", return_lines=True, lines=lines2
    )

    ax2, ln_s22 = n.plot(frequency, 22, axes=ax2, fmt="smith", return_lines=True, lines=ln_s22)

    if init:
        mplm.init_axes(ax2)
        smithchart_marker(ax2, f0, ylabel=False, lines=lines2)
        smithchart_marker(ax2, f0, lines=ln_s22)
        lines["ax2"] = lines2
        lines["s22"] = ln_s22
    else:
        mplm.draw_all(ax2)

    n._cache = None

def timeplot():
    stime = time()
    plot()
    print(time() - stime)

plot(init=True)
plt.show(block=False)


qapp = QApplication.instance()
if qapp is None:
    qapp = QApplication(sys.argv)



tuners = {
    "m_in.c2": dict(lower=1, upper=30, initial=12, label="C2 [pF]", multiplier=1e-12),
    "m_in.ms3": dict(lower=0.1, upper=2, initial=1.1, label="MS3 [in]", multiplier=1),
    "m_in.c1": dict(lower=10, upper=80, initial=40, label="C1 [pF]", multiplier=1e-12),
    "m_in.ms2": dict(lower=0.1, upper=1, initial=0.4, label="MS2 [in]", multiplier=1),
    "m_out.ms1": dict(lower=0.1, upper=1, initial=0.4, label="MS1 [in]", multiplier=1),
    "m_out.c1": dict(lower=10, upper=100, initial=65, label="C1 [pF]", multiplier=1e-12),
    "m_out.ms2": dict(lower=0.1, upper=2, initial=1.1, label="MS2 [in]", multiplier=1),
    "m_out.c2": dict(lower=5, upper=30, initial=15, label="C2 [pF]", multiplier=1e-12),
}




# add callback functions
for k, v in tuners.items():
    component = n
    for c in k.split("."):
        component = component[c]

    tuners[k]["callback"] = component.set_state

window = rfn.TunerGroup(tuners, plot)

window.setWindowTitle("Tuning")
window.show()
qapp.exec()

##################
