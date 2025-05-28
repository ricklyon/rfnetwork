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
frequency = np.arange(200, 700, 10) * 1e6 
f0 = 440e6 # design frequency

def smithchart_marker(ax, fc: float):
    """ place a smith chart marker by frequency rather than x/y position """
    
    f_idx = np.argmin(np.abs(frequency - fc))
    return mplm.line_marker(
        idx=f_idx, axes=ax, xline=False, yformatter=lambda x, y, idx: f"{frequency[idx]/1e6:.0f}MHz"
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

pa_m = pa_match()

# ax, lines = pa_m.plot_probe(frequency, ("u1|1", "m_in|2"), input_port=1, fmt="smith", return_lines=True)
ax, lines = pa_m.plot(frequency, 11,22, fmt="smith", return_lines=True)
mplm.init_axes(ax)

smithchart_marker(ax, f0)

# plt.show()
plt.show(block=False)

stime = time()
pa_m.evaluate(frequency)
print("time", time() - stime)



# pa_m.set_state(m_out=dict(ms2=0.2))

pa_m.state

qapp = QApplication.instance()
if qapp is None:
    qapp = QApplication(sys.argv)

I = 0

def callback_c1(val):
    pa_m.set_state(m_in=dict(c2=val * 1e-12))
    pa_m.plot(frequency, 11, fmt="smith", lines=lines)
    mplm.draw_all(ax)

def callback_ms3(val):
    pa_m.set_state(m_in=dict(ms3=val))
    pa_m.plot(frequency, 11, fmt="smith", lines=lines)
    mplm.draw_all(ax)

tuners = {
    "m_in.c2": dict(lower=1, upper=30, initial=12, label="C2 [pf]", callback=callback_c1),
    "m_in.ms3": dict(lower=0.1, upper=2, initial=1.1, label="MS3 [in]", callback=callback_ms3),
}

window = rfn.TunerGroup(tuners)

window.setWindowTitle("Tuning")
window.show()
qapp.exec_()
