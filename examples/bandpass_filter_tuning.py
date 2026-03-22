"""
Filters
=======

Interactively tune a lumped element band-pass filter. 
"""

import rfnetwork as rfn
import numpy as np
import matplotlib.pyplot as plt
import mpl_markers as mplm

f1 = 1.1e9
f2 = 1.6e9

f0 = (f1 + f2) / 2

g = [1, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000]
bpf = rfn.elements.LumpedElementFilter(fc=(f1, f2), btype="bandpass", prototype=g)

frequency = np.arange(10e6, 3e9, 1e6)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9), height_ratios=[1, 2], constrained_layout=True)
bpf.plot(11, 21, freq_unit="mhz", frequency=frequency, axes=ax1, tune=True)
ax1.set_ylim([-30, 2])

bpf.plot(11, freq_unit="mhz", fmt="smith", frequency=frequency, axes=ax2, tune=True)
mplm.line_marker(x=1200, axes=ax1, xlabel=True)

tuners = [
    dict(component="S1.l1", label="L1 [nH]", multiplier=1e-9),
    dict(component="S1.c2", label="C1 [pF]", multiplier=1e-12),
    dict(component="P2.l1", label="L2 [nH]", multiplier=1e-9),
    dict(component="P2.c2", label="C2 [pF]", multiplier=1e-12),
    dict(component="S3.l1", label="L3 [nH]", multiplier=1e-9),
    dict(component="S3.c2", label="C3 [pF]", multiplier=1e-12),
    dict(component="P4.l1", label="L4 [nH]", multiplier=1e-9),
    dict(component="P4.c2", label="C4 [pF]", multiplier=1e-12),
    dict(component="S5.l1", label="L5 [nH]", multiplier=1e-9),
    dict(component="S5.c2", label="C5 [pF]", multiplier=1e-12),
]

# start the tuner
bpf.tune(tuners)

# print tuned values (unless tune was canceled, then this just shows the initial values)
print(bpf.state)