"""
Filters
=======

Interactively tune a lumped element band-pass filter. 
"""

import rfnetwork as rfn
import numpy as np
import matplotlib.pyplot as plt

f1 = 1.1e9
f2 = 1.6e9

f0 = (f1 + f2) / 2

g = [1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0000]
bpf = rfn.elements.LumpedElementFilter(fc=(f1, f2), btype="bandpass", prototype=g)

frequency = np.arange(10e6, 3e9, 1e6)

ax = plt.axes()
bpf.plot(11, 21, freq_unit="mhz", frequency=frequency, axes=ax, tune=True)
ax.set_ylim([-30, 2])

plt.figure()
ax = plt.axes()
bpf.plot(11, freq_unit="mhz", fmt="smith", frequency=frequency, axes=ax, tune=True)

bpf.state

tuners = [
    dict(component="S1.l1", variable="value", lower=20, upper=45, label="L1 [nH]", multiplier=1e-9),
    dict(component="S1.c2", variable="value", lower=0.05, upper=2, label="C1 [pF]", multiplier=1e-12),
    dict(component="P2.l1", variable="value", lower=1, upper=3, label="L2 [nH]", multiplier=1e-9),
    dict(component="P2.c2", variable="value", lower=5, upper=10, label="C2 [pF]", multiplier=1e-12),
]

# start the tuner (this is disabled for the readthedocs runner)
bpf.tune(tuners)