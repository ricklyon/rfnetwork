"""
Filters
=======

Examples of basic, lumped element filter.
"""

import rfnetwork as rfn
import numpy as np
import matplotlib.pyplot as plt


# %%
# Bandstop Filter
# ----------------
# sphinx_gallery_thumbnail_number = -1
#

bpf = rfn.elements.BandStopFilter(fc1=70e6, fc2=150e6, n=3)

frequency = np.arange(10e6, 0.5e9, 1e6)

ax = plt.axes()
bpf.plot(21, freq_unit="mhz", frequency=frequency, axes=ax)
ax.set_ylim([-50, 0]);

# %%
# Bandpass Filter
# ----------------
#
bpf = rfn.elements.BandPassFilter(fc1=70e6, fc2=150e6, n=3)

frequency = np.arange(10e6, 0.5e9, 1e6)

ax = plt.axes()
bpf.plot(21, freq_unit="mhz", frequency=frequency, axes=ax)
ax.set_ylim([-50, 0]);


# %%
# Lowpass Filter
# ----------------
#
# Generate Figures 8.27a (0.5 Ripple) in Pozar 4th ed.

fc = 150e6
frequency = np.logspace(0, 11, 5000)
n_list = [1, 3, 5, 7, 9]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

for n in n_list:
    # build lowpass filter network
    fn = rfn.elements.LowPassFilter(fc, n)
    sdata = fn.evaluate(frequency)["s"]
    ax.plot((frequency / fc) - 1, -rfn.conv.db20_lin(sdata.sel(b=2, a=1)), linewidth=2)

# plot formatting
ax.set_ylim([0, 70])
ax.set_xscale("log")
ax.set_xlim([10e-2, 10])
ax.grid(True)
base_xticks = np.array([1, 2, 3, 5, 7])
xticks = np.concatenate([
        base_xticks * 1e-2, base_xticks * 1e-1, base_xticks, [10]
])
ax.set_xticks(xticks)

ax.set_xticklabels(
    xticks,
    fontsize=10
)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
ax.legend([f"n={n}" for n in n_list])
ax.set_xlabel(r"$|\frac{\omega}{\omega_c}| - 1$", fontsize=13)
ax.set_ylabel("Attenuation [dB]")