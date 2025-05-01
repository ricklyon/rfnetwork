
import rfnetwork as rfn
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

font = {'size'   : 8}
default_cycler = (cycler(color=['teal', 'm', 'y', 'k']))
plt.rc('axes', prop_cycle=default_cycler)

matplotlib.rc('font', **font)

DATA_DIR = Path(__file__).parent / 'data'

line50 = rfn.elements.Line(z0=50)
line70p7 = rfn.elements.Line(z0=70.7)

f0 = 3e9
lam0 = rfn.const.c0_in / f0
frequency = np.arange(0.6, 6, 0.01) * 1e9

lna = rfn.Component_SnP(file=DATA_DIR / "QPL9503_5V_with_NP.s2p")

class splitter(rfn.Network):
    """
    Non-isolated splitter.
    """
    p1 = line50(e_len=20, fc=f0)
    p2 = line70p7(lam0/4)
    p3 = line70p7(lam0/4)

    nodes = [
        (p1|2, p2|1, p3|1),
        (p3|2, "P3"),
        (p2|2, "P2"),
        (p1|1, "P1"),
    ]

class wilk(rfn.Network):
    """
    Isolated wilkison combiner/splitter
    """
    p1 = line50(e_len=20, fc=f0)
    p2 = line70p7(lam0/4)
    p3 = line70p7(lam0/4)
    r1 = rfn.elements.Resistor(100)

    nodes = [
        (p1|2, p2|1, p3|1),
        (p2|2, r1|1, "P2"),
        (p3|2, r1|2, "P3"),
        (p1|1, "P1"),
    ]

class single_amp(rfn.Network):
    u1 = lna
    att = rfn.elements.Attenuator(0.5)

    cascades = [
        ("P1", att, u1, "P2")
    ]


class tuned_amp(rfn.Network):
    u1 = lna
    att = rfn.elements.Attenuator(0.5)
    cap = rfn.elements.Capacitor(0.4e-12, shunt=True)
    l0 = line50(e_len=9.15, fc=1e9)

    cascades = [
        ("P1", cap, l0, att, u1, "P2")
    ]

class dual_amp(rfn.Network):
    u1 = single_amp()
    u2 = single_amp()
    sp = wilk()

    nodes = [
        (sp|1, "P1"),
        (sp|2, u1|1),
        (sp|3, u2|1),
        (u1|2, "P2"),
        (u2|2, "P3"),
    ]

class dual_amp_noniso(rfn.Network):
    u1 = single_amp()
    u2 = single_amp()
    sp = splitter()

    nodes = [
        (sp|1, "P1"),
        (sp|2, u1|1),
        (sp|3, u2|1),
        (u1|2, "P2"),
        (u2|2, "P3"),
    ]


class dual_amp_thru(rfn.Network):
    u1 = single_amp()
    sp1 = splitter()
    sp2 = splitter()

    cascades = [
        (sp1|2, u1, sp2|2),
    ]

    nodes = [
        (sp1|3, sp2|3),
        (sp1|1, "P1"),
        (sp2|1, "P2"),
    ]

# create instances of each network
amp_n = single_amp()
amp_tune_n = tuned_amp()
amp_dual_n = dual_amp()
amp_dual_noniso = dual_amp_noniso()
amp_dual_thru = dual_amp_thru()

# plot single amp
ads_nf = np.genfromtxt(DATA_DIR / 'ads_lna.csv', delimiter=",")
ads_nf_tune = np.genfromtxt(DATA_DIR / 'ads_tune.csv', delimiter=",")

fig, (ax1, ax2) = plt.subplots(2,1)
amp_n.plot(frequency, 21, fmt='nf', axes=ax1, label='Noise Wave')
ax1.plot(ads_nf[:,0], ads_nf[:, 1], linestyle="--", label='ADS')
ax1.legend(loc='upper left')
ax1.set_ylim([0.5, 2.5])
ax1.set_title('QPL9503 NF')


amp_tune_n.plot(frequency, 21, fmt='nf', axes=ax2, label='Noise Wave')
ax2.plot(ads_nf_tune[:,0], ads_nf_tune[:, 1], '--', label='ADS')
ax2.legend(loc='upper left')
ax2.set_title('Tuned QPL9503 NF')
ax2.set_ylim([0.5, 2.5])
plt.tight_layout()



ax = amp_dual_n.plot(frequency, 21, fmt='nf', label='Noise Wave')
ads_nf = np.genfromtxt(DATA_DIR / 'ads_wilk.csv', delimiter=",")
ax.plot(ads_nf[:,0], ads_nf[:, 1], '--', label='ADS')
ax.legend(loc='upper center')
ax.set_title('QPL9503 Pair NF(2,1)')


ads_nf = np.genfromtxt(DATA_DIR / 'ads_noniso.csv', delimiter=",")
ax = amp_dual_noniso.plot(frequency, 21, fmt='nf', label='Noise Wave')
ax.plot(ads_nf[:,0], ads_nf[:, 1], '--', label='ADS')
ax.legend(loc='upper center')
ax.set_title('QPL9503 Non-Iso Pair NF(2,1)', fontsize='medium')


ads_nf = np.genfromtxt(DATA_DIR / 'ads_noniso_unb.csv', delimiter=",")
ax = amp_dual_thru.plot(frequency, 21, fmt='nf', label='Noise Wave')
ax.plot(ads_nf[:,0], ads_nf[:, 1], '--', label='ADS')
ax.legend()
ax.set_title('QPL9503 Non-Iso Unbalanced NF', fontsize="medium")
ax.set_ylim([0, 5]);

plt.show()