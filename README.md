# rfnetwork


Linear circuit solver for networks of RF components.

Installation
============

``rfnetwork`` requires Python >= 3.9. Wheels are currently only available on Linux.

```bash
   pip install rfnetwork
```

To build from source on Windows or Linux, 

```bash
   git clone https://github.com/ricklyon/rfnetwork.git
   cd rfnetwork
   git submodule update --init --recursive
   pip install -e .
```

If building from source, a C++ compiler must be available on the system. On Windows systems the recommended compiler
is the [MS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).   
In addition,`nvcc` must be present if compiling the GPU accelerated solver. The CPU solver will still compile if `nvcc`
is not found, but the GPU solver will not be available.

Features
========

* Simulate networks of RF components with hierarchical building blocks.
* Compute noise figure of multi-port networks.
* Interactive tuning of variable components (i.e. switches, phase shifters, capacitors). 
* Supports internal voltage probes inside a network. 
* GPU accelerated FDTD solver.

Performance
===========

FDTD solver metrics for the [dipole.py example](https://rfnetwork.readthedocs.io/en/latest/auto_examples/dipole/html#sphx-glr-auto-examples-dipole-py):

CPU: Intel i7-9700 (8) @ 4.700GHz   
Memory Speed: 2666 MT/s  
GPU: NVIDIA Quadro P2000   

| Cells | Time Steps | CPU | GPU | File | Mesh Settings|
|------ |------------|-----|------|-----| ----- |
| 156k | 1291 | 5.43s  | 1.73s  | [dipole.py](https://rfnetwork.readthedocs.io/en/latest/auto_examples/dipole/html#sphx-glr-auto-examples-dipole-py) | d_max=0.01, d_min=0.005, t_len=400e-12 |
| 1.253M   | 1721 | 56.12s | 14.95s | [dipole.py](https://rfnetwork.readthedocs.io/en/latest/auto_examples/dipole/html#sphx-glr-auto-examples-dipole-py) | d_max = 0.02, d_min = 0.01, t_len = 600e-12 |
| 43k      | 50k  | 12.3s  | 5.8s   | [combline_stripline.py](https://rfnetwork.readthedocs.io/en/latest/auto_examples/combline_stripline.html#sphx-glr-auto-examples-combline-stripline-py) | d_max = 0.02, d_min = 0.005, t_len=1.11e-8|



Documentation
=============

Documentation can be found here,

https://rfnetwork.readthedocs.io/en/latest/

License
=============

rfnetwork is licensed under the MIT License.
