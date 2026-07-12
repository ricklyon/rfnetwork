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

FDTD solver metrics for the [dipole.py](https://rfnetwork.readthedocs.io/en/latest/auto_examples/dipole.html#sphx-glr-auto-examples-dipole-py) 
example on the following device:

CPU: Intel i7-9700 (8) @ 4.700GHz   
Memory Speed: 2666 MT/s  
GPU: NVIDIA Quadro P2000   

Model is simulated over a time span of 400ps.

| Cells | Time Steps | Grid Cell Max/Min [in]| CPU | GPU | 
|------ |------------|-----|------|-----|
| 156k | 860  |  0.02 / 0.01| 2.46s  | 1.06s   |
| 431k | 1721 | 0.015 / 0.005| 12.39s | 5.32s |


Documentation
=============

Documentation can be found here,

https://rfnetwork.readthedocs.io/en/latest/

License
=============

rfnetwork is licensed under the MIT License.
