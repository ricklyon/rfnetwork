# rfnetwork


Linear circuit solver for networks of RF components.

Installation
============

``rfnetwork`` requires Python >= 3.9. Wheels are currently only available on linux, and can be built from
source on Windows.

```bash
   pip install rfnetwork
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


Documentation
=============

Documentation can be found here,

https://rfnetwork.readthedocs.io/en/latest/

License
=============

rfnetwork is licensed under the MIT License.
