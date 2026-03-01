.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   api/index
   auto_examples/index
   
=========
rfnetwork
=========

Efficient linear circuit solver for networks of RF components.

Installation
============

``rfnetwork`` requires Python >= 3.9.

.. code-block:: bash

   pip install rfnetwork


Features
========

* Connect RF/microwave components into larger networks. 
* Manage complex RF systems with hierarchical building blocks.
* Compute noise figure of multi-port networks using noise-wave analysis.
* Interactive tuning of variable components (i.e. switches, phase shifters, capacitors). 
* Probe internal voltage waves inside a network. 
* Full wave FDTD solver for analyzing coupled lines and simple PCB geometry.


.. include:: auto_examples/index.rst
   :start-after: :orphan:
   :end-before: .. toctree::

.. include:: api/index.rst







