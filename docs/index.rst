.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contents:

   api/index
   auto_examples/index
   
=========
rfnetwork
=========

Linear circuit solver for networks of RF components.

.. |github| image:: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png
   :height: 20px

|github| `Github Repository <https://github.com/ricklyon/rfnetwork>`_

Installation
============

``rfnetwork`` requires Python >= 3.9.

.. code-block:: bash

   pip install rfnetwork


Features
========

* Simulate networks of linear components with hierarchical building blocks.
* Compute noise figure of multi-port networks.
* Interactive tuning of variable components (i.e. switches, phase shifters, capacitors). 
* Supports internal voltage probes inside a network. 
* GPU accelerated FDTD solver.


.. include:: auto_examples/index.rst
   :start-after: :orphan:
   :end-before: .. toctree::

.. include:: api/index.rst







