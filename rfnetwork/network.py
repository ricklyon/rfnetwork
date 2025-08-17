from .component import Component, Component_Data, Component_SnP
from .core import core
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from . import netlist as nlist
from np_struct import ldarray
from . import plots
from typing import Tuple, List, Union
from matplotlib.lines import Line2D


from typing import Tuple

def network_assemble(components: dict, nodes: list = list(), cascades: list = list(), probes: dict = dict()):
    """
    Build network netlist from lists of nodes and cascades.
    """
    # compile netlist and network port name to node mapping
    netlist, ports = nlist.build_netlist(nodes, cascades, components)

    # compile a probe map of all valid nodes in the network if probes is set to True
    if probes is True:
        probes = dict()
        for c, nodes in netlist.items():
            for i, n in enumerate(nodes):
                # exclude ground, open and port nodes
                if n not in [0, -1] and n not in ports.values():
                    probes[f"{c}|{i+1}"] = components[c]|(i + 1)

    # similar to the normal netlist, but each value is either None for no probe, or a probe name in the
    # index for the port it's assigned to. Probes the voltage wave leaving the port.
    probe_netlist, probe_names = nlist.build_probe_netlist(components, probes, netlist)

    return ports, netlist, probe_netlist, probe_names


class NetworkMeta(type):
    
    def __new__(metacls, cls, bases, classdict):

        if cls == "Network":
            return super().__new__(metacls, cls, bases, classdict)

        # keys are the ref designators, and values are a list of network nodes for each port of the component.
        nodes = classdict.pop("nodes", list())
        cascades = classdict.pop("cascades", list())
        components = classdict.pop("components", dict())
        probes = classdict.pop("probes", dict())

        for i, (key, value) in enumerate(classdict.items()):
            # isinstance calls type() on the value, which will return NetworkMeta if the value is a Network class.
            if isinstance(value, Component):
                components[key] = value

        # remove network blocks as class members since they are in the components dictionary now
        for key, value in components.items():
            if key in classdict.keys():
                classdict.pop(key)

        ports, netlist, probe_netlist, probe_names = network_assemble(components, nodes, cascades, probes)

        # add _cls_blocks and declared_ports as a class member and call type.__new__
        classdict["cls_components"] = components
        classdict["netlist"] = netlist
        classdict["probe_netlist"] = probe_netlist
        classdict["ports"] = ports
        classdict["probes"] = probes
        classdict["probe_names"] = probe_names

        ncls = super().__new__(metacls, cls, bases, classdict)
        return ncls


class Network(Component, metaclass=NetworkMeta):
    """
    Network of multiple components. 

    Class Attributes
    ----------------
    components : dict
        components of the network can either be declared as class variables, or in a class dictionary named
        "components".
    nodes : list
        list of tuples, where each tuple is a group of component ports that are connected into a single node.
        External network ports are defined by placing "P1" in the node, where "1" should be replaced with the 
        network port number.
    cascades : list
        list of tuples, where each tuple is a group of components that are connected end to end, port 2 to port 1.
        External network ports are defined by placing "P1" at either or both ends of the cascade list, where "1" should
        be replaced with the network port number.
    probes: dict
        dictionary of probe names to the component port they attach to. Probe voltage waves are defined as the 
        wave leaving the component from the specified port. For example, `dict(probe1=c1|1)`, would attach a probe
        to port 1 of the "c1" component. Probe names will appear in the coords of the "b" dimension of the s-matrix
        data returned by evaluate. 

    """
    def __init__(self, shunt: bool = False, passive: bool = False, state: dict = dict()):
        """
        Parameters
        ----------
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.
        passive : bool, default: True
            if True, ``evaluate`` calls ``evaluate_sdata`` instead of ``evaluate_data`` and noise correlation
            matrix is computed passively.
        state : dict, optional
            dictionary of state variables specific to each component. Keys must be component designators.
            The state values can be read with the ``state`` property and changed later with `set_state()`. Attempting
            to set the state of variables that were not included in the initial dictionary will raise an error.

        Examples
        --------
        >>> import rfnetwork as rfn

        >>> class Wilkinson(rfn.Network):
        ...    
        ...     upper = rfn.elements.Line(z0=70.7, length=0.4)
        ...     lower = rfn.elements.Line(z0=70.7, length=0.4)
        ... 
        ...     r1 = rfn.elements.Resistor(100)
        ... 
        ...     nodes = [
        ...         ("P1", upper|1, lower|1), # port 1 node
        ...         (upper|2, r1|1, "P2"),  # port 2 node
        ...         (lower|2, r1|2, "P3")  # port 3 noode
        ...     ]

        >>> w = Wilkinson()
        """
        n_ports = len(self.ports.keys())

        self.components = dict()
        self._frequency = None
        
        for k, v in self.cls_components.items():
            # make a copy of all network components so multiple instances of the network
            # can have different states
            self.components[k] = deepcopy(v)

        super().__init__(passive=passive, shunt=shunt, n_ports=n_ports)

        self.set_state(**state)
    
    def __getitem__(self, key) -> Component:
        """
        Get a network component
        """
        return self.components[key]
    
    def set_state(self, **kwargs):
        """
        Change the state variables of network components. Keys must be component designators, and values are
        dictionaries of variable/value keyword pairs specific to each component.
        """
        for k, v in kwargs.items():
            v = dict(value=v) if not isinstance(v, dict) else v
            self.components[k].set_state(**v)

    @property
    def state(self) -> dict:
        """
        Return state of all component variables.
        """
        return {k: v.state for k, v in self.components.items()}

    @property
    def frequency(self) -> np.ndarray:
        """
        Attempts to finds a frequency vector that doesn't require extrapolation of component data. If not found,
        raises an error.
        """
        if self._frequency is not None:
            return self._frequency
        
        lower_f, upper_f, step_f = -np.inf, np.inf, np.inf

        for c in self.components:
            # For components that would require extrapolation, set the upper and lower frequency bound
            # if the component spans a smaller frequency set than the current bounds. 
            if isinstance(c, (Component_SnP, Component_Data, Network)):
                cf = c.frequency
                lower_f = cf[0] if cf[0] > lower_f else lower_f
                upper_f = cf[-1] if cf[-1] < upper_f else upper_f

                # find the minimum frequency step size
                step_cf = np.diff(cf)[0] 
                step_f = step_cf if step_cf < step_f else step_f

        # if there are no components that require extrapolation in the network, use a default frequency vector
        if np.any(~np.isfinite([lower_f, upper_f, step_f])):
            self._frequency = np.arange(10e6, 10.01e9, 10e6)
        # if there is no overlapping region where all components have defined frequency data, raise an error
        elif lower_f > upper_f:
            raise ValueError(
                "Network components do not contain overlapping frequency vectors. Please use an explicit frequency"
                "vector to force extrapolation."
            )
        # return a frequency vector that doesn't require extrapolation
        else:
            self._frequency = np.arange(lower_f, upper_f + step_f, step_f)

        return self._frequency

    
    def plot_probe(
        self, 
        *paths : Tuple[int], 
        input_port : int = 1,
        frequency: np.ndarray = None, 
        fmt: str = "db",
        axes: plt.Axes = None, 
        tune : bool = False,
        freq_unit: str = "ghz",
        lines: List[Line2D] = None,
        **kwargs
    ) -> List[Line2D]:
        """
        Plots s-matrix or noise figure data over frequency

        Parameters
        ----------
        *paths : tuple | int
            Probe paths to plot. Each path must be a 2-tuple of probe name or network ports. 
            Valid probe names of the network can be found by calling ``evaluate`` and looking at the coords of the "b" 
            dimension. The voltage wave leaving the first probe is referenced to the wave leaving the
            second probe (or network port). For example, ``("ms2|1", "c2|2")`` would plot the ratio of the voltage wave 
            leaving port 1 of ``ms2`` to the voltage wave leaving port 2 of `c2`. 
        input_port : int, default: 1
            Sets which network ports is excited for all paths.
            Probes measure a different voltage wave depending on which port of the network is excited with a signal.
        frequency : np.ndarray, optional
            frequencies [Hz] to plot data over. If not provided, attempts to find a default frequency vector that 
            minimizes extrapolation of component data.
        fmt : str, default: "db"
            data format for y-axis data. Accepts the following values
            - "mag": Magnitude
            - "db" : 20log of magnitude 
            - "ang" : Phase angle
            - "ang_unwrap": Unwrapped phase angle
            - "vswr" : Voltage standing wave ratio
            - "real" : Real part of the complex s-matrix data
            - "imag" : Imaginary part of the complex s-matrix data
            - "realz" : Real part of the port input impedance
            - "imagz" : Imaginary part of the port input impedance
            - "nf" : Noise figure
        axes : matplotlib.Axes, optional
            Axes object to plot data on. If not provided, an axes is created with the default figure size. 
        tune : bool, optional
            If true, adds the plot as a tuning plot.
        freq_unit : {"Hz", "kHz", "MHz", "GHz"}, default: "GHz"
            Unit for frequency axis. 
        lines : list[Line2D], optional
            Line2D objects for each path. If provided, updates the existing lines instead of drawing new ones on the 
            plot.
        **kwargs
            parameters passed into :meth:`matplotlib.axes.plot`.

        Returns
        -------
        lines : list[Line2D]
            list of line objects that were created for each path. If ``lines`` parameter was used, returned lines
            are the same as the ``lines`` parameter.
        """
        ext_paths = []
        ref_paths = []
        labels = []

        for p in paths:

            if not (isinstance(p, (tuple, list)) and len(p) == 2):
                raise ValueError("Each path must be a tuple of length 2.")
            
            ext_paths += [(p[0], input_port)]
            ref_paths += [(p[1], input_port)]
            
            labels += [r"{}({}, {})$_{{{}}}$".format(plots.fmt_prefix[fmt], p[0], p[1], input_port)]
            
        return self.plot(
            *ext_paths, 
            frequency=frequency, 
            fmt=fmt, 
            axes=axes, 
            ref=ref_paths, 
            tune=tune,
            freq_unit=freq_unit,
            lines=lines,
            label=labels, 
            label_mode="override",  
            **kwargs
        )

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        """
        Returns the s-matrix matrix of the network.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data over, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        """
        return self.evaluate_data(frequency, noise=False)[0]

    def evaluate_data(self, frequency: np.ndarray, noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns s-matrix and noise correlation matrix of the network.

        Parameters
        ----------
        frequency : np.ndarray
            vector of frequency values to evaluate data over, in Hz.

        Returns
        -------
        sdata : np.ndarray
            MxNxN s-matrix where M is the number of frequency values and N is the number of ports. 
        ndata : np.ndarray
            MxNxN noise correlation matrix where M is the number of frequency values and N is the number of ports. 
        """
        component_data = {}
        
        # call evaluate on each of the components in the network
        for designator, comp in self.components.items():
            
            eq_designator = None
            # check if there is a component with identical sdata
            for designator_s in component_data.keys():
                # point to the sdata of an existing component if the component sdata matches (avoid evaluating twice)
                if self.components[designator_s].equals(comp):
                    eq_designator = designator_s
                    break
                
            if eq_designator:
                component_data[designator] = component_data[eq_designator]

            else:
                # evaluate the sdata at the network frequency points
                component_data[designator] = comp.evaluate(frequency, noise=noise) 

        # copy the netlist as it will be modified during the connections
        netlist = deepcopy(self.netlist)
        probe_netlist = deepcopy(self.probe_netlist)

        all_nodes = nlist.get_all_nodes(netlist)

        # component data is a dictionary with keys s, and n. If n is present and not None, connect_node should
        # compute the ndata as well as the sdata, updating the dictionaries for both.

        # connect all ground nodes first
        if 0 in all_nodes:
            node_is_external = 0 in self.ports.values()
            nlist.connect_node(component_data, netlist, probe_netlist, 0, node_is_external, noise=noise)

        # terminate open circuited ports
        if -1 in all_nodes:
            node_is_external = -1 in self.ports.values()
            nlist.connect_node(component_data, netlist, probe_netlist, -1, node_is_external, noise=noise)

        # update node list with 0 and -1 removed
        remaining_nodes = nlist.get_all_nodes(netlist)

        max_connections = 10_000
        for i in range(max_connections):

            min_cas_ports = np.inf
            min_node = np.nan

            # find the node connection that results in the fewest external ports
            for node in remaining_nodes:
                # get the port number from each component connected to the current node
                node_ports = nlist.get_node_ports(netlist, node)
                # sum of all ports from all components that are connected to this node
                n_comp_ports = sum([len(netlist[k]) for k in node_ports.keys()])
                # total number of ports connected to this node
                node_num_ports = sum([len(ports) for c, ports in node_ports.items()])

                # number of resulting ports after the connection. 
                # TODO:  include probes in this calculation
                num_cas_ports = n_comp_ports - node_num_ports

                if num_cas_ports < min_cas_ports:
                    min_cas_ports = num_cas_ports
                    min_node = node

            if not np.isfinite(min_node):
                break

            # remove this node from the list
            remaining_nodes = remaining_nodes[remaining_nodes != min_node]

            # make the connection
            node_is_external = min_node in self.ports.values()
            nlist.connect_node(component_data, netlist, probe_netlist, min_node, node_is_external, noise=noise)

        if i >= (max_connections - 1):
            raise RuntimeError("Reached maximum number of connections.")

        # after connecting all the nodes, we should have only one component left in the netlist which
        # is the network sdata. However, if there are isolated regions in the network, we can 
        # end up with multiple components. 
        if len(netlist.keys()) > 1:
            raise RuntimeError(
                "Network has isolated components or sub-networks. Please separate into multiple networks"
            )

        # the remaining component in the netlist is the network data and port list
        network_comp = list(netlist.keys())[0]
        # sdata for the network
        sdata = component_data[network_comp]["s"]

        # list of node numbers that each network port is assigned to. Order is more or less random at this point.
        port_nodes = netlist[network_comp]
        # list of probe names for each network port
        probe_names = probe_netlist[network_comp]

        s_b, s_a = sdata.shape[-2:]
        # list of node numbers that put the ports in the correct order
        port_nodes_ordered = [self.ports[f"P{p}"] for p in range(1, s_a + 1)]
        # indices of the current ports that put them in order
        port_node_idx = [port_nodes.index(node) for node in port_nodes_ordered]

        # indices of the probe names that put them in order
        probe_names_idx = [probe_names.index(name) for name in self.probe_names]

        # reorder the sdata rows to match the desired port order. leave the probe order as is
        row_order = port_node_idx + probe_names_idx
        sdata = sdata[:, np.array(row_order), :]
        # reorder the columns
        sdata[..., :] = sdata[..., np.array(port_node_idx)]

        # reorder ports on noise data
        if noise:
            ndata = component_data[network_comp]["n"]

            # reorder the ndata rows to match the desired port order
            ndata = ndata[:, np.array(port_node_idx), :]
            # reorder the columns
            ndata[..., :] = ndata[..., np.array(port_node_idx)]
        else:
            ndata = None
        
        # label the probe rows if present
        if s_b > s_a:
            a_ports = [a for a in range(1, s_a + 1)]
            b_names = a_ports + self.probe_names
            sdata = ldarray(
                sdata, coords=dict(frequency=frequency, b=b_names, a=a_ports)
            )

        # return the square matrix sdata (excluding probes), and the noise data
        return sdata, ndata


class DynamicNetwork(Network):
    """
    Network that allows the netlist to be defined at runtime instead of statically declared.
    """
    def __init__(self, 
        components: dict, 
        nodes: list = list(), 
        cascades: list = list(), 
        probes: dict = dict(),
        shunt: bool = False, 
        passive: bool = False, 
        state: dict = dict()
    ):
        """
        Parameters
        ----------
        components : dict
            dictionary of Component objects where the keys are the reference designators.
        nodes : list
            list of tuples, where each tuple is a group of component ports that are connected into a single node.
        cascades : list
            list of tuples, where each tuple is a group of components that are connected end to end, port 2 to port 1.
        probes: dict
            dictionary of probe names to the component port they attach to. Probe voltage waves are defined as the 
            wave leaving the component from the specified port. For example, `dict(probe1=c1|1)`, would attach a probe
            to port 1 of the "c1" component. Probe names will appear in the coords of the "b" dimension of s-matrix
            data returned by evaluate. 
        shunt : bool, default: False
            If True, port 2 is connected to ground and port 1 is transformed into a 2-port component that can be 
            cascaded with other components.
        passive : bool, default: True
            if True, ``evaluate`` calls ``evaluate_sdata`` instead of ``evaluate_data`` and noise correlation
            matrix is computed passively.
        state : dict, optional
            dictionary of state variables specific to each component. Keys must be component designators.
            The state values can be read with the ``state`` property and changed later with `set_state()`. Attempting
            to set the state of variables that were not included in the initial dictionary will raise an error.
        """
        ports, netlist, probe_netlist, probe_names = network_assemble(components, nodes, cascades, probes)

        self.ports = ports
        self.netlist = netlist
        self.probe_netlist = probe_netlist
        self.probe_names = probe_names
        self.cls_components = components
        self.probes = probes
        
        super().__init__(shunt=shunt, passive=passive, state=state)

    