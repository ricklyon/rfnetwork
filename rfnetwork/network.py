"""

find the connection that results in the smallest number of output ports. Do those first. Once a component is connected,
replace them with the subnetwork block and it's new port mapping to the remaining nodes.

continue until only one subnetwork block remains.

"""
from .component import Component
from .core import core
import numpy as np
from copy import deepcopy
from . import netlist as nlist
from np_struct import ldarray
from . import plots


from typing import Tuple

def network_assemble(components: dict, nodes: list = list(), cascades: list = list(), probes: dict = dict()):
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

    def __init__(self, shunt: bool = False, passive: bool = False, state: dict = dict()):
        nports = len(self.ports.keys())

        self.components = dict()
        
        for k, v in self.cls_components.items():
            # make a copy of all network components so multiple instances of the network
            # can have different states
            self.components[k] = deepcopy(v)

        super().__init__(passive=passive, shunt=shunt, pnum=nports)

        self.set_state(**state)
    
    def __getitem__(self, key):
        return self.components[key]
    
    def set_state(self, **kwargs):

        for k, v in kwargs.items():
            v = dict(value=v) if not isinstance(v, dict) else v
            self.components[k].set_state(**v)

    @property
    def state(self):
        return {k: v.state for k, v in self.components.items()}

    def plot_probe(self, axes, frequency, *paths, input_port=1, fmt= "db", **kwargs):

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
            axes, frequency, *ext_paths, ref=ref_paths, label=labels, label_mode="override", fmt=fmt, **kwargs
        )

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return self.evaluate_data(frequency, noise=False)[0]

    def evaluate_data(self, frequency: np.ndarray, noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
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
        """
        ports, netlist, probe_netlist, probe_names = network_assemble(components, nodes, cascades, probes)

        self.ports = ports
        self.netlist = netlist
        self.probe_netlist = probe_netlist
        self.probe_names = probe_names
        self.cls_components = components
        self.probes = probes
        
        super().__init__(shunt=shunt, passive=passive, state=state)

    