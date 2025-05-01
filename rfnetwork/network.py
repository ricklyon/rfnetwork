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

from typing import Tuple


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

        # compile netlist and network port name to node mapping
        netlist, ports = nlist.build_netlist(nodes, cascades, components)

        # similar to the normal netlist, but each value is either None for no probe, or a probe name in the
        # index for the port it's assigned to. Probes the voltage wave leaving the port.
        probe_netlist = nlist.build_probe_netlist(components, probes, netlist)

        # add _cls_blocks and declared_ports as a class member and call type.__new__
        classdict["cls_components"] = components
        classdict["netlist"] = netlist
        classdict["probe_netlist"] = probe_netlist
        classdict["ports"] = ports
        classdict["probes"] = probes

        ncls = super().__new__(metacls, cls, bases, classdict)
        return ncls


class Network(Component, metaclass=NetworkMeta):

    def __init__(self, shunt: bool = False, passive: bool = False, **kwargs):
        nports = len(self.ports.keys())

        self.components = dict()
        
        for k, v in self.cls_components.items():
            # make a copy of all network components so multiple instances of the network
            # can have different states
            self.components[k] = deepcopy(v)

        self.set_state(**kwargs)
        self._probe_data = None

        super().__init__(passive=passive, shunt=shunt, pnum=nports)
                 
    def equals(self, other):
        # TODO: check active states of all components in network and make sure they are equal.
        return False

    def __call__(self, **kwargs):
        # simple syntax for duplicating components in Network declarations
        nobj = deepcopy(self)
        nobj.set_state(**kwargs)
        return nobj
    
    def set_state(self, **kwargs):

        for k, v in kwargs.items():
            if isinstance(v, dict):
                self.components[k].set_state(**v)
            else:
                self.components[k].set_state(v)

    @property
    def state(self):
        return {k: v.state for k, v in self.components.items() if hasattr(v, "state") and v.state != "default"}

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
            nlist.connect_node(component_data, netlist, probe_netlist, ports=self.ports, node=0, noise=noise)

        # terminate open circuited ports
        if -1 in all_nodes:
            nlist.connect_node(component_data, netlist, probe_netlist, ports=self.ports, node=-1, noise=noise)

        # update node list with 0 and -1 removed
        remaining_nodes = nlist.get_all_nodes(netlist)

        max_connections = 10_000
        for i in range(max_connections):

            min_cas_ports = np.inf
            min_node = -1

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

            if min_node < 0:
                break

            # remove this node from the list
            remaining_nodes = remaining_nodes[remaining_nodes != min_node]

            # make the connection
            node_is_external = min_node in self.ports.values()
            nlist.connect_node(component_data, netlist, probe_netlist, min_node, node_is_external, noise=noise)

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

        # probe names in the order given in the dictionary
        probe_names_ordered = list(self.probes.keys())
        # indices of the probe names that put them in order
        probe_names_idx = [probe_names.index(name) for name in probe_names_ordered]

        # reorder the sdata rows to match the desired port order
        sdata[:] = sdata[:, np.array(port_node_idx + probe_names_idx), :]
        # reorder the columns
        sdata[..., :] = sdata[..., np.array(port_node_idx)]

        # reorder ports on noise data
        if noise:
            ndata = component_data[network_comp]["n"]

            # reorder the ndata rows to match the desired port order
            ndata[:] = ndata[:, np.array(port_node_idx), :]
            # reorder the columns
            ndata[..., :] = ndata[..., np.array(port_node_idx)]
        else:
            ndata = None
        
        # separate the probe matrix and save to a cached variable
        if s_b > s_a:
            self._probe_data = ldarray(
                sdata[:, s_a:], dim=dict(frequency=frequency, b=probe_names_ordered, a=np.arange(1, s_a + 1))
            )
        else:
            self._probe_data = None

        # return the square matrix sdata (excluding probes), and the noise data
        return sdata[:, :s_a], ndata
        


    