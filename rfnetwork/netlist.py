from .component import Component
import numpy as np
from typing import Tuple
from . core import core


class Junction(Component):
    """
    N-port lossless junction
    """

    def __init__(self, N: int):
        """
        Parameters:
        ----------
        N: int
            number of junction ports
        """
        super().__init__(n_ports=N)
        assert N > 0, "N must be positive"
        self.N = int(N)

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return core.junction_sdata(frequency, self.N)
    

class Open(Component):
    """
    1-port open-circuit load
    """
    def __init__(self):
        super().__init__(n_ports=1)

    def evaluate_sdata(self, frequency: np.ndarray) -> np.ndarray:
        return np.full((len(frequency), 1, 1), 1, dtype="complex128")


class Short(Component):
    """
    1-port short-circuit load
    """
    def __init__(self):
        super().__init__(n_ports=1)

    def evaluate_sdata(self, frequency: np.ndarray):
        return np.full((len(frequency), 1, 1), -1, dtype="complex128")
    

def cascade_to_nodes(cascades: list) -> list:
    """
    Converts a list of cascaded components to a list of connected nodes.
    """
    nodes = []

    # find connected ports from the cascade lists and append to the nodes list
    for i, cas in enumerate(cascades):
        if len(cas) < 2:
            raise ValueError("Cascade lists must have at least 2 components.")
        
        # endpoints of a cascade can be ports or components. If components are used at endpoints, convert to ports,
        # use 2nd port by default of first component, and 1st port of last component.
        cas = list(cas)
        first_port = cas[0]|2 if isinstance(cas[0], Component) else cas[0]
        nodes.append((first_port, cas[1]|1))

        for j in range(2, len(cas)-1):
            # connect the second port from the last component to the first port of the current one,
            # if the prev component is the first item in the list, use the port directly
            nodes.append((cas[j-1]|2, cas[j]|1))

        # last item can be port or component
        last_port = cas[-1]|1 if isinstance(cas[-1], Component) else cas[-1]
        nodes.append((cas[-2]|2, last_port))

    return nodes

def convert_to_refdes(nodes: list, components: dict) -> Tuple[list, dict]:
    """
    Parses a list of nodes and returns a dictionary of reference designators with the associated component object,
    and a new list of nodes that uses the reference designators instead of object pointers.

    If a component pointer is not found in the component dictionary, it automatically assigns a designator.
    """

    # counter for designators assigned automatically
    auto_designator_counter = 1

    # lists of the declared component objects and their reference designators, used to get the designator given an
    # object.
    refdes_lookup_by_obj = {id(v): k for k,v in components.items()}

    # initialize return values
    refdes = dict()
    nodes_refdes = []

    for i, node in enumerate(nodes):

        node_refdes = []
        # iterate over each port in the node list, skipping the node names
        for item in node:
            if isinstance(item, str):
                node_refdes.append(item)
                continue
            
            # each node item should be a component pointer and a port number
            obj, port = item

            # If component is declared in the class, get the designator
            if id(obj) in refdes_lookup_by_obj.keys():
                designator = refdes_lookup_by_obj[id(obj)]
            # otherwise assign a designator automatically
            else:
                designator = "u__{:04}".format(auto_designator_counter)
                auto_designator_counter += 1
                # add the designator to the lookup table so duplicates are avoided
                refdes_lookup_by_obj[id(obj)] = designator

            # add designator to component dictionary
            if designator not in refdes.keys():
                refdes[designator] = obj

            # add object port pair to the current node
            node_refdes.append((designator, port))
        
        # add this node to the list of nodes
        nodes_refdes.append(node_refdes)

    return nodes_refdes, refdes


def build_probe_netlist(components: dict, probes: dict, netlist: dict) -> Tuple[dict, list]:
    """
    Builds a netlist of probe names that are attached to each component port. If no probe is assigned to a port,
    the port value is None.
    """
    # reversed dictionary mapping obj ids to their reference
    refdes_lookup_by_obj = {id(v): k for k,v in components.items()}
    # initialize empty netlist with None assigned to every port
    probe_netlist = {}
    for k, v in netlist.items():
        probe_netlist[k] = [None] * len(v) 
            
    # populate the probe netlist with probe names attached to the top level network
    for name, (cobj, port) in probes.items():
        # check that the object is in the component mapping
        if id(cobj) not in refdes_lookup_by_obj.keys():
            raise ValueError(f"Component not found in component list: {cobj}")
        # get the reference designator from the obj memory id
        designator = refdes_lookup_by_obj[id(cobj)]

        if designator not in probe_netlist.keys():
            raise ValueError(f"Probe component not found in netlist: {designator}")
        # populate the netlist at the port index with the probe name
        probe_netlist[designator][port - 1] = name

    # add probes from sub-networks and compile a list of all probes from all sub-networks. 
    # The order here doesn't matter because evaluate will ensure the sdata rows match whatever order is assigned here.
    probe_names = list(probes.keys())
    for k, v in probe_netlist.items():
        # Extend the probe list to include the existing probes of a component that is a another network.
        if hasattr(components[k], "probe_names"):
            subntwk_probe_names = [f"{k}.{p}" for p in components[k].probe_names]

            probe_netlist[k] += subntwk_probe_names
            probe_names += subntwk_probe_names

    return probe_netlist, probe_names


def build_netlist(nodes: list, cascades: list, components: dict) -> dict:
    """
    Builds a netlist dictionary from a list of nodes and cascades as defined in the Network class constructor.

    Parameters
    ----------
    nodes : list
        each item in the list is a list of component/port pairs that are connected to that node.
        Has the form: [(obj, port), ...] where obj is a reference to a Component, and port is the integer port number.
    cascades : list
        each item in the list is a tuple of components strung together from port 2 to port 1.
    components : dict
        dictionary of component handles. Keys are the reference designator of each component, values are object
        references.
    probes : dict
        dictionary of probe names (keys) and the (component, port) they are assigned to

    Returns
    -------
    netlist : dict
        keys are component designators, values are list of integer node numbers. Each component's list is the 
        nodes that each port is connected to.
    ports : dict
        keys are the network port names, values are the node number assigned to that port.

    """

    # convert the cascades into a list of nodes and combine with the nodes list
    nodes = nodes + cascade_to_nodes(cascades)

    # convert object handles to their associated reference designator
    nodes, components = convert_to_refdes(nodes, components)

    # initialize netlist with -1 for all ports of each component
    netlist = {}
    for k, v in components.items():
        netlist[k] = [-1] * v.n_ports

    # list of node names corresponding to each node number
    ports = dict()
    node_num_counter = 1

    for i, node in enumerate(nodes):
    
        # look for a node name
        node_name = [n for n in node if isinstance(n, str)]
        # raise error if multiple node names defined for a single node
        if len(node_name) > 1:
            raise ValueError(f"Multiple names for node {i}. Got {node_name}")
        # set the node name to the first (and only) string value found
        elif len(node_name) == 1:
            node_name = node_name[0]
        else:
            node_name = ""

        # set node number to 0 if a ground connection
        if node_name.upper() == "GND":
            node_num = 0
        # look for existing nodes with the same node name, if one is found set the node number
        elif node_name in ports.keys():
            raise ValueError(f"Multiple entries for node {node_name}")
        # generate a new node number
        else:
            node_num = node_num_counter
            # increment the node counter for the next node
            node_num_counter += 1
            # add this node to the named node map 
            if node_name:
                ports[node_name] = node_num

        # iterate over each port in the node list, 
        for item in node:
            # skipping the node names
            if isinstance(item, str):
                continue

            designator, port = item
            
            # check that component port doesn't already exist in netlist
            if netlist[designator][port-1] >= 0:
                
                raise ValueError(
                    f"Port {designator}|{port} appears in multiple nodes. Ensure ports are only defined once in nodes list."
                )

            # populate the netlist port
            netlist[designator][port - 1] = node_num

        
    return netlist, ports


def get_all_nodes(netlist):
    """
    Get an ordered list of all node numbers in the netlist
    """
    nodes = []
    for k, v in netlist.items():
        nodes += v
    return np.sort(np.unique(nodes))


def get_node_ports(netlist, node):
    """ 
    Get a list of port numbers connected to node for each component in the netlist
    """ 
    return {c: np.where(np.array(netlist[c]) == node)[0] + 1 for c, c_ports in netlist.items() if node in c_ports}


def connect_node(
    comp_data: dict, netlist: dict, probe_netlist: dict, node: int, node_is_external: False, noise: bool = False
):
    """
    Connects all ports together that are assigned to ``node`` in the netlist.

    Parameters
    ----------
    comp_data: dict
        dictionary of s-matrices for each component in the netlist. Modified in place. The new smatrix data
        for the components connected together is saved into the first component found in the netlist attached
        to the node.
    netlist: dict
        keys are component designators, values are lists of the node of each port. Modified in place.
    probe_netlist: dict
        keys are component designators, values are lists of probe names of each port. If port is not
        assigned as a probe, list item is None. Modified in place.
    node: int   
        node number to connect.
    """

    node = int(node)
    # get the ports connected to this node from each component  
    node_ports = get_node_ports(netlist, node)

    # total number of ports connected to this node
    node_num_ports = sum([len(ports) for c, ports in node_ports.items()])

    if node_num_ports < 1:
        raise RuntimeError("Node {node} not found in netlist.")

    if node in [0, -1] and node_is_external:
        raise RuntimeError("Cannot make open or ground node external.")

    # case #0: ground and open loads
    if node in [0, -1]:
        c1 = list(comp_data.keys())[0]
        # short and open only use the length of the frequency vector, they are constant over frequency.
        frequency = np.zeros(len(comp_data[c1]["s"]))
        # generate sdata for a short (node == 0) or an open load (node == -1)
        
        if node == 0:
            load = Short().evaluate(frequency, noise=noise) 
        else:
            load = Open().evaluate(frequency, noise=noise)

        # connect a 1-port load to each port in this node
        for i, (c, ports) in enumerate(node_ports.items()):
            # connect ports from last to first so the lower port numbers don't change as the higher connections
            # are made
            ports = sorted(ports, reverse=True)
            for p in ports:
                _, comp_data[c] = core.connect(comp_data[c], load, (p, 1))
                # remove the connected port from this component's port list
                netlist[c].pop(p - 1)
                # component has one less port now, decrement the internal probe numbers attached to this component.
                # ground and loads cannot be assigned as probe ports and will be removed from the probe netlist
                probe_netlist[c].pop(p - 1)

    # if there is only one connecting port, skip this node, no action needed
    elif (node_num_ports == 1):
        return
    
    # case #1: node is a single connection between two ports from two different components
    elif node_num_ports == 2 and len(node_ports.keys()) == 2 and not node_is_external:
        # sdata for connected components
        c1, c2 = node_ports.keys()
        s1, s2 = comp_data[c1], comp_data[c2]
        # connected port numbers
        p1, p2 = [p.item() for c, p in node_ports.items()]

        # put the cascaded smatrix into c1. c2 is absorbed into c1.
        is_p1_probe = probe_netlist[c1][p1 - 1] is not None
        is_p2_probe = probe_netlist[c2][p2 - 1] is not None
        row_order, cas_data = core.connect(s1, s2, (p1, p2), probes=(is_p1_probe, is_p2_probe))
        # update the component smatrix
        comp_data[c1] = cas_data
        comp_data.pop(c2)

        # update the netlist, c1 is a new component with the ports from c1 + c2, component c2 no longer exists.
        s1_len = len(netlist[c1])
        netlist[c1] = netlist[c1] + netlist[c2] 
        netlist.pop(c2)
        # remove the connected ports from the new component's ports
        netlist[c1].pop(s1_len + p2 - 1)
        netlist[c1].pop(p1 - 1)

        # reorder the probe netlist and add to the first component's probe list
        pnet = probe_netlist[c1] + probe_netlist[c2]
        probe_netlist[c1] = list(np.array(pnet)[row_order])
        probe_netlist.pop(c2)


    # case #2: node is a connection between two ports of the same component
    elif node_num_ports == 2 and len(node_ports.keys()) == 1 and not node_is_external:
        c1 = list(node_ports.keys())[0]
        s1 = comp_data[c1]
        p1, p2 = node_ports[c1]

        is_p1_probe = probe_netlist[c1][p1 - 1] is not None
        is_p2_probe = probe_netlist[c1][p2 - 1] is not None
        row_order, cas_data = core.connect_self(s1, (p1, p2), probes=(is_p1_probe, is_p2_probe))
        # update the component smatrix
        comp_data[c1] = cas_data

        # update netlist
        netlist[c1].pop(p2 - 1)
        netlist[c1].pop(p1 - 1)

        # append the probe rows to the end of the external rows
        probe_netlist[c1] = list(np.array(probe_netlist[c1])[row_order])

    # case #3: there are more than two ports connected to the same node. 
    # Create a junction element and connect all ports to it
    else:
        # create sdata for the junction that all ports will connect to.
        N = node_num_ports + int(node_is_external)
        c1 = list(node_ports.keys())[0]

        nfreq = comp_data[c1]["s"].shape[-3]
        s0 = Junction(N).evaluate(np.zeros(nfreq), noise=noise)
        
        s0_netlist = [node] * N
        s0_probe_netlist = [None] * N

        # iterate over each component connected to this node and the port numbers that connect to it
        for i, (c, ports) in enumerate(node_ports.items()):

            # keep the external ports in the order that the components were passed in by connecting
            # the first port of the node, and putting the node block first. The node absorbs all
            # previously connected components, so it needs to be placed first to keep 
            # previous component first in the resulting port list
            connections = np.column_stack([np.arange(1, len(ports) + 1), ports])

            is_probe = [probe_netlist[c][p - 1] is not None for p in ports]
            probes = np.column_stack([np.zeros(len(ports)), is_probe])

            row_order, s0 = core.connect(s0, comp_data[c], connections, probes=probes)

            # update s0 netlist, remove the connected ports from s0 (first N), and add the remaining unconnected
            # ports from the component.
            s0_netlist = s0_netlist[len(ports):] + [p for p in netlist[c] if p != node]

            # append the probe rows to the end of the external rows
            pnet = s0_probe_netlist + probe_netlist[c]
            s0_probe_netlist = list(np.array(pnet)[row_order])

            # remove component from netlist since it was absorbed into s0
            comp_data.pop(c)
            probe_netlist.pop(c)
            netlist.pop(c)

        # with all the connections made, update the netlist with s0. Use the name of the first component
        c1 = list(node_ports.keys())[0]
        netlist[c1] = s0_netlist
        probe_netlist[c1] = s0_probe_netlist
        comp_data[c1] = s0
