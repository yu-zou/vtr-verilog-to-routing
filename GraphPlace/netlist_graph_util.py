from sys import exit
import networkx as nx
from scipy.sparse import csr_matrix
from numba.typed import List

class NetlistGraph:
    # The netlist graph is generated
    # from the netlist_edges generated by VTR
    # VTR is already modified to write out each edge to netlist_edges file
    # Each edge is in the format of: "<src type> <src> <dst type> <dst>"
    # So this function parses the netlist_edges to extract all the edges and construct the netlist graph
    # To get the netlist_edges, a VTR pass has to run first
    def __init__(self, netlist_edges):
        # create a netlist graph
        NG = nx.DiGraph()
        with open(netlist_edges, 'r') as f:
            for line in f:
                src_type, src, dst_type, dst = line.split()
                NG.add_node(src, type = src_type)
                NG.add_node(dst, type = dst_type)
                NG.add_edge(src, dst, weight = 1)

        # Construct two lists, one for IO, one for CLB
        # Each list contains all blocks compatible for the physical location type
        # e.g. IO_blocks list all blocks which can sit in IO sites
        self.IO_blocks, self.CLB_blocks = List(), List()
        for (idx, node) in enumerate(NG.nodes(data = True)):
            if node[1]['type'] == 'clb':
                # CLB
                self.CLB_blocks.append(idx)
            else:# IO
                self.IO_blocks.append(idx)

        if NG.number_of_nodes() == 0:
            print('Error: please run VPR once')
            exit(0)

        self.nodes = list(NG)
        self.NM = nx.adjacency_matrix(NG).todense()
