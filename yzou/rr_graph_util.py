import xml.etree.ElementTree as ET
import sys
import networkx as nx
import numpy as np
from tqdm import tqdm
from pyqubo import Binary, Spin, Array, Placeholder
from pprint import pprint
import wildqat as wq

def GlobalRoutingGraph(xmlfile, W, H):
    # create a routing resource graph
    RRG = nx.DiGraph()
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    rr_nodes = root[5]
    rr_edges = root[6]
    for rr_edge in tqdm(rr_edges, desc = 'Creating RRG'):
        src_id, dst_id = rr_edge.attrib['src_node'], rr_edge.attrib['sink_node']
        src_node, dst_node = rr_nodes[int(src_id)], rr_nodes[int(dst_id)]
        if src_node.attrib['type'] != 'SOURCE' and src_node.attrib['type'] != 'SINK' and dst_node.attrib['type'] != 'SOURCE' and dst_node.attrib['type'] != 'SINK':
            if src_node.attrib['type'] == 'OPIN' or src_node.attrib['type'] == 'IPIN':
                src_vertex = 'BLK_%s_%s' % (src_node[0].attrib['xlow'], src_node[0].attrib['ylow'])
            elif src_node.attrib['type'] == 'CHANX' or src_node.attrib['type'] == 'CHANY':
                src_vertex = '%s_%s_%s_%s' % (src_node.attrib['type'], src_node[0].attrib['xlow'], src_node[0].attrib['ylow'], src_node[0].attrib['ptc'])
            if dst_node.attrib['type'] == 'OPIN' or dst_node.attrib['type'] == 'IPIN':
                dst_vertex = 'BLK_%s_%s' % (dst_node[0].attrib['xlow'], dst_node[0].attrib['ylow'])
            elif dst_node.attrib['type'] == 'CHANX' or dst_node.attrib['type'] == 'CHANY':
                dst_vertex = '%s_%s_%s_%s' % (dst_node.attrib['type'], dst_node[0].attrib['xlow'], dst_node[0].attrib['ylow'], dst_node[0].attrib['ptc'])
            RRG.add_edge(src_vertex, dst_vertex)
    # create metric distance graph
    MDG = nx.DiGraph()
    for (src_x, src_y) in tqdm([(x0, y0) for x0 in range(W) for y0 in range(H)], desc = 'Creating MDG'):
        if (src_x, src_y) != (0, 0) and (src_x, src_y) != (0, H-1) and (src_x, src_y) != (W-1, 0) and (src_x, src_y) != (W-1, H-1):
            for (dst_x, dst_y) in [(x1, y1) for x1 in range(W) for y1 in range(H)]:
                if (dst_x, dst_y) != (0, 0) and (dst_x, dst_y) != (0, H-1) and (dst_x, dst_y) != (W-1, 0) and (dst_x, dst_y) != (W-1, H-1):
                    if (src_x, src_y) == (dst_x, dst_y):
                        src_vertex = 'BLK_%s_%s' % (src_x, src_y)
                        dst_vertex = 'BLK_%s_%s' % (dst_x, dst_y)
                        MDG.add_edge(src_vertex, dst_vertex, weight = 0)
                    else:
                        src_vertex = 'BLK_%s_%s' % (src_x, src_y)
                        dst_vertex = 'BLK_%s_%s' % (dst_x, dst_y)
                        MDG.add_edge(src_vertex, dst_vertex, weight = nx.shortest_path_length(RRG, source = src_vertex, target = dst_vertex) - 1)
    return MDG

def NetListGraph(xmlfile):
    # create a netlist graph
    NG = nx.DiGraph()
    NG.add_edge('n9', 'out:and_latch^out')
    NG.add_edge('and_latch^a_in', 'n9')
    NG.add_edge('and_latch^b_in', 'n9')
    return NG

def GraphEmbedding(NM, MDM):
    N, M = NM.shape[0], MDM.shape[0]
    # create an 2d embedding array
    x = Array.create('x', shape = (N, M), vartype = 'BINARY')
    # generate quadratic objective function
    obj = 0
    for i in range(N):
        for j in range(N):
            for k in range(M):
                for l in range(M):
                    obj += NM[i, j] * MDM[k, l] * x[i, k] * x[j, l]
    # generate constraints
    P = Placeholder('P')
    constr = 0
    for i in range(N):
        C = 0
        for j in range(M):
            C += x[i, j]
        constr += P * (C - 1) ** 2
    for j in range(M):
        C = 0
        for i in range(N):
            C += x[i, j]
        constr += P * (C - 1) ** 2
    H = obj + constr
    model = H.compile()
    weight, _2 = model.to_qubo(feed_dict = {'P': 200.0})
    # Fill QUBO matrix
    qubo = np.zeros([N, M])


def main():
    MDG = GlobalRoutingGraph('../temp/rr_graph.xml', W = 5, H = 5)
    NG = NetListGraph('../temp/and_latch.net')
    NM = np.array([
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ])
    MDM = np.array([
        [0, 8, 15],
        [8, 0, 13],
        [15, 13, 0]
    ])
    GraphEmbedding(NM, MDM)

if __name__ == '__main__':
    main()
