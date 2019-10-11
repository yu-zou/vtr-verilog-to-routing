import xml.etree.ElementTree as ET
from sys import exit
import networkx as nx
import numpy as np
from tqdm import tqdm
from pyqubo import Binary, Spin, Array, Placeholder, Constraint
from pprint import pprint
import re
import neal# D-Wave simulated annealing sampler

def index_convert(nrows, ncols, row, col):
    return (row * ncols + col)

def index_deconvert(nrows, ncols, idx):
    row = int(idx / ncols)
    col = int(idx % ncols)
    return (row, col)

def explicit_constraint_convert(row, col, X):
    return Constraint(X[row, col] ** 2, label = 'explicit: X[%d, %d]' % (row, col))

# Input: device layout XML and routing resource graph XML
# Output: a metric distance graph describing connections between CLBs
def GlobalRoutingGraph(deviceXML, rrgraphXML):
    # parse dimensions of device layout
    # current implementation assumes square layout (W = H)
    # which is the common case in current commercial FPGA devices
    # and also assume corners of the layout are empty
    # and perimeters of the layout are IO
    root = ET.parse(deviceXML).getroot()
    device_layout = root[2]
    W, H = int(device_layout[0].attrib['width']), int(device_layout[0].attrib['height'])
    
    # create a routing resource graph
    RRG = nx.DiGraph()
    root = ET.parse(rrgraphXML).getroot()
    rr_nodes, rr_edges = root[5], root[6]
    for rr_edge in tqdm(rr_edges, desc = 'Creating Routing Resource Graph'):
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
    for (src_x, src_y) in tqdm([(x0, y0) for x0 in range(W) for y0 in range(H)], desc = 'Creating Metric Distance Graph'):
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
                        MDG.add_edge(src_vertex, dst_vertex, weight = nx.shortest_path_length(RRG, source = src_vertex, target = dst_vertex) - 2)
    return MDG

# Input: a packed netlist XML
# Output: a netlist graph
# At current stage, clock signal is ignored as it seems VPR
# treats clock separately from other signals, we will look back at this
# in the future
def NetListGraph(netlistXML):
    # create a netlist graph
    NG = nx.DiGraph()
    NG.add_edge('n9', 'out:and_latch^out')
    NG.add_edge('and_latch^a_in', 'n9')
    NG.add_edge('and_latch^b_in', 'n9')
    return NG

# NM: netlist matrix
# MDM: metric distance matrix
def GraphEmbedding(NM, MDM):
    N, M = NM.shape[0], MDM.shape[0]
    
    # create an 2d embedding array
    X = Array.create('X', shape = (N, M), vartype = 'BINARY')
    # generate quadratic objective function
    obj = 0
    for i in tqdm(range(N), desc = 'Generating quadratic objective'):
        for j in range(N):
            obj += NM[i, j] * X[i, :].dot(MDM).dot(X[j, :].T)
    
    constr = 0
    # generate implicit constraints
    P_i = Placeholder('P_i')# strength of implicit constraint
    for i in tqdm(range(N), desc = 'Generating implicit constraints'): # the sum of each row should equal 1, which means every element is mapped
        C = 0
        label = 'implicit row: '
        for j in range(M):
            C += X[i, j]
            label += 'X[%d, %d] ' % (i, j)
        constr += P_i * Constraint((C - 1) ** 2, label = label)
    for j in tqdm(range(M), desc = 'Generating implicit constraints'): # the sum of each column should equal 1, which means every physical location is mapped
        C = 0
        label = 'implicit col: '
        for i in range(N):
            C += X[i, j]
            label += 'X[%d, %d] ' % (i, j)
        constr += P_i * Constraint((C - 1) ** 2, label = label)
    # generate explicit constraints, this should be given the highest priority
    P_e = Placeholder('P_e')# strength of explicit constraint
    constr += P_e * explicit_constraint_convert(0, 0, X)
    constr += P_e * explicit_constraint_convert(0, 1, X)
    constr += P_e * explicit_constraint_convert(0, 2, X)
    constr += P_e * explicit_constraint_convert(0, 6, X)
    constr += P_e * explicit_constraint_convert(0, 10, X)
    constr += P_e * explicit_constraint_convert(0, 11, X)
    constr += P_e * explicit_constraint_convert(1, 3, X)
    constr += P_e * explicit_constraint_convert(1, 4, X)
    constr += P_e * explicit_constraint_convert(1, 7, X)
    constr += P_e * explicit_constraint_convert(1, 8, X)
    constr += P_e * explicit_constraint_convert(2, 3, X)
    constr += P_e * explicit_constraint_convert(2, 4, X)
    constr += P_e * explicit_constraint_convert(2, 7, X)
    constr += P_e * explicit_constraint_convert(2, 8, X)
    constr += P_e * explicit_constraint_convert(3, 3, X)
    constr += P_e * explicit_constraint_convert(3, 4, X)
    constr += P_e * explicit_constraint_convert(3, 7, X)
    constr += P_e * explicit_constraint_convert(3, 8, X)
    
    # calculate QUBO parameters
    feed_dict = {'P_i': 2000000, 'P_e': 2000000}
    H = obj + constr
    model = H.compile()
    weight, _ = model.to_qubo(feed_dict = feed_dict)

    # solve QUBO
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(weight, num_reads = 100, num_sweeps = 2000)

    # interpret results
    sol = np.zeros([N, M])
    for (key, val) in tqdm(response.first.sample.items(), desc = 'Interpreting result'):
        row = int(re.findall(r'\d+', key)[0])
        col = int(re.findall(r'\d+', key)[1])
        sol[row, col] = val
    print('Embedding Matrix:')
    pprint(sol)

    # decode solution and check broken constraints
    _, broken, energy = model.decode_solution(response.first.sample, vartype = 'BINARY', feed_dict = feed_dict)
    print('Energy: ', energy)
    print('Broken rules:')
    pprint(broken)

def main():
    deviceXML = '../temp/fixed_k4_N1_90nm_yzou.xml'
    rrgraphXML = '../temp/rr_graph.xml'
    MDG = GlobalRoutingGraph(deviceXML = deviceXML, rrgraphXML = rrgraphXML)
    NG = NetListGraph('../temp/and_latch.net')
    # adjacency matrix of metric distance graph
    MDM = nx.adjacency_matrix(MDG, nodelist = MDG.nodes()).todense()
    pprint(MDG.nodes())
    pprint(MDM)
    # adjacency matrix of netlist graph
    NM = nx.adjacency_matrix(NG, nodelist = NG.nodes()).todense()
    pprint(NG.nodes())
    pprint(NM)
    # expand netlist matrix to the size of metric distance graph
    expanded_NM = np.pad(NM, ((0, MDM.shape[0] - NM.shape[0]),), 'constant', constant_values = 0)
    pprint(expanded_NM)
    GraphEmbedding(NM = expanded_NM, MDM = MDM)

if __name__ == '__main__':
    main()
