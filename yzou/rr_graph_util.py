import xml.etree.ElementTree as ET
from sys import exit
import networkx as nx
import numpy as np
from tqdm import tqdm
from pyqubo import Binary, Spin, Array, Placeholder, Constraint, solve_qubo
from pprint import pprint
import re
import argparse
import neal# D-Wave simulated annealing sampler
import dimod

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
        if (src_x, src_y) != (0, 0) and (src_x, src_y) != (0, H-1) and (src_x, src_y) != (W-1, 0) and (src_x, src_y) != (W-1, H-1):# corners are empty
            for (dst_x, dst_y) in [(x1, y1) for x1 in range(W) for y1 in range(H)]:
                if (dst_x, dst_y) != (0, 0) and (dst_x, dst_y) != (0, H-1) and (dst_x, dst_y) != (W-1, 0) and (dst_x, dst_y) != (W-1, H-1):# corners are empty
                    if (src_x, src_y) == (dst_x, dst_y):
                        src_vertex = 'BLK_%d_%d' % (src_x, src_y)
                        dst_vertex = 'BLK_%d_%d' % (dst_x, dst_y)
                        MDG.add_edge(src_vertex, dst_vertex, weight = 0)
                    else:
                        src_vertex = 'BLK_%d_%d' % (src_x, src_y)
                        dst_vertex = 'BLK_%d_%d' % (dst_x, dst_y)
                        MDG.add_edge(src_vertex, dst_vertex, weight = nx.shortest_path_length(RRG, source = src_vertex, target = dst_vertex) - 2)

    # Construct two lists, one for IO, one for CLB
    # Each list contains all physical locations compatible for the list type
    # e.g. IO_sites list all physical locations which IO blocks in netlist can sit
    # currently assuming only each physical location can be only compatible with one kind of block in netlist
    IO_sites = list()
    CLB_sites = list()
    for (idx, node) in enumerate(MDG.nodes()):
        x, y = int(re.findall(r'\d+', node)[0]), int(re.findall(r'\d+', node)[1])
        if x == 0 or x == W-1 or y == 0 or y == H-1:# an IO site because it's at perimeter
            IO_sites.append(idx)
        else:# an CLB site
            CLB_sites.append(idx)

    return MDG, IO_sites, CLB_sites

# # Input: a packed netlist XML
# # Output: a netlist graph
# # At current stage, clock signal is ignored as it seems VPR
# # treats clock separately from other signals, we will look back at this
# # in the future
# def NetListGraph(netlistXML):
    # # create a netlist graph
    # NG = nx.DiGraph()
    # root = ET.parse(netlistXML).getroot()
    # for block in tqdm(root[3:], desc = 'Creating Netlist Graph'):
        # # if block.attrib['instance'].find('clb', 0, 3) != -1:# io blocks are skipped
        # for iport in block[0][0].text.split():
            # if iport != 'open':
                # NG.add_edge(iport, block.attrib['name'])
    # print(NG.nodes())
    # print(NG.edges())
    # exit(0)

    # # Construct two lists, one for IO, one for CLB
    # # Each list contains all blocks compatible for the physical location type
    # # e.g. IO_blocks list all blocks which can sit in IO sites
    # IO_blocks = list()
    # CLB_blocks = list()
    # for (idx, node) in enumerate(NG.nodes()):
        # if re.match(r'n\d+', node):
            # # CLB
            # CLB_blocks.append(idx)
        # else:# IO
            # IO_blocks.append(idx)

    # return NG, IO_blocks, CLB_blocks

# For the simplicity of coding, the netlist graph is generated
# from the vpr.out generated by VTR
# VTR is already modified to write out each edge to vpr.out file
# Each edge is in the format of: "Edge: <src>-><dst>"
# So this function parses the vpr.out to extract all the edges and construct the netlist graph
# To get the vpr.out, a VTR pass has to run and remember to click "Toggle Net" in graphics
def NetListGraph(netlistXML):
    # create a netlist graph
    NG = nx.DiGraph()
    with open(netlistXML, 'r') as f:
        for line in f:
            if re.match(r'Edge: .+->.+', line[:-1]):
                src, dst = line[6:-1].split(r'->')
                NG.add_edge(src, dst)

    if NG.number_of_nodes() == 0:
        print('Error print run VPR once with graphics enabled')
        exit(0)
    
    # Construct two lists, one for IO, one for CLB
    # Each list contains all blocks compatible for the physical location type
    # e.g. IO_blocks list all blocks which can sit in IO sites
    IO_blocks = list()
    CLB_blocks = list()
    for (idx, node) in enumerate(NG.nodes()):
        if re.match(r'n\d+', node):
            # CLB
            CLB_blocks.append(idx)
        else:# IO
            IO_blocks.append(idx)

    return NG, IO_blocks, CLB_blocks

# NM: netlist matrix
# MDM: metric distance matrix
def GraphEmbedding(NM, MDM, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
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
    for i in tqdm(range(N), desc = 'Generating implicit row constraints'): # the sum of each row should equal 1, which means every element is mapped
        C = 0
        label = 'implicit row: '
        for j in range(M):
            C += X[i, j]
            label += 'X[%d, %d] ' % (i, j)
        constr += P_i * Constraint((C - 1) ** 2, label = label)
    for j in tqdm(range(M), desc = 'Generating implicit col constraints'): # the sum of each column should equal 1, which means every physical location is mapped
        C = 0
        label = 'implicit col: '
        for i in range(N):
            C += X[i, j]
            label += 'X[%d, %d] ' % (i, j)
        constr += P_i * Constraint((C - 1) ** 2, label = label)
    
    # generate explicit constraints
    P_e = Placeholder('P_e')# strength of explicit constraint
    for (i, j) in tqdm([(io_block, clb_site) for io_block in IO_blocks for clb_site in CLB_sites], desc = 'Generating explicit constraints'):
        constr += P_e * explicit_constraint_convert(i, j, X)
    for (i, j) in tqdm([(clb_block, io_site) for clb_block in CLB_blocks for io_site in IO_sites], desc = 'Generating explicit constraints'):
        constr += P_e * explicit_constraint_convert(i, j, X)
    
    # calculate QUBO parameters
    feed_dict = {'P_i': 2000000, 'P_e': 2000000}
    H = obj + constr
    model = H.compile()
    qubo, _ = model.to_qubo(feed_dict = feed_dict)

    # solve QUBO
    # raw_solution = solve_qubo(qubo, num_reads = 100, sweeps = 2000)
    sampler = neal.SimulatedAnnealingSampler()
    raw_solution = sampler.sample_qubo(qubo, num_reads = 100, num_sweeps = 5000).first.sample
    # raw_solution = dimod.ExactSolver().sample_qubo(qubo).first().sample

    # interpret results
    sol = np.zeros([N, M])
    for (key, val) in tqdm(raw_solution.items(), desc = 'Interpreting result'):
        row, col = int(re.findall(r'\d+', key)[0]), int(re.findall(r'\d+', key)[1])
        sol[row, col] = val
    print('Embedding Matrix:')
    pprint(sol)

    # decode solution and check broken constraints
    _, broken, energy = model.decode_solution(raw_solution, vartype = 'BINARY', feed_dict = feed_dict)
    print('Energy: ', energy)
    print('Broken rules:')
    pprint(broken)

def main():

    parser = argparse.ArgumentParser(description = 'GraphPlace')
    parser.add_argument('deviceXML', type = str, help = 'Device architecture XML file')
    parser.add_argument('rrgraphXML', type = str, help = 'Routing resource graph XML file')
    parser.add_argument('netlist', type = str, help = 'vpr.out which exported edges')

    deviceXML = '../temp/fixed_k4_N1_W4_90nm.xml'
    rrgraphXML = '../temp/rr_graph.xml'
    netlistXML = '../temp/vpr.out'
    
    MDG, IO_sites, CLB_sites = GlobalRoutingGraph(deviceXML = deviceXML, rrgraphXML = rrgraphXML)
    NG, IO_blocks, CLB_blocks = NetListGraph(netlistXML)

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
    
    GraphEmbedding(NM = expanded_NM, MDM = MDM,  IO_blocks = IO_blocks, CLB_blocks = CLB_blocks, IO_sites = IO_sites, CLB_sites = CLB_sites)

if __name__ == '__main__':
    main()
