import xml.etree.ElementTree as ET
from sys import exit
import os
import networkx as nx
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import re
from numba.typed import List

def saveMDG(MDG, mdg_checkpoint):
    nx.write_edgelist(MDG, mdg_checkpoint, data = True)

def loadMDG(mdg_checkpoint):
    return nx.read_edgelist(mdg_checkpoint, create_using = nx.DiGraph, edgetype = np.uint8)

class RRGraph:
    # Input: device layout XML, routing resource graph XML, and MDG_Checkpoint file
    # Output: a metric distance graph describing connections between physical sites
    def __init__(self, deviceXML, rrgraphXML, mdg_checkpoint):
        # parse dimensions of device layout
        # current implementation assumes square layout (W = H)
        # which is the common case in current commercial FPGA devices
        # and also assume corners of the layout are empty
        # and perimeters of the layout are IO
        root = ET.parse(deviceXML).getroot()
        device_layout = root[2]
        if device_layout[0].tag != 'fixed_layout':
            print('GraphPlace can only work with fixed layout architecture XML file')
            exit(0)
        W, H = int(device_layout[0].attrib['width']), int(device_layout[0].attrib['height'])
        if W != H:
            print('GraphPlace can only work with square layout')
            exit(0)
        
        if (os.path.exists(mdg_checkpoint)):
            print('Found an existing MDG checkpoint, loading it', flush=True)
            # if MDG is already generated before, it's unnecessary to generate that
            # directly load it
            MDG = loadMDG(mdg_checkpoint)
        else:# generate a new MDG and store it to mdg_file
            print('Did not find an existing MDG checkpoint, generating it', flush=True)
            
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
            for (src_x, src_y) in tqdm([(x0, y0) for x0 in range(W) for y0 in range(H)], desc='Creating Metric Distance Graph'):
                if (src_x, src_y) != (0, H-1) and (src_x, src_y) != (0, 0) and (src_x, src_y) != (W-1, H-1) and (src_x, src_y) != (W-1, 0):
                    src_vertex = 'BLK_%d_%d' % (src_x, src_y)
                    shortest_lengths = nx.single_source_shortest_path_length(RRG, src_vertex)
                    for (dst_x, dst_y) in [(x1, y1) for x1 in range(W) for y1 in range(H)]:
                        if (dst_x, dst_y) != (0, H-1) and (dst_x, dst_y) != (0, 0) and (dst_x, dst_y) != (W-1, H-1) and (dst_x, dst_y) != (W-1, 0):
                            dst_vertex = 'BLK_%d_%d' % (dst_x, dst_y)
                            if src_vertex == dst_vertex:
                                MDG.add_edge(src_vertex, dst_vertex, weight=np.uint8(0))
                            else:
                                MDG.add_edge(src_vertex, dst_vertex, weight=np.uint8(shortest_lengths[dst_vertex]-2))

            # Save MDG checkpoint
            saveMDG(MDG, mdg_checkpoint)

        # Construct two lists, one for IO, one for CLB
        # Each list contains all physical locations compatible for the list type
        # e.g. IO_sites list all physical locations which IO blocks in netlist can sit
        # currently assuming each physical location can be only compatible with one kind of block in netlist
        self.IO_sites = List()
        self.CLB_sites = List()
        # for (idx, node) in enumerate(MDG.nodes()):
        for (idx, node) in enumerate(MDG):
            x, y = int(re.findall(r'\d+', node)[0]), int(re.findall(r'\d+', node)[1])
            if x == 0 or x == W-1 or y == 0 or y == H-1:# an IO site because it's at perimeter
                self.IO_sites.append(idx)
            else:# an CLB site
                self.CLB_sites.append(idx)

        self.nodes = list(MDG)
        self.MDM = nx.adjacency_matrix(MDG).todense()
