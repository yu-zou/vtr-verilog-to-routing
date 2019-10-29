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
    return nx.read_edgelist(mdg_checkpoint, create_using = nx.Graph, edgetype = float)

class RRGraph:
    # Input: device layout XML, placement_delta_delay_lookup, and metric distance graph checkpoint file
    # Output: a metric distance graph describing shortest delay between each pair of physical sites
    def __init__(self, deviceXML, placement_delay_lookup_file, mdg_checkpoint):
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
            print('Found an existing MDG checkpoint, loading it', flush = True)
            # if MDG is already generated before, it's unnecessary to generate that
            # directly load it
            MDG = loadMDG(mdg_checkpoint)
        else:# generate a new MDG and store it to mdg_file
            print('Did not find an existing MDG checkpoint, generating it', flush = True)

            # generate a delta delay lookup dictionary
            delta_delay_lookup_dict = dict()
            with open(placement_delay_lookup_file, 'r') as f:
                next(f)
                for line in f:
                    tokens = line.split()
                    delta_y = int(tokens[0])
                    for delta_x in range(len(tokens)-1):
                        delta_delay_lookup_dict[(delta_x, delta_y)] = float(tokens[delta_x+1]) * 1e9

            # create metric distance graph
            MDG = nx.Graph()
            for (source_x, source_y) in [(x, y) for x in range(W) for y in range(H)]:
                if (source_x, source_y) not in [(0, 0), (0, H-1), (W-1, 0), (W-1, H-1)]:
                    for (sink_x, sink_y) in [(x, y) for x in range(W) for y in range(H)]:
                        if (sink_x, sink_y) not in [(0, 0), (0, H-1), (W-1, 0), (W-1, H-1)]:
                            delta_x, delta_y = abs(source_x - sink_x), abs(source_y - sink_y)
                            delay = delta_delay_lookup_dict[(delta_x, delta_y)]
                            source_vertex, sink_vertex = 'BLK_%d_%d' % (source_x, source_y), 'BLK_%d_%d' % (sink_x, sink_y)
                            MDG.add_edge(source_vertex, sink_vertex, weight = delay)

            # Save MDG checkpoint
            saveMDG(MDG, mdg_checkpoint)

        # Construct two lists, one for IO, one for CLB
        # Each list contains all physical locations compatible for the list type
        # e.g. IO_sites list all physical locations which IO blocks in netlist can sit
        # currently assuming each physical location can be only compatible with one kind of block in netlist
        self.IO_sites = List()
        self.CLB_sites = List()
        for (idx, node) in enumerate(MDG):
            x, y = int(re.findall(r'\d+', node)[0]), int(re.findall(r'\d+', node)[1])
            if x == 0 or x == W-1 or y == 0 or y == H-1:# an IO site because it's at perimeter
                self.IO_sites.append(idx)
            else:# an CLB site
                self.CLB_sites.append(idx)

        self.nodes = list(MDG)
        self.MDM = nx.adjacency_matrix(MDG).todense()
        self.W = W
        self.H = H
