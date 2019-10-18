import rr_graph_util
import netlist_graph_util
import argparse
from pprint import pprint
from sys import getsizeof, exit
import graph_embedding
from numba.typed import List
import numpy as np
from scipy.sparse import coo_matrix
import re
import os
import logging

def main():
    parser = argparse.ArgumentParser(description = 'GraphPlace')
    parser.add_argument('deviceXML', type = str, help = 'Device architecture XML file')
    parser.add_argument('rrgraphXML', type = str, help = 'Routing resource graph XML file')
    parser.add_argument('netlist_edges', type = str, help = 'netlist_edges where edges are exported')
    parser.add_argument('mdg_checkpoint', type = str, help = 'MDG checkpoint file path, will be created if not exists')
    parser.add_argument('placement_file', type = str, help = 'File path for the result placement')
    parser.add_argument('vpr_placement', type = str, help = 'VPR generate baseline placement')
    parser.add_argument('--log', type = str, default = 'WARNING', help = 'log level setting, DEBUG, INFO, WARNING, ERROR, CRITICAL')
    
    args = parser.parse_args()

    # warmup
    graph_embedding.QUBOConstruct_warmup()

    deviceXML = args.deviceXML
    rrgraphXML = args.rrgraphXML
    netlist_edges = args.netlist_edges
    mdg_checkpoint = args.mdg_checkpoint
    placement_file = args.placement_file
    vpr_placement = args.vpr_placement

    # set log level
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log.upper())
    logging.basicConfig(level = numeric_level)

    RRGraph = rr_graph_util.RRGraph(deviceXML, rrgraphXML, mdg_checkpoint)
    # print('RRGraph:')
    # pprint(RRGraph.MDM)
    # print(RRGraph.nodes, RRGraph.IO_sites, RRGraph.CLB_sites)

    NGraph = netlist_graph_util.NetlistGraph(netlist_edges)
    # print('NGraph:')
    # pprint(NGraph.NM)
    # print(NGraph.nodes, NGraph.IO_blocks, NGraph.CLB_blocks)

    expanded_NM = np.pad(NGraph.NM, ((0, RRGraph.MDM.shape[0] - NGraph.NM.shape[0]),), 'constant', constant_values = np.uint8(0))
    # print('Expanded NGraph:')
    # pprint(expanded_NM)

    coo_qubo, offset = graph_embedding.QUBOConstruct(expanded_NM, RRGraph.MDM, 200, 200, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites)
    # pprint(coo_qubo.todense())
    print('Offset:', offset)

    csr_sol = graph_embedding.QUBOSolve_SA(coo_qubo, offset, RRGraph.MDM.shape[0])
    # print('Solution:')
    # pprint(csr_sol.todense())
    # print(csr_sol)

    print('# rules violated: ', graph_embedding.QUBOSolValid(csr_sol, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites))
    
    # write out placement results to a file
    coo_sol = csr_sol.tocoo()
    with open(placement_file, 'w') as f:
        # f.write('Netlist File: xxx.net Netlist_ID: SHA256:36d6e7fbb60f073466b5f86dd331f687581d2fb58a465dc3ce0dcbf00ef037f8' + os.linesep)
        # f.write('Array size: %d x %d logic blocks\n' % (RRGraph.MDM.shape[0], RRGraph.MDM.shape[1]))
        f.write('#block name x\ty\tsubblk\tblock number\n')
        f.write('#----------	--	--	------	------------\n')
        blk_number = 0
        for (row, col) in zip(coo_sol.row, coo_sol.col):
            if (row < NGraph.NM.shape[0]):
                blk_name = NGraph.nodes[row]
                [site_x, site_y] = re.findall(r'\d+', RRGraph.nodes[col])
                f.write('%s\t%s\t%s\t%d\t#%d\n' % (blk_name, site_x, site_y, 0, blk_number))
                blk_number += 1

    # evaluate baseline
    graph_embedding.QUBOVPRSolEval(vpr_placement, NGraph.nodes, RRGraph.nodes, RRGraph.MDM.shape[0], coo_qubo, offset, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites)

if __name__ == '__main__':
    main()
