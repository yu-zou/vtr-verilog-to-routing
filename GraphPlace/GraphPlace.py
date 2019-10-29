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

def main():
    parser = argparse.ArgumentParser(description = 'GraphPlace')
    parser.add_argument('deviceXML', type = str, help = 'Device architecture XML file')
    parser.add_argument('netlist_edges_criticality_file', type = str, help = 'netlist_edges_criticality where edges with criticalities are exported')
    parser.add_argument('mdg_checkpoint_file', type = str, help = 'MDG checkpoint file path, will be created if not exists')
    parser.add_argument('placement_file', type = str, help = 'File path for the result placement')
    parser.add_argument('vpr_placement_file', type = str, help = 'VPR generate baseline placement')
    parser.add_argument('delta_delay_lookup_file', type = str, help = 'File path for delta delay lookup')
    parser.add_argument('--num_reads', type = int, default = 100, help = 'SA num_reads')
    parser.add_argument('--num_sweeps', type = int, default = 1000, help = 'SA num_sweeps')
    
    args = parser.parse_args()

    # warmup
    graph_embedding.QUBOConstruct_warmup()

    deviceXML = args.deviceXML
    netlist_edges_criticality_file = args.netlist_edges_criticality_file
    mdg_checkpoint_file = args.mdg_checkpoint_file
    placement_file = args.placement_file
    delta_delay_lookup_file = args.delta_delay_lookup_file
    vpr_placement_file = args.vpr_placement_file
    num_reads = args.num_reads
    num_sweeps = args.num_sweeps

    RRGraph = rr_graph_util.RRGraph(deviceXML, delta_delay_lookup_file, mdg_checkpoint_file)
    # print('RRGraph:')
    # pprint(RRGraph.MDM)
    # print(RRGraph.nodes, RRGraph.IO_sites, RRGraph.CLB_sites)

    NGraph = netlist_graph_util.NetlistGraph(netlist_edges_criticality_file)
    # print('NGraph:')
    # pprint(NGraph.NM)
    # print(NGraph.nodes, NGraph.IO_blocks, NGraph.CLB_blocks)

    expanded_NM = np.pad(NGraph.NM, ((0, RRGraph.MDM.shape[0] - NGraph.NM.shape[0]),), 'constant', constant_values = float(0))
    # print('Expanded NGraph:')
    # pprint(expanded_NM)

    coo_qubo, offset = graph_embedding.QUBOConstruct(expanded_NM, RRGraph.MDM, 200.0, 200.0, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites)
    # print(coo_qubo)
    # print('Offset:', offset)

    csr_sol = graph_embedding.QUBOSolve_SA(coo_qubo, offset, RRGraph.MDM.shape[0], num_reads, num_sweeps)
    # print('Solution:')
    # pprint(csr_sol.todense())
    # print(csr_sol)

    print('# rules violated: ', graph_embedding.QUBOSolValid(csr_sol, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites))
    
    # write out placement results to a file
    coo_sol = csr_sol.tocoo()
    with open(placement_file, 'w') as f:
        f.write('Netlist_File: custom_place.net Netlist_ID: SHA256:67b80b0eae5fe96d6cc5c99e38a1582e69ae9eaa5879ba5ef584ccc975dba980' + os.linesep)
        f.write('Array size: %d x %d logic blocks' % (RRGraph.W, RRGraph.H) + os.linesep)
        f.write(os.linesep)
        f.write('#block name x\ty\tsubblk\tblock number' + os.linesep)
        f.write('#----------	--	--	------	------------' + os.linesep)
        blk_number = 0
        for (row, col) in zip(coo_sol.row, coo_sol.col):
            if (row < NGraph.NM.shape[0]):
                blk_name = NGraph.nodes[row]
                [site_x, site_y] = re.findall(r'\d+', RRGraph.nodes[col])
                f.write('%s\t%s\t%s\t%d\t#%d' % (blk_name, site_x, site_y, 0, blk_number) + os.linesep)
                blk_number += 1

    # evaluate baseline
    graph_embedding.QUBOVPRSolEval(vpr_placement_file, NGraph.nodes, RRGraph.nodes, RRGraph.MDM.shape[0], coo_qubo, offset, NGraph.IO_blocks, NGraph.CLB_blocks, RRGraph.IO_sites, RRGraph.CLB_sites)

if __name__ == '__main__':
    main()
