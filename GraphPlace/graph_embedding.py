from numba import jit, prange
from numba.typed import List
import numpy as np
from pyqubo import Binary, Array, Placeholder, Constraint
import re
from sys import exit
from pprint import pprint
from tqdm import tqdm, trange
import wildqat as wq
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import neal
import dimod
import time
import logging

@jit(nopython = True, cache = True, parallel = True)
def QUBOConstruct_kernel(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
    N = MDM.shape[0]
    qubo, offset = np.zeros((N**2, N**2), dtype = np.int16), 0
    
    # generate quadratic objective
    for i in prange(N):
        for j in prange(N):
            temp = NM[i, j] * MDM
            for k in prange(N):
                for l in prange(N):
                    qubo[i*N+k, j*N+l] = temp[k, l]

    # generate implicit constraints
    # each row sums up to 1
    row_constr_mat = np.full((N, N), P_i, dtype = np.int16)
    np.fill_diagonal(row_constr_mat, -P_i)
    for i in prange(N):
        qubo[i*N:i*N+N, i*N:i*N+N] += row_constr_mat
    # each col sums up to 1
    for i in prange(N):
        for j in prange(N):
            if i == j:
                qubo[i*N:i*N+N, j*N:j*N+N] += -P_i * np.eye(N, dtype = np.int16)
            else:
                qubo[i*N:i*N+N, j*N:j*N+N] += P_i * np.eye(N, dtype = np.int16)
    offset += P_i * 2 * N

    # generate explicit constraints
    for (i, j) in [(io_block, clb_site) for io_block in IO_blocks for clb_site in CLB_sites]:
        qubo[i*N+j, i*N+j] += P_e
    for (i, j) in [(clb_block, io_site) for clb_block in CLB_blocks for io_site in IO_sites]:
        qubo[i*N+j, i*N+j] += P_e

    # convert to upper triangle matrix
    for i in prange(N**2):
        for j in prange(i+1, N**2):
            qubo[i, j], qubo[j, i] = qubo[i, j] + qubo[j, i], 0

    return qubo, offset

def QUBOConstruct(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
    qubo, offset = QUBOConstruct_kernel(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites)
    return coo_matrix(qubo), offset

def QUBOSolve_SA(coo_qubo, offset, N):
    qubo_dict = dict(zip([('X[%d]' % (i), 'X[%d]' % (j)) for (i, j) in zip(coo_qubo.row, coo_qubo.col)], coo_qubo.data))
    
    sampler = neal.SimulatedAnnealingSampler()
    start = time.time()
    response = sampler.sample_qubo(qubo_dict, num_reads = 100, num_sweeps = 1000).first
    end = time.time()

    # create a solution lil matrix, since most of entries would be zero
    # and lil is friendly to change data
    # and fill it
    lil_sol = lil_matrix((N, N), dtype = np.uint8)
    for (key, value) in response.sample.items():
        if int(value) != 0:
            idx = int(re.findall(r'\d+', key)[0])
            row, col = int(idx / N), int(idx % N)
            lil_sol[row, col] = 1

    print('Elapsed %.2f seconds' % (end - start))
    print('Energy: ', response.energy + offset)
    return lil_sol.tocsr()

def QUBOSolValid(csr_sol, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
    violate_count = 0# counter of rule violations
    # check implicit constraints
    violate_count += csr_sol.sum() - csr_sol.shape[0]
    # check explicit constraints
    for (i, j) in [(io_block, clb_site) for io_block in IO_blocks for clb_site in CLB_sites]:
        violate_count += csr_sol[i, j] != 0
    for (i, j) in [(clb_block, io_site) for clb_block in CLB_blocks for io_site in IO_sites]:
        violate_count += csr_sol[i, j] != 0
    return violate_count

# numba function requires to run once to warmup (precompiled)
def QUBOConstruct_warmup():
    NM, MDM = np.array([[0, 3], [2, 4]], dtype = np.int64), np.array([[1, 2], [3, 4]], dtype = np.int64)
    P_i, P_e = 200, 200
    IO_blocks, CLB_blocks, IO_sites, CLB_sites = List(), List(), List(), List()
    IO_blocks.append(0)
    CLB_blocks.append(1)
    IO_sites.append(0)
    CLB_sites.append(1)
    QUBOConstruct(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites)

def QUBOConstruct_test():
    NM, MDM = np.random.randint(2, size = (64, 64), dtype = np.int64), np.random.randint(2, size = (64, 64), dtype = np.int64)
    P_i, P_e = 200, 200
    IO_blocks, CLB_blocks, IO_sites, CLB_sites = List(), List(), List(), List()
    IO_blocks.append(0)
    CLB_blocks.append(1)
    IO_sites.append(0)
    CLB_sites.append(1)
    return QUBOConstruct(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites)

def QUBOConstruct_goldenref(NM, MDM, P_i, P_e, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
    # use pyqubo to generate qubo matrix as a golden reference
    N = MDM.shape[0]
    X = Array.create('X', shape = (N, N), vartype = 'BINARY')
    print(N)

    # generate quadratic objective function
    obj = 0
    for i in trange(N, desc = 'Creating quadratic objective'):
        for j in range(N):
            obj += NM[i, j] * X[i, :].dot(MDM).dot(X[j, :].T)

    constr = 0

    # generate implicit constraints
    for i in trange(N, desc = 'Creating row implicit constraints'):
        C = 0
        for j in range(N):
            C += X[i, j]
        constr += P_i * (C - 1) ** 2
    for j in trange(N, desc = 'Creating col implicit constraints'):
        C = 0
        for i in range(N):
            C += X[i, j]
        constr += P_i * (C - 1) ** 2

    # generate explicit constraints
    for (i, j) in tqdm([(io_block, clb_site) for io_block in IO_blocks for clb_site in CLB_sites], desc = 'Creating explicit constraint type 1'):
        constr += P_e * (X[i, j] ** 2)
    for (i, j) in tqdm([(clb_block, io_site) for clb_block in CLB_blocks for io_site in IO_sites], desc = 'Creating explicit constraint type 2'):
        constr += P_e * (X[i, j] ** 2)

    H = obj + constr
    model = H.compile()
    qubo_dict, offset = model.to_qubo()

    qubo_mat = np.zeros((N**2, N**2))
    for (key, val) in qubo_dict.items():
        src_x, src_y = re.findall(r'\d+', key[0])
        dst_x, dst_y = re.findall(r'\d+', key[1])
        qubo_mat[int(src_x)*N+int(src_y), int(dst_x)*N+int(dst_y)] = val

    # convert to upper triangle matrix
    for (i, j) in tqdm([(i, j) for i in range(N**2) for j in range(i+1, N**2)], desc = 'Converting to upper triangle matrix'):
        qubo_mat[i, j], qubo_mat[j, i] = qubo_mat[i, j] + qubo_mat[j, i], 0

    return coo_matrix(qubo_mat), offset

def QUBOVPRSolEval(vpr_placement, netlist_nodes, rrgraph_nodes, N, coo_qubo, offset, IO_blocks, CLB_blocks, IO_sites, CLB_sites):
    # extract VPR baseline solution and return energy
    netlist_idx_dict = dict((k, v) for v, k in enumerate(netlist_nodes))
    rrgraph_idx_dict = dict((k, v) for v, k in enumerate(rrgraph_nodes))
    lil_sol = lil_matrix((N, N), dtype=np.uint8)
    with open(vpr_placement, 'r') as f:
        next(f)
        next(f)
        next(f)
        next(f)
        next(f)
        for line in f:
            tokens = line.split()
            blk_name, phys_site = tokens[0], 'BLK_%s_%s' % (tokens[1], tokens[2])
            row, col = netlist_idx_dict[blk_name], rrgraph_idx_dict[phys_site]
            lil_sol[row, col] = 1
    # generate two lists, one for empty row, one for empty col
    empty_rows, empty_cols = [], []
    for row in range(N):
        if lil_sol[row, :].sum() == 0:
            empty_rows.append(row)
    for col in range(N):
        if lil_sol[:, col].sum() == 0:
            empty_cols.append(col)
    for (row, col) in zip(empty_rows, empty_cols):
        lil_sol[row, col] = 1
    # print('Baseline solution:')
    # pprint(lil_sol.todense())
    # check violations
    print('Baseline # rules violated: ', QUBOSolValid(lil_sol.tocsr(), IO_blocks, CLB_blocks, IO_sites, CLB_sites))
    # flatten solution to vector
    flat_csr_sol = lil_sol.reshape((1, N**2)).tocsr()
    energy = flat_csr_sol.dot(coo_qubo.tocsr()).dot(flat_csr_sol.T)
    print('Baseline Energy: ', energy.tocoo().data[0] + offset)
