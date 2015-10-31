__author__ = 'eczech'


import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres, spsolve
from scipy.sparse import csgraph
from scipy import sparse


def markov_stationary_components(P, tol=1e-12):
    """
    Split the chain first to connected components, and solve the
    stationary state for the smallest one
    """
    n = P.shape[0]

    # 0. Drop zero edges
    P = P.tocsr()
    P.eliminate_zeros()

    # 1. Separate to connected components
    n_components, labels = csgraph.connected_components(P, directed=True, connection='strong')

    # 2. Pick the smallest one
    sizes = [(labels == j).sum() for j in range(n_components)]
    min_j = np.argmin(sizes)
    indices = np.flatnonzero(labels == min_j)

    #print("Solving for component {0}/{1} of size {2}".format(min_j, n_components, indices.size))

    # 3. Solve stationary state for it
    p = np.zeros(n)
    if indices.size == 1:
        # Simple case
        p[indices] = 1
    else:
        p[indices] = markov_stationary_one(P[indices,:][:,indices], tol=tol)

    return p


def markov_stationary_one(P, tol=1e-12, direct=False):
    """
    Solve stationary state of Markov chain by replacing the first
    equation by the normalization condition.
    """
    if P.shape == (1, 1):
        return np.array([1.0])

    n = P.shape[0]
    dP = P - sparse.eye(n)
    A = sparse.vstack([np.ones(n), dP.T[1:,:]])
    rhs = np.zeros((n,))
    rhs[0] = 1

    if direct:
        # Requires that the solution is unique
        return spsolve(A, rhs)
    else:
        # GMRES does not care whether the solution is unique or not, it
        # will pick the first one it finds in the Krylov subspace
        p, info = gmres(A, rhs, tol=tol)
        if info != 0:
            raise RuntimeError("gmres didn't converge")
        return p


def get_ranks(score_matrix):
    m = score_matrix.values
    m = sparse.csr_matrix(m)
    m = m.multiply(sparse.csr_matrix(1/m.sum(1).A))
    ranks = markov_stationary_components(m)
    return pd.Series(ranks, index=score_matrix.columns)