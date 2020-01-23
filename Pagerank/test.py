"""https://en.wikipedia.org/wiki/PageRank#Python
"""
import numpy as np


def pagerank(M, num_iterations: int = 100, d: float = 0.85):
  """PageRank: The trillion dollar algorithm.

  Parameters
  ----------
  M : numpy array
      adjacency matrix where M_i,j represents the link from 'j' to 'i', such
      that for all 'j' sum(i, M_i,j) = 1
  num_iterations : int, optional
      number of iterations, by default 100
  d : float, optional
      damping factor, by default 0.85

  Returns
  -------
  numpy array
      a vector of ranks such that v_i is the i-th rank from [0, 1],
      v sums to 1

  """
  N = M.shape[1]
  v = np.random.rand(N, 1)
  v = v / np.linalg.norm(v, 1)
  M_hat = (d * M + (1 - d) / N)
  for i in range(num_iterations):
    v = M_hat @ v
  return v


M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
v = pagerank(M, 100, 0.85)
print(v)
