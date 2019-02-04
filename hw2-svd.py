from scipy import linalg
import numpy as np

M = np.array([[1,2],[2,1],[3,4],[4,3]])
U,S,V_t = linalg.svd(M)

MTM = M.T.dot(M)
evals, evecs = linalg.eigh(MTM)
zips = list(zip(evals, evecs))
zips = sorted(zips, key=lambda x: -x[0])
evals, evecs = zip(*zips)

np.allclose(V_t.T, evecs)