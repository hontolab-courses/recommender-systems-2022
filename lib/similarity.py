import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine(M):
    _M = M.copy()
    _M[np.isnan(_M)] = 0
    similarities = cosine_similarity(_M)
    return similarities