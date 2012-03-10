import numpy as np

def get_log_likelihood(pdf, obs):
    return np.log(pdf(obs)).sum()


def logsumexp(arr, axis=0):
    #taken from scikit-learn
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


