import numpy as np

def get_log_likelihood(pdf, obs):
    return np.log(pdf(obs)).sum()


