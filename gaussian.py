import numpy as np

from scipy.optimize import minimize
from scipy.stats.distributions import norm

from solver import Solver
from utils import get_log_likelihood

def gaussian_generator(mu=0, sigma=1., size=10000):
    return np.random.normal(mu, sigma, size=size)

def test_max_likelihood():
    import pdb
    pdb.set_trace()
    x = [1., 2.]
    obs = gaussian_generator(mu=x[0], sigma=x[1])
    pdf = norm(x[0], x[1]).pdf

    func_to_minimize = lambda x: -get_log_likelihood( norm(x[0], x[1]).pdf, obs )

    solver = Solver(func_to_minimize, minimize)

    x0 = [0., 0.5]
    res = solver.solve(x0)
    print res


