import numpy as np

from scipy.optimize import minimize
from scipy.stats.distributions import norm

from solver import Solver
from utils import get_log_likelihood

def gaussian_generator(mu=0, sigma=1., size=1000):
    return np.random.normal(mu, sigma, size=size)

def test_max_likelihood():
    x = [10., 2.]
    obs = gaussian_generator(x[0], x[1])
    pdf = norm(x[0], x[1]).pdf

    func_to_minimize = lambda x: -get_log_likelihood( norm(x[0], x[1]).pdf, obs )

    solver = Solver(func_to_minimize, minimize)

    res = solver.solve([0., 0.5])
    print res


