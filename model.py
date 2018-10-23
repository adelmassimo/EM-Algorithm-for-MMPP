import numpy as np
from numpy import linalg as LA
from scipy.misc import factorial
import random

class Model:
	"""docstring for Model"""
	N = 3
	observations = [(4, 0), (3, 0), (5, 1), (2,0), (9, 2), (11, 1), (8, 1), (20, 2), (17, 2), (18, 3)]
	# P0 = np.matrix('-.4 .4 0; 0 -.5 .5; 0 0 -.6')
	P1 = np.eye( N )*.5
	M = np.matrix('.7 .2 .1 0; .3 .5 .2 0; .1 .2 .1 .6')
	
	uniformization_rate = max( [observations[i][0] for i in range(len(observations)) ] )

	# Define a stochastic matrix P0
	P0 = np.matrix( np.random.rand(N, N) )
	P0 = P0/P0.sum(axis=1)

	# Get N elements, each of these less than the respective element on diagonal of P0
	d = P0.diagonal()
	d = [random.uniform(0, d[0, i]) for i in range(N)]

	# P1 is exactly a diagonal NxN matrix with d as diagonal
	P1 = d * np.eye(N)

	# P0 + P1 must be stochastic, so:
	P0 = P0 - P1
