import numpy as np
from numpy import linalg as LA
from scipy.misc import factorial
import random

class Model():

    def __init__(self):
        """docstring for Model"""
        self.N = 3
        self.observations = [(4, 0), (3, 0), (5, 0), (2,0), (9, 2), (11, 2), (8, 2), (20, 3), (17, 3), (18, 3)]

        self.M = np.matrix('.7 .2 .998 0.002; .3 .5 .199 0.001; .1 .2 .1 .6')

        self.uniformization_rate = max( [self.observations[i][0] for i in range(len(self.observations)) ] )

        # self.P0 = np.matrix('.1 .2 .3; 0 .3 .2; 0 0 .6')
        # self.P1 = np.matrix('.4 0 0; 0 .5 0; 0 0 .4')
        self.P0, self.P1 = self.generateTransitionProbabilities( self.N )

    def generateTransitionProbabilities(self, N):
        # Define a stochastic matrix P0
        P0 = np.matrix(np.random.rand(N, N))
        P0 = P0 / P0.sum(axis=1)

        # Get N elements, each of these less than the respective element on diagonal of P0
        d = P0.diagonal()
        d = [random.uniform(0, d[0, i]) for i in range(N)]

        # P1 is exactly a diagonal NxN matrix with d as diagonal
        P1 = d * np.eye(N)

        # P0 + P1 must be stochastic, so:
        P0 = P0 - P1

        return P0, P1