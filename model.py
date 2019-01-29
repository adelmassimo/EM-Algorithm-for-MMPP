import numpy as np
import random
import utils

class Model():

    def __init__(self, n, dataset, m_shape):
        # Number of states
        self.N = n
        # Number of different symbols emitted
        self.R = dataset.count_symbols()
        # Number of traces loaded T
        self.T = dataset.size()

        # Set traces
        self.traces = dataset.traces

        self.M = self.generate_emission_probabilities(m_shape)

        # compute the uniformization rate
        self.uniformisation_rate = dataset.uniformisatoin_rate()

        # Generate P0 and P1 matricies
        self.P0, self.P1 = self.generate_transition_probabilities( self.N )
        # self.P0, self.P1 = self.fixed_probabilities()

        self.compute_generators()
        self.D0 = np.zeros(self.N)
        self.D1 = np.zeros(self.N)

    def generate_transition_probabilities(self, N):
        # Define a stochastic matrix P0
        # TODO controllo su eventuali zeri
        P0 = np.matrix(
            np.triu(np.random.rand(N, N), 0)
            # np.random.rand(N, N)
        )
        P0 = P0 / P0.sum(axis=1)

        # Get N elements, each of these less than the respective element on diagonal of P0
        d = P0.diagonal()
        d = [random.uniform(0, d[0, i]) for i in range(N)]
        # d = [random.uniform(d[0, i]/2, d[0, i]) for i in range(N)]

        # P1 is exactly a diagonal NxN matrix with d as diagonal
        P1 = d * np.eye(N)

        # P0 + P1 must be stochastic, so:
        P0 = P0 - P1
        D1 = P1 * self.uniformisation_rate
        D0 = P0 * self.uniformisation_rate - self.uniformisation_rate * np.eye(self.N, self.N)

        return P0, np.matrix(P1)

    def generate_emission_probabilities(self, shape):
        # Define emission probability matrix
        M = np.zeros((self.N, self.R))

        if shape == "constrained":
            M[0:self.N - 1, 0:self.R - 1] = 1 / (self.R - 1)
            M[self.N - 1, self.R - 1] = 1
        elif shape == "uniform":
            M[0:self.N, 0:self.R] = 1 / self.R
        elif shape == "soft":
            M[0:self.N - 1, 0:self.R - 1] = 1 / (self.R - 1)
            M[self.N-1, 0:self.R-1] = .1
            M[self.N - 1, self.R - 1] = 1 - (self.R-1)*.1
            print(M)

        M = np.matrix(M)
        return M

    def compute_generators(self):
        self.D0 = self.P0 * self.uniformisation_rate - self.uniformisation_rate * np.eye(self.N, self.N)
        self.D1 = self.P1 * self.uniformisation_rate

        print("\nD0:")
        print(self.D0)
        print("\nD1:")
        print(self.D1)

    def backward_likelihood_absolute(self, i, trace):
        N = self.N
        M = len(trace)
        likelihoods = np.ones((N, 1))
        if i < M:
            P = utils.randomization(self.P0, self.uniformisation_rate, trace[i][0])

            likelihoods = np.multiply(
                P.dot(self.P1).dot(self.backward_likelihood_absolute(i + 1, trace)),
                self.M[:, trace[i][1]]
            )
        return likelihoods

    def forward_likelihood_absolute(self, i, trace):
        N = self.N
        likelihoods = np.zeros((1, N))
        likelihoods[0, 0] = 1

        if i != 0:
            P = utils.randomization(self.P0, self.uniformisation_rate, trace[i][0])

            likelihoods = np.multiply(
                self.forward_likelihood_absolute(i - 1, trace).dot(P).dot(self.P1),
                np.transpose( self.M[:, trace[i][1]] )
            )

        return likelihoods

    def forward_likelihood(self, i, trace):
        likelihoods = self.forward_likelihood_absolute(i, trace)

        if likelihoods.sum() != 0:
            likelihoods = likelihoods / likelihoods.sum()

        return likelihoods

    def backward_likelihood(self, i, trace):
        likelihoods = self.backward_likelihood_absolute(i, trace)

        if likelihoods.sum() != 0:
            likelihoods = likelihoods / likelihoods.sum()

        return likelihoods

    # def fixed_probabilities(self):
    #     P0 = np.matrix("[0.45 0.15 0.1 0.05; 0 0.27 0.4 0.1; 0 0 0.6 0.05; 0 0 0 0.1]")
    #     P1 = np.matrix("[0.25 0 0 0; 0 0.23 0 0; 0 0 0.5 0; 0 0 0 .9]")
    #     # P0 = np.matrix("[0.5  0.3 ; 0 0.3]")
    #     # P1 = np.matrix("[0.2 0 ; 0 0.7]")
    #
    #     # D1 = P1 * self.uniformization_rate
    #     # D0 = (P0 - np.eye(self.N, self.N)) * self.uniformization_rate   # *\
    #
    #     return P0, P1
