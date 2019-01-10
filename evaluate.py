import model
import numpy as np
import datasetReader as df
import main

# Number of traces loaded T
T = 1
# Generate traces
traces_factory = df.DatasetFactory()
traces_factory.createDataset(T)
traces = traces_factory.traces

P0 = np.matrix("[ .02 0;"
                "0 0 0.5;"
                "0 0 0]")

P1 = np.matrix("[0.1 0 0;"
                "0 0.5 0;"
                "0 0 0.9]")

M = np.matrix("[0.25 0 0;"
                "0 0.23 0;"
                "0 0 0.85]")



def backward_likelihood(i, trace):
    N = model.N
    M = len( trace )
    likelihoods = np.ones((N, 1))

    if i < M:
        P = main.randomization(P0, model.uniformization_rate, trace[i][0])
        # P = stored_p_values[i, :, :]
        likelihoods = np.multiply(
            P.dot( model.P1 ).dot( backward_likelihood(i+1, trace) ),
            model.M[:, trace[i][1]] )

        if likelihoods.sum() != 0:
            likelihoods = likelihoods / likelihoods.sum()

    return likelihoods