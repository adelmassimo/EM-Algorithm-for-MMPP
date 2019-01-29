import pickle
from os import listdir
import datasetReader as df
import numpy as np
import time
import matplotlib.pyplot as plt


dataset = df.Dataset('/Users/adel/Documents/workspace/datasetSimulator/logs/goodDs', 8)
#model4states-201901131640.p
#model4states-201901201617.p
#model4states-201901131823.p
#model4states-201901231849.p ***
#model4states-201901231857.p
#model4states-201901241659.p + goodDs
# 201901280025 + newDs 8 track
# 201901281032 + newDs 0 track
# 201901281057
# 201901281116
# 201901281617 const 4tracks goodds
#201901281637
trained_models = [f for f in listdir("saved_models") if f != ".DS_Store" and f == "model6states-201901281642.p"]
for model_name in trained_models:

    trained_model = pickle.load( open( "saved_models/"+model_name, "rb" ) )
    print("trained on " + str(len(trained_model.traces)) + "traces")
    print(trained_model.M)
    # trained_model.D0[2,3] = 1
    print(trained_model.D1)
    print(trained_model.D0)
    # for j in range(8):
    j = 0
    exit_times = -1/( trained_model.D0 + trained_model.D1 ).diagonal()
    exit_times[0, -1] = 0

    # tot_time =  sum( [dataset.traces[j][i][0] for i in range(len(dataset.traces[j]))] )
    tot_time = 0
    for i in range(len(dataset.traces[j])):
        tot_time = tot_time + dataset.traces[j][i][0]

    absolute_time = 0

    m = len(dataset.traces[j])
    # print(
    #     trained_model.uniformisation_rate**m *
    #     trained_model.forward_likelihood( m-1, dataset.traces[j]).sum()
    # )

    ''' D1 evaluation '''
    for k in range(len(trained_model.D1)):
        d1_error = [ np.sqrt((1/trained_model.D1[k,k] - dataset.traces[j][i][0])**2)
                    for i in range(len(dataset.traces[j]))]
        # d1_error = d1_error / max([abs(e) for e in d1_error ])
        print(d1_error)

        # Plot the error
        plt.plot(d1_error, label="$\lambda_"+str(k)+"$")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(trained_model.D1), mode="expand", borderaxespad=0.)
    plt.show()

    ''' Prediction evaluation '''
    ettf = []
    gttf = []
    for i in range(len(dataset.traces[j])):

        state_probability = trained_model.forward_likelihood(i, dataset.traces[j])
        absolute_time = absolute_time + dataset.traces[j][i][0]
        # print(state_probability)

        sum = 0
        for state in range(trained_model.N):
            time_to_fail = exit_times[0, state:-1].sum()
            estimated_time_to_fail = state_probability[0,state]*time_to_fail
            sum = sum + estimated_time_to_fail
        ettf.append(sum)
        gttf.append(tot_time-absolute_time)
        # print( str(i)+")" + str(sum) +" "+str(tot_time-absolute_time) )

    ettf_norm = ettf / np.asarray(max(ettf))
    gttf_norm = gttf / np.asarray(max(gttf))
    print(ettf_norm)
    print(gttf_norm)

    plt.plot(ettf_norm, label="prevision")
    plt.plot(gttf_norm, label="groundtruth")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


    plt.plot(ettf, label="prevision")
    plt.plot(gttf, label="groundtruth")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

    print("Absolute error:")
    print(np.mean([
        np.sqrt( (e - g) ** 2 )
        for e, g in zip(ettf_norm, gttf_norm)
        ])
    )

    print("Absolute error:")
    print(np.mean([
        np.sqrt( (e - g) ** 2 )
        for e, g in zip(ettf, gttf)
        ])
    )
