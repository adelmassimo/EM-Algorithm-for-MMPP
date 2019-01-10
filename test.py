import pickle
from os import listdir
import datasetReader as df

dataset = df.DatasetFactory()
dataset.createDataset(1)

trained_models = [f for f in listdir("saved_models") if f != ".DS_Store" and f == "model5states-201901081348.p"]
# print(trained_models)

for model_name in trained_models:
    # print("\n"+model_name)
    trained_model = pickle.load( open( "saved_models/"+model_name, "rb" ) )
    # trained_model.compute_generators()
    print(trained_model.M)
    print(trained_model.D0)
    print(trained_model.D1)
    j = 0
    print(dataset.traces[j])
    print(trained_model.traces)
    for i in range(len(dataset.traces[j])):
        print(trained_model.forward_likelihood(i, dataset.traces[j]))
    # for trace in trained_model.traces:
    #     print(trace)
    #     print( trained_model.forward_likelihood(1,trace) )
    #     print( trained_model.forward_likelihood(2,trace) )
    #     print( trained_model.forward_likelihood(3,trace) )
    #     print( trained_model.forward_likelihood(4,trace) )
    #     print( trained_model.forward_likelihood(5,trace) )
    #     print( trained_model.forward_likelihood(6,trace) )
    #     print( trained_model.forward_likelihood(7,trace) )
    #     print( trained_model.forward_likelihood(12,trace) )

# print(trained_model.P0)

# q = 0.78
# t = 116.8020823304408
# k = 100
# e = np.exp(-q*t)*np.power(q*t, k)/factorial(k)

# print(factorial(k))
# print( np.exp(-q*t)*np.power(q*t, k) )
# print( e )