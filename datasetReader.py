import csv
import numpy as np

myAlphabet = {'A': 1,
              'B': 0,
              'A2': 1,
              "B2": 2,
              "C2": 1,
              'end': 3}


class Dataset():

    def __init__(self, path, size):
        self.traces = []
        self.base_path = path
        self.create_dataset(size)

    def create_dataset(self, size):
        for i in range(size):
            X = self.read_csv(self.base_path+'/out_'+str(i+1)+'.csv')
            self.traces.append(X)

    @staticmethod
    def read_csv(path):
        with open(path) as f:
            reader = csv.reader(f,delimiter=';')
            rows = []
            for row in reader:
                # row[1] = np.float(row[1])
                if row[0] in myAlphabet:
                    rows.append( (np.float(row[1]), myAlphabet[row[0]]) )

            return rows

    def size(self):
        return len(self.traces)

    @staticmethod
    def count_symbols():
        return myAlphabet['end']+1

    def uniformisatoin_rate(self):
        champion = -np.inf
        for j in range(self.size()):
            uniformisatoin_rate = max([1 / self.traces[j][i][0] for i in range(len(self.traces[j]))])
            if uniformisatoin_rate > champion:
                champion = uniformisatoin_rate
        return champion

    # def smooth(self, threshold):
    #     for trace in self.traces:
    #         mean = np.mean([t[0] for t in trace])
    #         for t in trace:
    #             if abs(mean - t[0]) > threshold:
    #                 t[0] = mean