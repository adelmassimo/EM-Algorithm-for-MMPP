import csv
import numpy as np

myAlphabet = {'A' : 1,
              'B': 0,
              'A2': 1,
              "B2": 2,
              'end': 3}

class DatasetFactory():

    def __init__(self):
        """docstring for Model"""
        self.traces = []
        self.base_path = '/Users/adel/Documents/workspace/datasetSimulator/logs'

    def read_csv(self, path):
        with open(path) as f:
            reader = csv.reader(f,delimiter=';')
            rows = []
            for row in reader:
                # row[1] = np.float(row[1])
                if row[0] != 't0' and row[0] != 't1':
                    rows.append( (np.float(row[1]), myAlphabet[row[0]]) )

            return rows

    def createDataset(self, size):
        for i in range(size):
            X = self.read_csv( self.base_path+'/out_'+str(i+1)+'.csv' )
            self.traces.append(X)


