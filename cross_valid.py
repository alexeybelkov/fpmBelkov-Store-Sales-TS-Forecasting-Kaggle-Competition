#from sklearn.model_selection import TimeSeriesSplit
#from pandas import DataFrame, Series as pandas_DataFrame, pandas_Series
#from numpy import array as numpy_array

class CrossValid:
    
    def __init__(self, train_init_size, test_size, gap=0):
        self.train_init_size = train_init_size
        self.test_size = test_size
        self.gap = gap

    def split(self, data, step=1):
        train_begin, train_end = 0, self.train_init_size
        test_begin, test_end = train_end + self.gap, train_end + self.gap + self.test_size

        while test_end <= data.shape[0]:
            yield data[train_begin:train_end], data[test_begin:test_end]
            train_begin += step
            train_end += step
            test_begin += step
            test_end += step

        #if type(data) == type(pandas_DataFrame) or type(data) == type(pandas_Series):
        #elif type(data) == type(numpy_array):
            #pass