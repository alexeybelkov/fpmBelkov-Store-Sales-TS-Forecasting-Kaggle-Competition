import pandas as pd
import numpy as np

def MAPE(y, y_hat):
    return np.mean(np.abs(y-y_hat)/np.abs(y_hat))

def SMAPE(y,y_hat):
    return np.mean(np.sum(2*np.abs(y-y_hat)/(np.abs(y)+np.abs(y_hat))))

def MSE(y, y_hat):
    return np.mean((y-y_hat)**2)

def RMSE(y, y_hat):
    return np.sqrt(MSE(y, y_hat))


def feature_extractor(dataframe, gcols, inplace, **kwargs):
    data = dataframe if inplace else dataframe.copy()
    grouped = data.groupby(gcols)
    new_features = {key:[] for key in kwargs}
    for key in kwargs:
        if key == 'lag':
            params = kwargs['lag']
            for lag, col in zip(params['lags'],params['columns']):
                new_feature_name = f'{col}_[{lag}]'
                if new_feature_name in data.columns:
                    continue
                data[new_feature_name] = grouped[col].shift(lag)
                new_features[key].append(new_feature_name)
        elif key == 'rolling':
            params = kwargs['rolling']
            for lag, col, func in zip(params['lags'], params['columns'], params['funcs']):
                new_feature_name = f'{col}_R[{lag}]_{func.__name__}'
                if new_feature_name in data.columns:
                    continue
                data[new_feature_name] = grouped[col].rolling(lag).apply(func)
                new_features[key].append(new_feature_name)

        elif key == 'expanding':
            params = kwargs['expanding']
            for lag, col, func in zip(params['lags'], params['columns'], params['funcs']):
                new_feature_name = f'{col}_E[{lag}]_{func.__name__}'
                if new_feature_name in data.columns:
                    continue
                data[new_feature_name] = grouped[col].expanding(lag).apply(func)
                new_features[key].append(new_feature_name)
    return data, new_features