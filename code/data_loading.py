"""
- The purpose of this module is to read the real data from CSV files or to generate
a synthetic sinewave to test the code.
- It scale the data according to a selected scaler from sikit-learn.
- It has a differencing option to make the time-series stationary but differencing it once,
if needed, should set to 'True'.
- Finally this module has a window shift function.
"""
import numpy as np
import pandas as pd
import sklearn.preprocessing as skl_pre

def window_shift(data, _len):
    """
    A function for sliding window implementation.
    Args:
        - data: the data we want to apply the sliding window to.
        - len: sequence length for the sliding window.
    Returns:
        - shifted_data: a list of arrays.
    """
    shifted_data = []
    shift_ = 24
    for i in range(0, int((len(data) - _len)/24) ):
        _x = data[i * shift_  : _len + i * shift_]
        shifted_data.append(_x)
    return shifted_data

def sine_data_generation (samples_no = 100, seq_len = 1000, dim = 20):
    """
    Sine data generation.
    Args:
        - samples_no: the number of samples
        - seq_len: sequence length of the time-series
        - T: feature dimensions
    Returns:
        - data: generated data
        - shifted data: list of arrays
    """
    np.random.seed(2)
    seq_len = int(seq_len)
    data = np.empty((samples_no, seq_len), 'int64')
    arr1 = np.array(range(seq_len))
    arr2 = np.random.randint(-4 * seq_len, 4 * seq_len, samples_no).reshape(samples_no, 1)
    data[:] =  arr1 + arr2
    data = np.sin(data / 1.0 / dim).astype('float64')
    data = data.T
    shifted_data = window_shift(data, samples_no)
    return data, shifted_data

def real_data_loading (seq_len, is_differenced = False, scaler_name = 'MinMax'):
    """
    Load and preprocess real-world datasets.
    Args:
        - data_name: stock or energy
        - seq_len: sequence length
        - is_differenced: 'True' if we want to difference the data, it is default is 'False'
        - scaler_name: the scaler name from sklearn to scale the data, it is default is 'MinMax'
    Returns:
        - data: preprocessed data
        - scaler: the fitted scaler
        - ori_data_: the original raw data
    """
    
    ori_data_ = pd.read_csv('../data/datatraining.txt').reset_index(drop = True)
        
    ori_data_ = ori_data_[['Temperature', 'CO2', 'Humidity']]
    ori_data_ = ori_data_.reset_index(drop=True)

    ori_data = np.copy(ori_data_)

    if is_differenced is True:
        ori_data = ori_data_.diff().dropna()
        ori_data = ori_data.to_numpy()

    scalers = {'Standard' : skl_pre.StandardScaler(), 'MinMax': skl_pre.MinMaxScaler(feature_range=(-1, 1)),
                'MaxAbs': skl_pre.MaxAbsScaler(),'Robust': skl_pre.RobustScaler(),
                'PowerTransformer': skl_pre.PowerTransformer(),
                'QuantileTransformer': skl_pre.QuantileTransformer()}

    scaler = scalers[scaler_name]
    scaler.fit(ori_data)
    ori_data_scaled = scaler.transform(ori_data)
    shifted_data = window_shift(ori_data_scaled, seq_len)
    return shifted_data, scaler, ori_data_
