'''
@author: Yang Hu
'''

import datetime

import numpy as np


class Time:
    """
    Class for displaying elapsed time.
    """
    
    def __init__(self):
        self.date = str(datetime.date.today())
        self.start = datetime.datetime.now()
    
    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))
    
    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed
    

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data) + 1e-8) / (_range + 1e-8)


def sigmoid_gate(data, ratio=20, bias=-0.6):
    '''
    Args:
        data: a float or a nparray of floats
    '''
    return 1 / (1 + np.exp(-(data + bias) * ratio))


def np_info(np_arr, name=None, elapsed=None, full_np_info=False):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    *reference the code from: https://github.com/deroneriksson/python-wsi-preprocessing
    
    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """
    
    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"
    
    if full_np_info is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
          name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


if __name__ == '__main__':
    
    
    p1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print(sigmoid_gate(p1))
