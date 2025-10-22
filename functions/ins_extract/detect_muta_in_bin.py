"""Merge consecutive positive or negative runs within a 1D histogram."""
import numpy as np


def detect_muta_in_bin(bin_arr):
    """
    Detect and merge consecutive positive and negative values in bin array
    Groups consecutive positive values and sums them, same for negative values
    
    Parameters:
    -----------
    bin_arr : numpy.ndarray
        Input bin array
        
    Returns:
    --------
    muta_bin : numpy.ndarray
        Modified bin array with merged consecutive values
        
    """
    muta_bin = bin_arr.copy()
    
    i = 0
    
    while i < len(muta_bin):
        if bin_arr[i] >= 0:
            inc_sum = 0
            start_i = i
            
            while i < len(muta_bin) and bin_arr[i] >= 0:
                inc_sum = inc_sum + bin_arr[i]
                i = i + 1
            
            muta_bin[start_i] = inc_sum
            
        elif bin_arr[i] < 0:
            dec_sum = 0
            
            while i < len(muta_bin) and bin_arr[i] < 0:
                dec_sum = dec_sum + bin_arr[i]
                i = i + 1
            
            muta_bin[i - 1] = dec_sum
            
        else:
            i = i + 1
    
    return muta_bin
