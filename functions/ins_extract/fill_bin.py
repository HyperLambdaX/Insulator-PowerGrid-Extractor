"""Fill zero-valued bins by averaging the nearest non-zero neighbors."""
import numpy as np


def fill_bin(bin_arr):
    """
    Fill zero values in bin array using average of adjacent non-zero values
    
    Parameters:
    -----------
    bin_arr : numpy.ndarray
        Input bin array
        
    Returns:
    --------
    bin_filled : numpy.ndarray
        Bin array with zeros filled
        
    """
    bin_filled = bin_arr.copy()
    
    bin_num = bin_filled.shape[0]
    
    for i in range(bin_num):
        if bin_filled[i] == 0:
            # Previous non-zero bin
            last_bin = 0  # Not found, may be at the first position, default is 0
            
            # Search backwards for last non-zero bin
            for j in range(i, -1, -1):  # From i down to 0
                if bin_filled[j] != 0:
                    last_bin = bin_filled[j]
                    break
            
            # Next non-zero bin
            next_bin = 0  # Not found, may be at the end, default is 0
            
            # Search forwards for next non-zero bin
            for j in range(i, bin_num):
                if bin_filled[j] != 0:
                    next_bin = bin_filled[j]
                    break
            
            bin_filled[i] = int(np.floor((last_bin + next_bin) / 2))
    
    return bin_filled
