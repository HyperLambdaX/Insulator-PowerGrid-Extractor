"""Locate tower cross arms from a binary projection image."""
import numpy as np


def cross_location(img, ratio):
    """
    Calculate the position of the cross arm of the binary image tower
    
    Parameters:
    -----------
    img : numpy.ndarray
        Binary image of tower projection
    ratio : float
        Ratio for threshold calculation (typically 4)
        
    Returns:
    --------
    loc : numpy.ndarray
        Array of shape (N, 2) with [start, end] positions of cross arms
        First cross arm is at the top
        
    """
    zd = np.sum(img, axis=1)
    
    # Take the upper part
    half_len = int(np.floor(len(zd) / 2))
    
    dzd = zd[len(zd) - half_len:]
    
    # Cut off the first third of the length (actually cuts by ratio)
    dzd = dzd - np.max(dzd) / ratio
    
    dzd[dzd < 0] = 0
    
    # Dislocation histogram
    dz_move = np.zeros(len(dzd))
    
    dz_move[1:] = dzd[:-1]
    
    # Dislocation addition
    sum_dz = dzd + dz_move
    
    ind_begin_temp = np.where((dzd == sum_dz) & (sum_dz != 0))[0]
    ind_begin = ind_begin_temp + half_len
    
    ind_end_temp = np.where((dz_move == sum_dz) & (sum_dz != 0))[0]
    ind_end = ind_end_temp + half_len
    
    max_zd = np.max(zd)
    k = 0
    while k < len(ind_end):
        if k >= len(ind_begin):
            break
        if np.max(zd[ind_begin[k]:ind_end[k] + 1]) * 1.8 < max_zd:
            ind_begin = np.delete(ind_begin, k)
            ind_end = np.delete(ind_end, k)
        else:
            k += 1
    
    if len(ind_begin) != len(ind_end):
        # Add image height as final end position
        ind_end_extended = np.append(ind_end, img.shape[0] - 1)
        loc = np.column_stack([ind_begin, ind_end_extended[:len(ind_begin)]])
        loc = np.flipud(loc)
    else:
        # The first cross arm is at the top
        loc = np.column_stack([ind_begin, ind_end])
        loc = np.flipud(loc)
    
    return loc
