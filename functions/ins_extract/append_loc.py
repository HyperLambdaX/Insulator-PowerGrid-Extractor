"""
Append location information

"""
import numpy as np
from ..draw_results.drow_zbarh import drow_zbarh


def append_loc(img, loc):
    """
    Append location information by finding additional positions
    
    Parameters:
    -----------
    img : numpy.ndarray
        Binary image
    loc : numpy.ndarray
        Current location array [start, end]
        
    Returns:
    --------
    loc_p : numpy.ndarray
        Extended location array with additional row
        
    """
    loc_end = loc[-1, 1] if loc.ndim == 2 else loc[1]
    up_img = img[loc_end:, :]

    # Find top 2 maximum width differences
    dwid = drow_zbarh(up_img, 1, 'Dwid')
    # maxk returns k largest values and their indices
    loc_ind = np.argsort(dwid)[-2:][::-1]  # Get indices of top 2, descending order

    new_row = np.array([[np.min(loc_ind) + loc_end, np.max(loc_ind) + loc_end]])

    # Prepend new row to loc
    loc_p = np.vstack([new_row, loc])
    
    return loc_p

