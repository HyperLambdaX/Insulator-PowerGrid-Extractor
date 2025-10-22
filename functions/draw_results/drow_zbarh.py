"""Utility functions to derive projection features from a binary image."""
import numpy as np


def drow_zbarh(img, direction, param_type=None):
    """
    Calculate projection feature parameters from a binary image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Binary image array
    direction : int
        Projection direction:
        1: from left to right
        -1: from right to left
        2: from bottom to top
        -2: from top to bottom
    param_type : str, optional
        Type of parameter to return:
        'sum': Sum of pixels in each row
        'Dsum': Absolute difference of sum
        'fir': First non-zero pixel position
        'Dfir': Absolute difference of first position
        'end': Last non-zero pixel position
        'Dend': Absolute difference of last position
        'wid': Width (first to last non-zero)
        'Dwid': Absolute difference of width
        'epy': Empty pixels within width
        
    Returns:
    --------
    para : numpy.ndarray
        Selected parameter array
    """
    if direction == -1:  # from right to left
        img = np.fliplr(img)
    elif direction == 2:  # from bottom to top
        img = np.flipud(img.T)
    elif direction == -2:  # from top to bottom
        img = np.fliplr(img.T)
    elif direction != 1:
        raise ValueError('undefined projection type')
    
    bar_len = img.shape[0]
    
    img_s = np.sum(img, axis=1)
    
    img_w = np.zeros(bar_len)
    img_f = np.zeros(bar_len)
    img_e = np.zeros(bar_len)
    img_ey = np.zeros(bar_len)
    
    for i in range(bar_len - 1, -1, -1):
        is_true = np.where(img[i, :])[0]
        
        if len(is_true) == 0:  # blank line
            continue
        
        img_f[i] = is_true[0] + 1
        
        img_e[i] = is_true[-1] + 1
        
        img_w[i] = is_true[-1] - is_true[0] + 1
        
        # Count empty pixels within the width
        img_ey[i] = np.sum(img[i, is_true[0]:is_true[-1] + 1] == 0)
    
    if param_type is not None:
        if param_type == 'sum':
            para = img_s
        elif param_type == 'Dsum':
            para = np.abs(np.diff(img_s))
        elif param_type == 'fir':
            para = img_f
        elif param_type == 'Dfir':
            para = np.abs(np.diff(img_f))
        elif param_type == 'end':
            para = img_e
        elif param_type == 'Dend':
            para = np.abs(np.diff(img_e))
        elif param_type == 'wid':
            para = img_w
        elif param_type == 'Dwid':
            para = np.abs(np.diff(img_w))
        elif param_type == 'epy':
            para = img_ey
        else:
            raise ValueError('undefined projection type')
    else:
        # Return all parameters as dictionary if no type specified
        para = {
            'sum': img_s,
            'fir': img_f,
            'end': img_e,
            'wid': img_w,
            'epy': img_ey
        }
    
    return para
