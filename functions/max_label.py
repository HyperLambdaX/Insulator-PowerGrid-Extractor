"""Return the label associated with the largest cluster."""
import numpy as np


def max_label(labels):
    """
    Returns the label number corresponding to the largest cluster
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Array of cluster labels
        
    Returns:
    --------
    max_label_ind : scalar
        Label corresponding to the largest cluster
        
    """
    uni_labels = np.unique(labels)
    
    cnum = len(uni_labels)
    
    cpts_num = np.zeros(cnum)
    
    for i in range(cnum):
        cpts_num[i] = np.sum(labels == uni_labels[i])
    
    tep_ind = np.argmax(cpts_num)
    
    max_label_ind = uni_labels[tep_ind]
    
    return max_label_ind
