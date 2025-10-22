"""Variant of SplitPowerLine_C4 with simplified cluster merging."""
import numpy as np
from sklearn.cluster import DBSCAN


def split_power_line_c41(line, get_num, dist):
    """
    Another version of SplitPowerLine_C4
    Only modifies line 24 of SplitPowerLine_C4
    Input point cloud does not contain the ground line
    
    Parameters:
    -----------
    line : numpy.ndarray
        Line point cloud of shape (N, 3) [X, Y, Z]
    get_num : int
        Number of line groups to extract
    dist : float
        DBSCAN distance threshold
        
    Returns:
    --------
    cross_line : list of numpy.ndarray
        List containing separated power line groups
        
    """
    dbscan_model = DBSCAN(eps=dist, min_samples=2)
    labels = dbscan_model.fit_predict(line)
    
    labels_uni = np.unique(labels)
    
    c_num = len(labels_uni)
    
    z_mid = np.zeros(c_num)
    
    for i in range(c_num):
        cluster_pts = line[labels == labels_uni[i], :]
        
        z_mid[i] = np.min(cluster_pts[:, 2]) + (np.max(cluster_pts[:, 2]) - np.min(cluster_pts[:, 2])) / 2
    
    new_labels = np.zeros_like(labels)
    
    for i in range(c_num):
        new_labels[labels == labels_uni[i]] = i
    
    sort_ind1 = np.argsort(z_mid)
    z_mid_sort = z_mid[sort_ind1]
    
    z_diff = np.abs(np.diff(z_mid_sort))
    cut_pos = np.argsort(z_diff)[-(get_num - 1):][::-1] if get_num > 1 else []
    
    if len(cut_pos) > 0:
        c_section = np.concatenate([[-1], np.sort(cut_pos), [c_num - 1]])
    else:
        c_section = np.array([-1, c_num - 1])
    
    cross_line = []
    
    for i in range(get_num):
        indi = sort_ind1[int(c_section[i]) + 1:int(c_section[i + 1]) + 1]
        
        cross_line_i = np.zeros((0, 3))
        
        for j in range(len(indi)):
            cross_line_i = np.vstack([cross_line_i, line[new_labels == indi[j], :]])
        
        cross_line.append(cross_line_i)
    
    # Reverse order
    cross_line = cross_line[::-1]
    
    return cross_line
