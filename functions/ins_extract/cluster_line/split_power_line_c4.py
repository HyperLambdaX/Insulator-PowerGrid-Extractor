"""Cluster-based splitter for power-line point clouds."""
import numpy as np
from sklearn.cluster import DBSCAN


def split_power_line_c4(line, get_num, dist):
    """
    Separating power line point clouds using clustering methods
    
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
        Ordered from top to bottom
    """
    # Clustering
    dbscan_model = DBSCAN(eps=dist, min_samples=2)
    labels = dbscan_model.fit_predict(line)
    
    line = line[labels != -1, :]
    labels = labels[labels != -1]
    
    labels_uni = np.unique(labels)
    
    c_num = len(labels_uni)
    
    # Count the elevation midpoint of each cluster
    z_mid = np.zeros(c_num)
    
    for i in range(c_num):
        cluster_pts = line[labels == labels_uni[i], :]
        
        z_mid[i] = np.min(cluster_pts[:, 2]) + (np.max(cluster_pts[:, 2]) - np.min(cluster_pts[:, 2])) / 2
    
    # Number clusters in order
    new_labels = np.zeros_like(labels)
    
    for i in range(c_num):
        new_labels[labels == labels_uni[i]] = i
    
    # Merge clusters
    sort_ind1 = np.argsort(z_mid)
    z_mid_sort = z_mid[sort_ind1]
    
    # Find GetNum largest differences
    z_diff = np.abs(np.diff(z_mid_sort))
    # maxk returns k largest values and their indices
    cut_pos = np.argsort(z_diff)[-get_num:][::-1]  # Get indices of largest differences
    
    c_section = np.concatenate([[-1], np.sort(cut_pos), [c_num - 1]])
    
    cross_line = []
    
    for i in range(get_num):
        indi = sort_ind1[int(c_section[i] + 1):int(c_section[i + 1] + 1)]
        
        cross_line_i = np.zeros((0, 3))
        
        for j in range(len(indi)):
            cross_line_i = np.vstack([cross_line_i, line[new_labels == indi[j], :]])
        
        # Reverse order: store in reverse
        cross_line.append(cross_line_i)
    
    # Reverse to get top to bottom order
    cross_line = cross_line[::-1]
    
    return cross_line
