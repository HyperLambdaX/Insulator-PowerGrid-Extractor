"""Remove lightning-protection wires by dropping the highest clusters."""
import numpy as np
from sklearn.cluster import DBSCAN


def split_power_line_c5(line, get_num, dist):
    """
    Separate lightning protection wire
    Removes the highest GetNum clusters (lightning protection wires)
    
    Parameters:
    -----------
    line : numpy.ndarray
        Line point cloud of shape (N, 3) [X, Y, Z]
    get_num : int
        Number of highest clusters to remove
    dist : float
        DBSCAN distance threshold
        
    Returns:
    --------
    cross_line : numpy.ndarray
        Point cloud with highest clusters removed
    """
    # Clustering
    dbscan_model = DBSCAN(eps=dist, min_samples=2)
    labels = dbscan_model.fit_predict(line)
    
    labels_uni = np.unique(labels)
    
    c_num = len(labels_uni)
    
    # Count the highest elevation point of each cluster
    z_mid = np.zeros(c_num)
    
    for i in range(c_num):
        cluster_pts = line[labels == labels_uni[i], :]
        
        z_mid[i] = np.min(cluster_pts[:, 2]) + (np.max(cluster_pts[:, 2]) - np.min(cluster_pts[:, 2])) / 2
    
    # Number clusters in order
    new_labels = np.zeros_like(labels)
    
    for i in range(c_num):
        new_labels[labels == labels_uni[i]] = i + 1
    
    # Delete the highest GetNum clusters
    de_labels = []
    
    z_mid_copy = z_mid.copy()
    for i in range(get_num):
        max_label = np.argmax(z_mid_copy)
        
        # Find indices of points belonging to this cluster
        de_labels.extend(np.where(new_labels == max_label + 1)[0].tolist())
        
        # Set to minimum so it won't be selected again
        z_mid_copy[max_label] = np.min(z_mid_copy)
    
    # Remove selected points
    keep_mask = np.ones(line.shape[0], dtype=bool)
    keep_mask[de_labels] = False
    cross_line = line[keep_mask, :]
    
    return cross_line
