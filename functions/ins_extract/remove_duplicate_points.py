"""Remove points in ``big_pts`` that duplicate entries from ``small_pts``."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def remove_duplicate_points(big_pts, small_pts, thre):
    """
    Remove duplicate points: point_set2 is a subset of point_set1,
    remove point_set2 from point_set1
    
    Parameters:
    -----------
    big_pts : numpy.ndarray
        Large point set to remove points from
    small_pts : numpy.ndarray
        Small point set containing points to be removed
    thre : float
        Distance threshold for considering points as duplicates
        
    Returns:
    --------
    big_pts_cleaned : numpy.ndarray
        Point set with duplicates removed
        
    """
    if big_pts.shape[0] == 0 or small_pts.shape[0] == 0:
        return big_pts
    
    # Find 8 nearest neighbors for each small point in big_pts
    nbrs = NearestNeighbors(n_neighbors=min(8, big_pts.shape[0]), algorithm='auto')
    nbrs.fit(big_pts)
    distances, indices = nbrs.kneighbors(small_pts)
    
    # Remove points where distance is less than threshold
    # Flatten the arrays and find unique indices to remove
    mask = distances < thre
    indices_to_remove = indices[mask]
    
    # Get unique indices to remove
    unique_indices_to_remove = np.unique(indices_to_remove)
    
    # Create boolean mask for points to keep
    keep_mask = np.ones(big_pts.shape[0], dtype=bool)
    keep_mask[unique_indices_to_remove] = False
    
    # Return filtered points
    big_pts_cleaned = big_pts[keep_mask, :]
    
    return big_pts_cleaned
