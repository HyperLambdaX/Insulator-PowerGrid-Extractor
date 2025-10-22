"""Project a point cloud onto one axis and count samples per grid bin."""
import numpy as np


def sin_pro(pts, direction, sam_gap):
    """
    Project the point cloud onto any coordinate axis (2D to 1D)
    and return the number of points in each grid
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Point cloud array of shape (N, 2) or (N, 3)
    direction : int
        Direction/axis to project onto
    sam_gap : float
        Sampling gap / grid size
        
    Returns:
    --------
    pts_d : numpy.ndarray
        Density array - number of points in each grid bin
    pts_ind : numpy.ndarray
        Grid index for each point
        
    """
    dir_col = pts[:, direction - 1]

    # Calculate range
    col_range = np.max(dir_col) - np.min(dir_col)

    # Handle edge case: all points at same location
    if col_range < sam_gap / 10:  # Very small range
        sam_len = 1
        pts_ind = np.zeros(len(pts), dtype=int)
        pts_d = np.array([len(pts)])
        return pts_d, pts_ind

    sam_len = int(np.ceil(col_range / sam_gap))

    # Ensure sam_len is at least 1
    if sam_len < 1:
        sam_len = 1

    pts_ind = np.floor((dir_col - np.min(dir_col)) / sam_gap).astype(int)

    # Handle boundary points and ensure no negative indices
    pts_ind = np.clip(pts_ind, 0, sam_len - 1)

    pts_d = np.zeros(sam_len)

    for i in range(len(pts_ind)):
        pts_d[pts_ind[i]] += 1
    
    return pts_d, pts_ind
