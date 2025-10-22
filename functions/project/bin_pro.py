"""Project a 3D point cloud onto a selected coordinate plane (3D to 2D)."""
import numpy as np


def bin_pro(pts, grid_width, axle_x, axle_y):
    """
    Project the point cloud onto any plane formed by the coordinate axes (3D to 2D)
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Point cloud array of shape (N, 3) with columns [X, Y, Z]
    grid_width : float
        Grid cell size for projection
    axle_x : int
        Column index for x-axis (0, 1, or 2 for X, Y, Z)
    axle_y : int
        Column index for y-axis (0, 1, or 2 for X, Y, Z)
        
    Returns:
    --------
    img : numpy.ndarray
        Binary image of shape (GridH, GridW)
        
    """
    x = pts[:, axle_x - 1]
    y = pts[:, axle_y - 1]
    
    grid_w = int(np.ceil((np.max(x) - np.min(x)) / grid_width))
    
    grid_h = int(np.ceil((np.max(y) - np.min(y)) / grid_width))
    
    ind = np.column_stack([
        np.floor((y - np.min(y)) / grid_width).astype(int),
        np.floor((x - np.min(x)) / grid_width).astype(int)
    ])
    
    img = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            if np.sum((ind[:, 0] == i) & (ind[:, 1] == j)) > 0:
                img[i, j] = 1
    
    return img
