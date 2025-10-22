"""Project a point cloud onto an axis-aligned plane and bucket point indices."""
import numpy as np


def bin_projection(pts, grid_width, axle_x, axle_y):
    """
    Project the point cloud onto a plane formed by the chosen coordinate axes
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Electric tower point cloud of shape (N, 3)
    grid_width : float
        Projection grid size
    axle_x : int
        Projected x-axis
    axle_y : int
        Projected y-axis
        
    Returns:
    --------
    img : numpy.ndarray
        Binary image (data organized with lower left as origin)
    grid : list of lists
        Cell array saving point indices in each grid cell
        
    Notes:
    ------
    Binary image data is organized in the form of original data:
    - Lower left corner is coordinate origin
    - To the right is x-axis
    - Up is y-axis
    """
    x = pts[:, axle_x - 1]
    y = pts[:, axle_y - 1]
    
    grid_w = int(np.ceil((np.max(x) - np.min(x)) / grid_width))
    
    grid_h = int(np.ceil((np.max(y) - np.min(y)) / grid_width))
    
    # Create 2D list to store point indices
    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    
    ind = np.column_stack([
        np.floor((y - np.min(y)) / grid_width).astype(int),
        np.floor((x - np.min(x)) / grid_width).astype(int)
    ])
    
    ind[ind[:, 0] >= grid_h, 0] = grid_h - 1
    ind[ind[:, 1] >= grid_w, 1] = grid_w - 1
    
    for i in range(len(pts)):
        row_idx = ind[i, 0]
        col_idx = ind[i, 1]
        grid[row_idx][col_idx].append(i)
    
    img = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            if len(grid[i][j]) > 0:
                img[i, j] = 1
    
    return img, grid
