"""Split power lines by clustering their 2D grid projection."""
import numpy as np
from sklearn.cluster import DBSCAN
from ...project.bin_projection import bin_projection


def split_power_line_2d(line_pts, grid_width, get_num, eps):
    """
    Split power line using 2D projection clustering
    
    Parameters:
    -----------
    line_pts : numpy.ndarray
        Line point cloud of shape (N, 3) [X, Y, Z]
    grid_width : float
        Grid width for projection
    get_num : int
        Number of power lines to extract
    eps : float
        DBSCAN epsilon parameter
        
    Returns:
    --------
    cross_line : list of numpy.ndarray
        List containing separated power line point clouds
        Ordered from top to bottom (first is highest)
    """
    bin_xz, grid = bin_projection(line_pts, grid_width, 2, 3)
    
    # Get positions of non-zero elements
    pos_x, pos_y = np.where(bin_xz.T)
    
    # Clustering using DBSCAN
    # But for clustering we use the actual values
    dbscan_model = DBSCAN(eps=eps, min_samples=1)
    idx = dbscan_model.fit_predict(np.column_stack([pos_x, pos_y]))
    
    labels = np.unique(idx)
    
    labels_num = len(labels)
    
    # Calculate the lowest point coordinates of each cluster
    min_cor = np.zeros(labels_num)
    
    for i in range(labels_num):
        min_cor[i] = np.min(pos_y[idx == labels[i]])
    
    # Sort by minimum coordinate (ascending)
    min_ind = np.argsort(min_cor)
    
    # Extract the lowest GetNum clusters
    cross_line = []
    
    for i in range(get_num):
        # Grid row coordinates
        grid_py = pos_y[idx == labels[min_ind[i]]]
        
        # Grid column coordinates
        grid_px = pos_x[idx == labels[min_ind[i]]]
        
        cur_line_ind = []
        
        for j in range(len(grid_py)):
            cur_line_ind.extend(grid[grid_py[j]][grid_px[j]])
        
        # Reverse order: lowest becomes last, highest becomes first
        cross_line.append(line_pts[cur_line_ind, :])
    
    # Reverse the list so first element is the highest line
    cross_line = cross_line[::-1]
    
    return cross_line
