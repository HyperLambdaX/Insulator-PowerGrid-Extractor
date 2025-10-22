"""Merge segmented insulator cells and compute their lengths."""
import numpy as np
from .redirect.rot_with_axle import rot_with_axle


def merge_cell3(cell_pts):
    """
    Merge cell data and calculate lengths
    
    Parameters:
    -----------
    cell_pts : list of lists or 2D list
        Cell array containing point cloud segments
        Each cell contains a numpy array of shape (N, 3)
        
    Returns:
    --------
    m_pts : numpy.ndarray
        Merged points with label column, shape (M, 4) [X, Y, Z, Label]
    lens : numpy.ndarray
        Array of lengths for each segment
        
    """
    m_pts = np.zeros((0, 4))
    
    lens = []
    
    k = 0
    
    # Iterate through cell array
    if not isinstance(cell_pts, (list, tuple)):
        # If it's a 1D list, convert to 2D
        cell_pts = [cell_pts] if not isinstance(cell_pts[0], (list, tuple)) else cell_pts
    
    for i in range(len(cell_pts)):
        row = cell_pts[i]
        # Handle both 1D and 2D cell structures
        if not isinstance(row, (list, tuple)):
            row = [row]
            
        for j in range(len(row)):
            pts_k = row[j]
            
            # Skip if empty or not array
            if pts_k is None or (isinstance(pts_k, np.ndarray) and pts_k.size == 0):
                pts_k = np.array([])
            elif not isinstance(pts_k, np.ndarray):
                continue
                
            if pts_k.shape[0] > 5:
                lj_r3, _ = rot_with_axle(pts_k, 3)
                
                lj_r32, _ = rot_with_axle(lj_r3, 2)
                
                length = np.max(lj_r32[:, 0]) - np.min(lj_r32[:, 0])
            else:
                length = 0
            
            lens.append(length)
            
            k = k + 1
            
            if pts_k.size > 0 and not (np.sum(pts_k[0, :]) == 0):
                # Add label column
                labels = np.full((pts_k.shape[0], 1), k)
                pts_with_label = np.hstack([pts_k, labels])
                m_pts = np.vstack([m_pts, pts_with_label])
    
    lens = np.array(lens)
    
    return m_pts, lens
