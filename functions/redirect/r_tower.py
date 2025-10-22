"""Clean and reorient tower point clouds so the main axis aligns globally."""
import numpy as np
from .rot_with_axle import rot_with_axle, apply_rotation


def r_tower(tower_pts):
    """
    Tower reorientation - denoising and main direction alignment
    
    Denoising: removes potential noise points above the power tower point cloud
    Reorientation: aligns the main direction of the tower with the coordinate axis
    
    Parameters:
    -----------
    tower_pts : numpy.ndarray
        Tower point cloud array of shape (N, 3) with columns [X, Y, Z]
        
    Returns:
    --------
    tower_pts_r : numpy.ndarray
        Rotated tower point cloud
    theta : float
        Rotation angle in radians
        
    """
    top_indices = np.argsort(tower_pts[:, 2])[-10:][::-1]  # Sort descending, get top 10
    top_z = tower_pts[top_indices, 2]
    
    z_diff = np.diff(top_z)
    large_drops = np.where(z_diff < -1)[0]
    
    cut_ind = None
    if len(large_drops) > 0:
        cut_ind_idx = large_drops[-1] + 1
        if cut_ind_idx < len(top_indices):
            cut_ind = top_indices[cut_ind_idx]
    
    if cut_ind is not None:
        cut_z = tower_pts[cut_ind, 2]
        tower_pts = tower_pts[tower_pts[:, 2] <= cut_z, :]
    
    max_z = np.max(tower_pts[:, 2])
    top_3m_pts = tower_pts[tower_pts[:, 2] > max_z - 3, :]
    
    _, theta = rot_with_axle(top_3m_pts, 3)  # Rotate around Z-axis
    
    # Apply rotation to entire tower point cloud
    tower_pts_r = apply_rotation(tower_pts, theta, 'z')
    
    return tower_pts_r, theta
