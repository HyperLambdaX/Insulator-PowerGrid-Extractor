"""Fit left/right boundary lines for the first cross arm of a tower."""
import numpy as np
from scipy.spatial import ConvexHull
from ...project.sin_pro import sin_pro
from .ransac_fitline import ransac_fitline
from .boundary_2d import boundary_2d


def fit_plane2(points):
    """
    Fit left and right boundary lines for the first cross arm
    
    Parameters:
    -----------
    points : numpy.ndarray
        Cross arm tower points of shape (N, 3) [X, Y, Z]
        
    Returns:
    --------
    fleft : numpy.ndarray
        Left boundary line parameters [slope, intercept]
    fright : numpy.ndarray
        Right boundary line parameters [slope, intercept]
        
    Notes:
    ------
    This function extracts boundary points and fits left/right boundary lines
    for the first cross arm (horizontal insulator extraction).
    """
    plane_pts = points[:, [1, 2]]
    
    try:
        k = boundary_2d(plane_pts, shrink_factor=0.5)
        pts = plane_pts[k, :]
    except Exception as e:
        # Fallback if boundary extraction fails
        print(f"Warning: boundary_2d failed ({e}), using all points")
        pts = plane_pts.copy()
      
    # --------------------------------Delete upper and lower boundary points---------------------------
    sam_gap = 0.2
    
    # Project to Y axis
    pts_d, _ = sin_pro(pts, 2, sam_gap)
    
    mid_idx = int(np.ceil(len(pts_d) / 2))
    
    cut_ind1 = np.argmax(pts_d[:mid_idx])  # Index in first half
    cut_ind2 = np.argmax(pts_d[mid_idx:])   # Index in second half (relative)
    
    cut_pos1 = np.min(pts[:, 1]) + (cut_ind1 + 1) * sam_gap
    cut_pos2 = np.min(pts[:, 1]) + (mid_idx + cut_ind2) * sam_gap
    
    pts_c = pts[(pts[:, 1] >= cut_pos1) & (pts[:, 1] <= cut_pos2), :]
    
    if len(pts_c) == 0:
        # Fallback if filtering removes all points
        pts_c = pts.copy()
    
    # -----------------------------Fit the left and right straight lines separately------------------------
    mid_x = np.min(pts_c[:, 0]) + (np.max(pts_c[:, 0]) - np.min(pts_c[:, 0])) / 2
    
    pts_left = pts_c[pts_c[:, 0] < mid_x, :]
    
    if len(pts_left) > 2:
        fleft = ransac_fitline(pts_left, 10000, sam_gap)
    else:
        # Fallback if not enough points
        fleft = np.array([0, 0])
    
    pts_right = pts_c[pts_c[:, 0] > mid_x, :]
    
    if len(pts_right) > 2:
        fright = ransac_fitline(pts_right, 10000, sam_gap)
    else:
        # Fallback if not enough points
        fright = np.array([0, 0])
    
    return fleft, fright
