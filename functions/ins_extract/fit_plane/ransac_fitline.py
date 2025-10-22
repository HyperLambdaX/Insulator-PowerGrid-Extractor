"""Robust 2D line fitting using a RANSAC-style sampling procedure."""
import numpy as np


def ransac_fitline(pts, iter_num, t):
    """
    RANSAC algorithm for line fitting in 2D
    
    Parameters:
    -----------
    pts : numpy.ndarray
        2D points of shape (N, 2) [X, Y]
    iter_num : int
        Number of RANSAC iterations
    t : float
        Distance threshold from point to line
        
    Returns:
    --------
    bestline : numpy.ndarray
        Line parameters [slope, intercept] where y = slope*x + intercept
        
    Notes:
    ------
    This function uses RANSAC to robustly fit a line to 2D points,
    sampling from both upper and lower halves to ensure good coverage.
    """
    points = np.zeros_like(pts)
    points[:, 0] = pts[:, 0] - np.min(pts[:, 0])
    points[:, 1] = pts[:, 1] - np.min(pts[:, 1])
    
    max_num = 0  # The number of optimal fitting interior points
    fin_dist = None  # Will store the final distances
    
    for j in range(iter_num):
        mid_y = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
        
        d_pts = points[points[:, 1] < mid_y, :]  # Points in the lower half
        u_pts = points[points[:, 1] > mid_y, :]  # Points in the upper half
        
        # Check if we have enough points in both halves
        if len(d_pts) == 0 or len(u_pts) == 0:
            continue
        
        sample_num = min(int(np.ceil(len(d_pts) / 3)), int(np.ceil(len(u_pts) / 3)))
        
        if sample_num == 0:
            continue
        
        sample_idx_d = np.random.permutation(len(d_pts))[:sample_num]
        sample_idx_u = np.random.permutation(len(u_pts))[:sample_num]
        
        sam_pts = np.vstack([d_pts[sample_idx_d, :], u_pts[sample_idx_u, :]])
        
        # Preliminary line fitting
        f = np.polyfit(sam_pts[:, 0], sam_pts[:, 1], 1)
        
        # Calculate distance from each point to the line
        distance = np.abs(f[0] * points[:, 0] - points[:, 1] + f[1]) / np.sqrt(f[0]**2 + 1)
        
        near_sum = np.sum(distance < t)
        
        if near_sum > sample_num:
            near_pts = points[distance < t, :]
            f = np.polyfit(near_pts[:, 0], near_pts[:, 1], 1)
            distance = np.abs(f[0] * points[:, 0] - points[:, 1] + f[1]) / np.sqrt(f[0]**2 + 1)
            near_sum = np.sum(distance < t)
        
        if near_sum > max_num:
            fin_dist = distance.copy()
            max_num = near_sum
    
    # Handle case where no good fit was found (fin_dist is None)
    if fin_dist is None:
        # Fallback: use all points for fitting
        best_pts = pts
    else:
        best_pts = pts[fin_dist < t, :]

    # Ensure we have enough points for fitting
    if len(best_pts) < 2:
        # Not enough points, use all input points
        best_pts = pts

    bestline = np.polyfit(best_pts[:, 0], best_pts[:, 1], 1)
    
    if np.abs(bestline[0]) < 20:
        mid_y = np.min(best_pts[:, 1]) + (np.max(best_pts[:, 1]) - np.min(best_pts[:, 1])) / 2
        d_pts = best_pts[best_pts[:, 1] < mid_y, :]
        u_pts = best_pts[best_pts[:, 1] > mid_y, :]
        
        if len(d_pts) > 0 and len(u_pts) > 0:
            sample_num = min(int(np.ceil(len(d_pts) / 3)), int(np.ceil(len(u_pts) / 3)))
            
            if sample_num > 0:
                sample_idx_d = np.random.permutation(len(d_pts))[:sample_num]
                sample_idx_u = np.random.permutation(len(u_pts))[:sample_num]
                best_pts = np.vstack([d_pts[sample_idx_d, :], u_pts[sample_idx_u, :]])
                bestline = np.polyfit(best_pts[:, 0], best_pts[:, 1], 1)
    
    return bestline
