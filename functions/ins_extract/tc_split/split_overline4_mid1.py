"""Split mid-span crossover lines away from the main power-line segments."""
import numpy as np
from ...redirect.rot_with_axle import rot_with_axle
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ..detect_muta_in_bin import detect_muta_in_bin


def split_overline4_mid1(cross_pts, grid_width):
    """
    Separate power lines and crossover lines located in the middle of the tower
    
    Parameters:
    -----------
    cross_pts : numpy.ndarray
        Cross arm points of shape (N, 3) [X, Y, Z]
    grid_width : float
        Grid width for projection
        
    Returns:
    --------
    line_pts : numpy.ndarray
        Separated power line points
        
    Notes:
    ------
    This function recursively cuts the point cloud to separate power lines
    from crossover lines using projection-based methods.
    """
    half_len = (np.max(cross_pts[:, 1]) - np.min(cross_pts[:, 1])) / 2
    mid_yl = np.min(cross_pts[:, 1]) + half_len
    
    line_pts = np.zeros((0, 3))
    
    for i in range(1, 3):  # 1 and 2
        half_cross = cross_pts[
            (cross_pts[:, 1] > np.min(cross_pts[:, 1]) + half_len * (i - 1)) &
            (cross_pts[:, 1] < np.min(cross_pts[:, 1]) + half_len * i), :]
        
        if len(half_cross) == 0:
            continue
        
        d_min = np.abs(mid_yl - np.min(half_cross[:, 1]))
        d_max = np.abs(mid_yl - np.max(half_cross[:, 1]))
        part_len = (np.max(half_cross[:, 1]) - np.min(half_cross[:, 1])) / 3
        
        if d_min < d_max:
            # The smaller Y coordinate value is closer to the power tower
            edgeline = half_cross[half_cross[:, 1] > np.max(half_cross[:, 1]) - part_len, :]
        else:
            # The larger Y coordinate value is closer to the power tower
            edgeline = half_cross[half_cross[:, 1] < np.min(half_cross[:, 1]) + part_len, :]
        
        if len(edgeline) == 0:
            edgeline = half_cross.copy()
        
        edgeline_r3, theta1 = rot_with_axle(edgeline, 3)
        edgeline_r32, theta2 = rot_with_axle(edgeline_r3, 2)
        
        rot_z = _rotz(np.rad2deg(theta1))
        rot_y = _roty(np.rad2deg(theta2))
        
        half_cross_r32 = half_cross @ rot_z @ rot_y
        
        half_cross_r32_r = half_cross_r32[:, [0, 2, 1]]
        
        bin_xz, _ = bin_projection(half_cross_r32_r, grid_width, 1, 3)
        max_wid_ind = np.argmax(drow_zbarh(bin_xz, 1, 'wid')) + 1
        
        if max_wid_ind > bin_xz.shape[0] / 2:
            # Insulator on top
            line_r32 = _cut1(half_cross_r32_r, grid_width)
            line_r32 = _cut2(line_r32, grid_width)
        else:
            # Insulator below
            half_cross_r32_r[:, 2] = -half_cross_r32_r[:, 2]
            line_r32 = _cut1(half_cross_r32_r, grid_width)
            line_r32 = _cut2(line_r32, grid_width)
            line_r32[:, 2] = -line_r32[:, 2]
        
        rot_y_inv = _roty(-np.rad2deg(theta2))
        rot_z_inv = _rotz(-np.rad2deg(theta1))
        
        line = line_r32[:, [0, 2, 1]] @ rot_y_inv @ rot_z_inv
        
        line_pts = np.vstack([line_pts, line])
    
    return line_pts


def _cut1(l, grid_width):
    """
    First cut operation
    
    Parameters:
    -----------
    l : numpy.ndarray
        Point cloud to cut
    grid_width : float
        Grid width for projection
        
    Returns:
    --------
    l_result : numpy.ndarray
        Points above the cut position (insulators)
    """
    bin_xz, _ = bin_projection(l, grid_width, 1, 3)
    z_wid = drow_zbarh(bin_xz, 1, 'wid')
    d_z_wid = np.diff(z_wid)
    nd_z_wid = detect_muta_in_bin(d_z_wid)
    
    cut_indices = np.where(nd_z_wid > 35)[0]
    if len(cut_indices) > 0:
        cutpos = cut_indices[0]  # first element
    else:
        cutpos = 0
    
    l_result = l[l[:, 2] > np.min(l[:, 2]) + (cutpos + 1) * grid_width, :]
    
    if len(l_result) == 0:
        # Fallback to original if cut removes everything
        l_result = l.copy()
    
    return l_result


def _cut2(l, grid_width):
    """
    Second cut operation
    
    Parameters:
    -----------
    l : numpy.ndarray
        Point cloud to cut
    grid_width : float
        Grid width for projection
        
    Returns:
    --------
    l_result : numpy.ndarray
        Points above the cut position
    """
    bin_xy, _ = bin_projection(l, grid_width, 1, 2)
    z_wid = drow_zbarh(bin_xy, 1, 'wid')
    d_z_wid = np.diff(z_wid)
    nd_z_wid = detect_muta_in_bin(d_z_wid)
    cutpos = np.argmax(nd_z_wid)
    
    if cutpos > 0:
        l_result = l[l[:, 1] > np.min(l[:, 1]) + (cutpos + 1) * grid_width, :]
        
        if len(l_result) == 0:
            # Fallback
            l_result = l.copy()
    else:
        l_result = l.copy()
    
    return l_result


def _rotz(angle_deg):
    """Rotation matrix around Z axis"""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def _roty(angle_deg):
    """Rotation matrix around Y axis"""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
