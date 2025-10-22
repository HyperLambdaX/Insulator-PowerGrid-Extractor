"""Extract insulators from horizontal power lines (method variant 3)."""
import numpy as np
from ...redirect.rot_with_axle import rot_with_axle
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ..fill_bin import fill_bin


def ins_in_h_line3(t, l, grid_width):
    """
    Extract insulator from horizontal power lines (Method 3)
    
    Parameters:
    -----------
    t : numpy.ndarray
        Tower points of shape (N, 3) [X, Y, Z]
    l : numpy.ndarray
        Power line points of shape (M, 3) [X, Y, Z]
    grid_width : float
        Grid width for projection
        
    Returns:
    --------
    ins_pts : numpy.ndarray
        Extracted insulator points (in rotated coordinates)
    theta1 : float
        First rotation angle (around Z axis) in radians
    theta2 : float
        Second rotation angle (around Y axis) in radians
        
    Notes:
    ------
    This function extracts insulator by rotating and projecting to XY/XZ planes.
    Returns points in rotated coordinates - caller should rotate back.
    """
    ins_pts = np.zeros((0, 3))
    
    lr3, theta1 = rot_with_axle(l, 3)
    lr32, theta2 = rot_with_axle(lr3, 2)
    
    rot_z = _rotz(theta1 * 180 / np.pi)
    rot_y = _roty(theta2 * 180 / np.pi)
    tr32 = t @ rot_z @ rot_y
    
    tx_mid = np.min(tr32[:, 0]) + (np.max(tr32[:, 0]) - np.min(tr32[:, 0])) / 2
    
    bin_xy, _ = bin_projection(lr32, grid_width, 1, 2)
    xy_wid = drow_zbarh(bin_xy, -2, 'wid')
    f_xy_wid = fill_bin(xy_wid)
    max_ind = np.argmax(xy_wid)
    mid_pos = int(np.ceil(len(f_xy_wid) / 2))
    
    is_cut = False
    
    if np.abs(tx_mid - np.min(lr32[:, 0])) > np.abs(tx_mid - np.max(lr32[:, 0])):
        # Insulator on the far side from tower (high X values)
        max_v = np.max(f_xy_wid[:mid_pos])
        if np.sum(f_xy_wid[:mid_pos] == max_v) <= 3:
            # Find indices in the range [:mid_pos] where value equals max_v
            indices = np.where(f_xy_wid[:mid_pos] == max_v)[0]
            f_xy_wid[indices] = max_v - 1
        
        thre = np.max(f_xy_wid[:mid_pos]) + 1
        
        cut_indices = np.where(f_xy_wid[mid_pos:] > thre)[0]
        if len(cut_indices) > 0:
            cut_pos = mid_pos + cut_indices[0]
            threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
            ins_pts = lr32[lr32[:, 0] > threshold, :]
            is_cut = True
    else:
        # Insulator on the near side to tower (low X values)
        max_v = np.max(f_xy_wid[mid_pos:])
        if np.sum(f_xy_wid[mid_pos:] == max_v) <= 3:
            # Find indices in the range [mid_pos:] where value equals max_v
            indices = np.where(f_xy_wid[mid_pos:] == max_v)[0]
            f_xy_wid[indices] = max_v - 1
        
        thre = np.max(f_xy_wid[mid_pos:]) + 1
        
        cut_indices = np.where(f_xy_wid[:mid_pos] > thre)[0]
        if len(cut_indices) > 0:
            cut_pos = cut_indices[-1] + 1
            threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
            ins_pts = lr32[lr32[:, 0] < threshold, :]
            is_cut = True
    
    if not is_cut:
        bin_xz, _ = bin_projection(lr32, grid_width, 1, 3)
        xz_wid = drow_zbarh(bin_xz, -2, 'wid')
        f_xz_wid = fill_bin(xz_wid)
        
        if np.abs(tx_mid - np.min(lr32[:, 0])) > np.abs(tx_mid - np.max(lr32[:, 0])):
            thre = 4
            cut_indices = np.where(f_xz_wid > thre)[0]
            if len(cut_indices) > 0:
                cut_pos = cut_indices[-1] + 1
                threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
                ins_pts = lr32[lr32[:, 0] > threshold, :]
        else:
            thre = 4
            cut_indices = np.where(f_xz_wid > thre)[0]
            if len(cut_indices) > 0:
                cut_pos = cut_indices[0] + 1
                threshold = np.min(lr32[:, 0]) + cut_pos * grid_width
                ins_pts = lr32[lr32[:, 0] < threshold, :]
    
    return ins_pts, theta1, theta2


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
