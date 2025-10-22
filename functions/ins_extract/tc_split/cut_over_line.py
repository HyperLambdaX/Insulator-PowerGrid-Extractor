"""Separate jumper wires from power lines using fitted boundary lines."""
import numpy as np
from ...redirect.rot_with_axle import rot_with_axle
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ..fill_bin import fill_bin


def cut_over_line(l, grid_width, line, mid_yt):
    """
    Cut off the overline from power lines based on fitted boundary line
    
    Parameters:
    -----------
    l : numpy.ndarray
        Power line points of shape (N, 3) [X, Y, Z]
    grid_width : float
        Grid width for projection
    line : numpy.ndarray
        Fitted boundary line parameters [slope, intercept]
    mid_yt : float
        Midpoint of tower Y coordinate
        
    Returns:
    --------
    lres : numpy.ndarray
        Remaining power line points (insulators)
    ores : numpy.ndarray
        Overline points (jumpers)
        
    Notes:
    ------
    This function separates insulators from jumpers (crossover lines)
    by rotating the point cloud and projecting to XZ plane.
    """
    xd = np.min(l[:, 0]) + (np.max(l[:, 0]) - np.min(l[:, 0])) / 2
    yd = np.min(l[:, 1]) + (np.max(l[:, 1]) - np.min(l[:, 1])) / 2
    
    if line[0] * xd - yd + line[1] < 0:
        # The power line is to the left of the fitted straight line
        lc = l[line[0] * l[:, 0] - l[:, 1] + line[1] < 0, :]
    else:
        lc = l[line[0] * l[:, 0] - l[:, 1] + line[1] > 0, :]
    
    if len(lc) == 0:
        # Fallback if no points on that side
        lc = l.copy()
    
    d_min = np.abs(mid_yt - np.min(lc[:, 1]))
    d_max = np.abs(mid_yt - np.max(lc[:, 1]))
    part_len = (np.max(lc[:, 1]) - np.min(lc[:, 1])) / 3

    if d_min < d_max:
        # The smaller Y coordinate value is closer to the power tower
        edgeline = lc[lc[:, 1] > np.max(lc[:, 1]) - part_len, :]
    else:
        # The larger Y coordinate value is closer to the power tower
        edgeline = lc[lc[:, 1] < np.min(lc[:, 1]) + part_len, :]
    
    if len(edgeline) == 0:
        # Fallback
        edgeline = lc.copy()
    
    edgeline_r3, theta1 = rot_with_axle(edgeline, 3)
    edgeline_r32, theta2 = rot_with_axle(edgeline_r3, 2)
    
    # Create rotation matrices
    rot_z = _rotz(np.rad2deg(theta1))
    rot_y = _roty(np.rad2deg(theta2))
    
    lcr32 = lc @ rot_z @ rot_y
    lr32 = l @ rot_z @ rot_y

    bin_xz, _ = bin_projection(lcr32, grid_width, 1, 3)
    xz_wid = drow_zbarh(bin_xz, 1, 'wid')
    f_xz_wid = fill_bin(xz_wid)
    max_ind = np.argmax(f_xz_wid)
    
    mid_pos = int(np.ceil(len(f_xz_wid) / 2))
    
    if max_ind > mid_pos:
        # Insulator on top
        thre = np.max(f_xz_wid) / 2
        
        cut_indices = np.where(f_xz_wid[mid_pos:] > thre)[0]
        if len(cut_indices) > 0:
            cut_pos = mid_pos + cut_indices[0]
        else:
            cut_pos = mid_pos
        
        lres_r32 = lr32[lr32[:, 2] > np.min(lcr32[:, 2]) + cut_pos * grid_width, :]
        ores_r32 = lr32[lr32[:, 2] <= np.min(lcr32[:, 2]) + cut_pos * grid_width, :]
    else:
        # Insulator below
        thre = np.max(f_xz_wid) / 2

        cut_indices = np.where(f_xz_wid[:mid_pos] > thre)[0]
        if len(cut_indices) > 0:
            cut_pos = cut_indices[-1] + 1
        else:
            cut_pos = 0
        
        lres_r32 = lr32[lr32[:, 2] < np.min(lcr32[:, 2]) + cut_pos * grid_width, :]
        ores_r32 = lr32[lr32[:, 2] >= np.min(lcr32[:, 2]) + cut_pos * grid_width, :]
    
    rot_y_inv = _roty(-np.rad2deg(theta2))
    rot_z_inv = _rotz(-np.rad2deg(theta1))
    
    lres = lres_r32 @ rot_y_inv @ rot_z_inv
    ores = ores_r32 @ rot_y_inv @ rot_z_inv
    
    return lres, ores


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
