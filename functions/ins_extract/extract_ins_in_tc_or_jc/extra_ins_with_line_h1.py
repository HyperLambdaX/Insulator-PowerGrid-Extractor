"""Extract transverse insulators based on fitted tower-arm lines."""
import numpy as np
from .ins_in_h_line3 import ins_in_h_line3


def extra_ins_with_line_h1(l, t, line, is_up_cross, grid_width):
    """
    Extract insulator with horizontal line fitting
    
    Parameters:
    -----------
    l : numpy.ndarray
        Power line points of shape (M, 3) [X, Y, Z]
    t : numpy.ndarray
        Tower points of shape (N, 3) [X, Y, Z]
    line : numpy.ndarray
        Fitted boundary line parameters [slope, intercept]
    is_up_cross : int
        Flag: 1 if upper cross arm (vertical plane), 0 if horizontal plane
    grid_width : float
        Grid width for projection
        
    Returns:
    --------
    ins_pts : numpy.ndarray
        Extracted insulator points
    ins_len : float
        Length of insulator
        
    Notes:
    ------
    This function first cuts the power line and tower based on fitted line,
    then calls InsInHLine3 to extract insulators.
    """
    ins_pts = np.zeros((0, 3))
    ins_len = 0
    
    if is_up_cross:
        yd = np.min(l[:, 1]) + (np.max(l[:, 1]) - np.min(l[:, 1])) / 2
        zd = np.min(l[:, 2]) + (np.max(l[:, 2]) - np.min(l[:, 2])) / 2
        
        if line[0] * yd - zd + line[1] < 0:
            # The power line is to the left of the fitted straight line
            lp = l[line[0] * l[:, 1] - l[:, 2] + line[1] < 0, :]
            tp = t[line[0] * t[:, 1] - t[:, 2] + line[1] < 0, :]
        else:
            lp = l[line[0] * l[:, 1] - l[:, 2] + line[1] > 0, :]
            tp = t[line[0] * t[:, 1] - t[:, 2] + line[1] > 0, :]
    else:
        xd = np.min(l[:, 0]) + (np.max(l[:, 0]) - np.min(l[:, 0])) / 2
        yd = np.min(l[:, 1]) + (np.max(l[:, 1]) - np.min(l[:, 1])) / 2
        
        if line[0] * xd - yd + line[1] < 0:
            # The power line is to the left of the fitted straight line
            lp = l[line[0] * l[:, 0] - l[:, 1] + line[1] < 0, :]
            tp = t[line[0] * t[:, 0] - t[:, 1] + line[1] < 0, :]
        else:
            lp = l[line[0] * l[:, 0] - l[:, 1] + line[1] > 0, :]
            tp = t[line[0] * t[:, 0] - t[:, 1] + line[1] > 0, :]
    
    # Check if we have points after cutting
    if len(lp) == 0 or len(tp) == 0:
        return ins_pts, ins_len
    
    ins_pts1_r32, theta1, theta2 = ins_in_h_line3(t, lp, grid_width)
    
    if len(ins_pts1_r32) > 0:
        rot_z = _rotz(theta1 * 180 / np.pi)
        rot_y = _roty(theta2 * 180 / np.pi)
        tpr32 = tp @ rot_z @ rot_y
        
        ins_pts2_r32 = tpr32[
            (tpr32[:, 1] < np.max(ins_pts1_r32[:, 1])) & 
            (tpr32[:, 1] > np.min(ins_pts1_r32[:, 1])) &
            (tpr32[:, 2] < np.max(ins_pts1_r32[:, 2])) & 
            (tpr32[:, 2] > np.min(ins_pts1_r32[:, 2])), :]
        
        ins_pts_r32 = np.vstack([ins_pts1_r32, ins_pts2_r32])
        ins_len = np.max(ins_pts_r32[:, 0]) - np.min(ins_pts_r32[:, 0])
        
        rot_y_inv = _roty(-theta2 * 180 / np.pi)
        rot_z_inv = _rotz(-theta1 * 180 / np.pi)
        ins_pts = ins_pts_r32 @ rot_y_inv @ rot_z_inv
   
    return ins_pts, ins_len


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
