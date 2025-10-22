"""Identify the transmission-tower type from tower and line point clouds."""
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
from ..redirect.rot_with_axle import rot_with_axle, apply_rotation
from ..redirect.r_tower import r_tower
from ..project.bin_projection import bin_projection
from ..project.bin_pro import bin_pro
from ..draw_results.drow_zbarh import drow_zbarh
from ..max_label import max_label
from .cross_location import cross_location
from .o_tower_detect import o_tower_detect


def type_detect(tower_pts, line_pts):
    """
    Detect tower type based on point cloud characteristics
    
    Tower types:
    1 - Wine glass tower (O-shaped, 1 cross arm)
    2 - Cat head tower (O-shaped, multiple cross arms)
    3 - Single cross arm tower
    4 - Dry-font tower (2 cross arms, no cavity)
    5 - Tension type (multiple cross arms, large vertical gaps)
    6 - DC type (multiple cross arms, small vertical gaps)
    
    Parameters:
    -----------
    tower_pts : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    line_pts : numpy.ndarray
        Power line point cloud of shape (M, 3) [X, Y, Z]
        
    Returns:
    --------
    type_id : int
        Tower type identifier (1-6)
        
    """
    # This duplicates RTower functionality but is inline in the original
    top_indices = np.argsort(tower_pts[:, 2])[-10:][::-1]
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
    _, theta = rot_with_axle(top_3m_pts, 3)
    
    tower_pts_r = apply_rotation(tower_pts, theta, 'z')
    
    line_pts_r_cut = apply_rotation(line_pts, theta, 'z')
    
    grid_width1 = 0.2
    grid_width2 = 0.1
    
    # Void detection - project onto Y-Z plane (axes 1 and 3)
    bin_yz_o, _ = bin_projection(tower_pts_r, grid_width1, 1, 3)
    
    # Cross arm inspection
    bin_yz_c, _ = bin_projection(tower_pts_r, grid_width2, 1, 3)
    
    # Detect cross arm position
    loc = cross_location(bin_yz_c, 4)
    
    gan_num = loc.shape[0] if loc is not None and loc.size > 0 else 0
    
    if o_tower_detect(bin_yz_o, 1/2):
        if gan_num == 1:
            type_id = 1
        else:
            type_id = 2
    elif gan_num == 1 and o_tower_detect(bin_yz_o, 1/3):
        type_id = 1
    elif gan_num == 2:
        if o_tower_detect(bin_yz_o, 1/2, 500):
            type_id = 2
        else:
            type_id = 4
    elif gan_num == 1:
        type_id = 3
    else:
        dbscan_model = DBSCAN(eps=1, min_samples=1)
        labels = dbscan_model.fit_predict(line_pts_r_cut)
        
        max_label_val = max_label(labels)
        cross_line = line_pts_r_cut[labels == max_label_val, :]
        
        # Use void in vertical axis direction to determine tower type
        bin_xz = bin_pro(cross_line, grid_width2, 2, 3)
        
        zf = drow_zbarh(bin_xz, -2, 'epy')
        
        # Counted by pixel, 10 pixels is 1m
        if np.mean(zf) > 4:
            type_id = 5
        else:
            type_id = 6
    
    return type_id
