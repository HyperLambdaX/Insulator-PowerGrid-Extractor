"""Extract longitudinal insulators for Type-1, Type-6, and Type-8 towers."""
import numpy as np
from sklearn.cluster import DBSCAN
from ...redirect.rot_with_axle import rot_with_axle, apply_rotation
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ...max_label import max_label


def ins_extract_zl(tower_pts, line_pts, loc, grid_width, tower_type):
    """
    Extract insulators for type 1, 6, 8 towers (vertical insulators)
    
    Parameters:
    -----------
    tower_pts : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    line_pts : list of numpy.ndarray
        Power line point clouds, each element is shape (M, 3)
    loc : numpy.ndarray
        Cross arm locations of shape (K, 2) [start, end]
    grid_width : float
        Grid width for processing
    tower_type : int
        Tower type (1, 6, or 8)
        
    Returns:
    --------
    ins_pts : list of numpy.ndarray
        Extracted insulator points for each phase
    ins_len : numpy.ndarray
        Lengths of extracted insulators
        
    """
    if tower_type in [1, 8]:
        cross_num = 1
        loc = loc[-1:, :] if loc.ndim == 2 else loc.reshape(1, -1)
        
    elif tower_type == 2:
        cross_num = 2
        if loc.shape[0] != cross_num:
            loc = merge_loc(loc, cross_num)
            
    elif tower_type == 6:
        loc = loc[-3:, :]
        cross_num = 3
        
    else:
        raise ValueError(f"Undefined tower type: {tower_type}")
    
    ins_pts = [np.zeros((0, 3)) for _ in range(3 * cross_num)]
    
    ins_len = np.zeros(3 * cross_num)
    
    for i in range(cross_num):
        cross_line = line_pts[i]
        
        if cross_line.shape[0] == 0:
            continue
        
        dbscan_model = DBSCAN(eps=0.5, min_samples=1)
        labels = dbscan_model.fit_predict(cross_line)
        
        max_label_val = max_label(labels)
        main_line = cross_line[labels == max_label_val, :]
        _, theta1 = rot_with_axle(main_line, 3)
        
        cross_beg = np.min(tower_pts[:, 2]) + grid_width * (loc[i, 0] + 1)
        
        cross_end = np.min(tower_pts[:, 2]) + grid_width * (loc[i, 1] + 1)
        
        cross_end_pts = tower_pts[
            (tower_pts[:, 2] < cross_end) & 
            (tower_pts[:, 2] > cross_beg - 3 * grid_width), :
        ]
        
        cross_end_pts_r = apply_rotation(cross_end_pts, theta1, 'z')
        
        ins_in_tower = tower_pts[tower_pts[:, 2] < cross_beg, :]
        
        cross_line_c = cross_line[cross_line[:, 2] < cross_beg, :]
        
        cross_line_cr3 = apply_rotation(cross_line_c, theta1, 'z')
        
        if cross_end_pts_r.shape[0] > 0:
            third_len = (np.max(cross_end_pts_r[:, 1]) - np.min(cross_end_pts_r[:, 1])) / 3
            min_y = np.min(cross_end_pts_r[:, 1])
        else:
            # If no cross end points, skip this cross arm
            continue
        
        thied_cell = []
        
        thied_cell.append(cross_line_cr3[cross_line_cr3[:, 1] < min_y + third_len, :])
        
        thied_cell.append(cross_line_cr3[
            (cross_line_cr3[:, 1] >= min_y + third_len) &
            (cross_line_cr3[:, 1] < min_y + 2 * third_len), :
        ])
        
        thied_cell.append(cross_line_cr3[cross_line_cr3[:, 1] > min_y + 2 * third_len, :])
        
        for j in range(3):
            third_len_pts = apply_rotation(thied_cell[j], -theta1, 'z')
            
            if third_len_pts.shape[0] < 5:
                continue
            
            ins, length = ins_extrat_partone(third_len_pts, ins_in_tower, grid_width)
            
            if ins.shape[0] > 0:
                ins_pts[i * 3 + j] = ins
                
                ins_len[i * 3 + j] = length
    
    return ins_pts, ins_len


def ins_extrat_partone(line, tower, grid_width):
    """
    Extracting insulators from individual power lines
    
    Parameters:
    -----------
    line : numpy.ndarray
        Power line points of shape (N, 3)
    tower : numpy.ndarray
        Tower points of shape (M, 3)
    grid_width : float
        Grid width for processing
        
    Returns:
    --------
    ins : numpy.ndarray
        Extracted insulator points
    length : float
        Length of insulator
        
    """
    ins = np.zeros((0, 3))
    length = 0
    
    if line.shape[0] == 0:
        return ins, length
    
    line_r3, theta1 = rot_with_axle(line, 3)
    
    line_r32, theta2 = rot_with_axle(line_r3, 2)
    
    bin_xy, _ = bin_projection(line_r32, grid_width, 1, 3)
    
    cut_pos = get_cutpos1(bin_xy)
    
    if cut_pos is not None:
        ins_lr = line_r32[line_r32[:, 2] > np.min(line_r32[:, 2]) + (cut_pos + 1) * grid_width, :]
        
        if ins_lr.shape[0] == 0:
            return ins, length
        
        tower_r = apply_rotation(tower, theta1, 'z')
        tower_r = apply_rotation(tower_r, theta2, 'y')
        
        min_z_lr = np.min(ins_lr[:, 2])
        max_z_lr = np.max(ins_lr[:, 2])
        z_threshold = min_z_lr + (max_z_lr - min_z_lr) / 3 * 2
        ins_lrc = ins_lr[(ins_lr[:, 2] >= min_z_lr) & (ins_lr[:, 2] <= z_threshold), :]
        
        if ins_lrc.shape[0] == 0:
            return ins, length
        
        ins_tr = tower_r[
            (tower_r[:, 0] < np.max(ins_lrc[:, 0])) &
            (tower_r[:, 0] > np.min(ins_lrc[:, 0])) &
            (tower_r[:, 1] < np.max(ins_lrc[:, 1])) &
            (tower_r[:, 1] > np.min(ins_lrc[:, 1])) &
            (tower_r[:, 2] > np.min(ins_lr[:, 2])), :
        ]
        
        ins_r = np.vstack([ins_lr, ins_tr])
        
        length = np.max(ins_r[:, 2]) - np.min(ins_r[:, 2])
        
        ins_r = apply_rotation(ins_r, -theta2, 'y')
        ins_r = apply_rotation(ins_r, -theta1, 'z')
        ins = np.vstack([ins, ins_r])
    
    return ins, length


def get_cutpos1(bin_img):
    """
    Using hole detection to determine crop location
    
    Parameters:
    -----------
    bin_img : numpy.ndarray
        Binary projection image
        
    Returns:
    --------
    cut_pos : int or None
        Cut position index
        
    """
    if bin_img.size == 0:
        return None
    
    epy = drow_zbarh(bin_img, 1, 'epy')
    
    epy0 = epy.copy()
    epy0[epy0 != 0] = -1
    epy0[epy0 == 0] = 1
    epy0[epy0 == -1] = 0
    
    d_epy0 = np.diff(epy0)
    
    beg = np.where(d_epy0 == 1)[0]
    
    if len(beg) == 0:
        return None
    
    ds = drow_zbarh(bin_img, 1, 'Dsum')
    dwid_max_ind = np.argmax(ds)
    
    cut_pos_ind_arr = np.where((beg + 1) - (dwid_max_ind + 1) >= 0)[0]
    
    if len(cut_pos_ind_arr) == 0:
        return None
    
    # First occurrence
    cut_pos_ind = cut_pos_ind_arr[0]
    
    cut_pos = beg[cut_pos_ind]
    
    return cut_pos


def merge_loc(loc, cross_num):
    """
    Merge location array to desired number of cross arms
    
    Parameters:
    -----------
    loc : numpy.ndarray
        Location array of shape (K, 2)
    cross_num : int
        Desired number of cross arms
        
    Returns:
    --------
    loc : numpy.ndarray
        Merged location array of shape (cross_num, 2)
        
    """
    while loc.shape[0] > cross_num:
        gap = loc[:-1, 1] - loc[1:, 0]
        
        mingap_ind = np.argmin(gap)
        
        loc[mingap_ind, 0] = loc[mingap_ind + 1, 0]
        
        # Delete row mingap_ind + 1
        loc = np.delete(loc, mingap_ind + 1, axis=0)
    
    return loc
