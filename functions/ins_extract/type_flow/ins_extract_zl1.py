"""Extract insulators for Type-2 cable-stayed (cat-head) transmission towers."""
import numpy as np
from sklearn.cluster import DBSCAN
from ...redirect.rot_with_axle import rot_with_axle, apply_rotation
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ...max_label import max_label
from ..fill_bin import fill_bin


def ins_extract_zl1(t, l, loc, grid_width):
    """
    Extract insulators for type 2 tower (cable-stayed cat head tower)
    
    Parameters:
    -----------
    t : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    l : list of numpy.ndarray
        Power line point clouds
    loc : numpy.ndarray
        Cross arm locations
    grid_width : float
        Grid width for processing
        
    Returns:
    --------
    ins_pts : list of numpy.ndarray
        Extracted insulator points
    is_cable : int
        Flag indicating if cable-stayed type (0 or 1)
    ins_len : numpy.ndarray
        Lengths of extracted insulators
        
    """
    ins_pts = [np.zeros((0, 3)) for _ in range(6)]
    ins_len = np.zeros(6)
    is_cable = 0
    
    if loc.shape[0] == 1:
        bin_xz1, _ = bin_projection(t, grid_width, 1, 3)
        
        up_img = bin_xz1[int(loc[0, 1]):, :]
        
        d_sum = drow_zbarh(up_img, 1, 'Dsum')
        loc_ind = np.argsort(d_sum)[-2:][::-1]
        
        new_row = np.array([[
            np.min(loc_ind) + 1 + loc[0, 1] - 1,
            np.max(loc_ind) + 1 + loc[0, 1] - 1
        ]])
        loc = np.vstack([new_row, loc])
    else:
        loc = loc[-2:, :]
    
    cross1 = l[0]
    
    cross1_r1, theta1 = rot_with_axle(cross1, 1)
    
    bin_xz2, _ = bin_projection(cross1_r1, grid_width, 1, 3)
    
    zw = drow_zbarh(bin_xz2, 1, 'wid')
    fzw = fill_bin(zw)
    
    max_w = np.max(fzw)
    mid_pos = int(np.ceil(fzw.shape[0] / 2))
    
    if max_w > 12:
        is_cable = 1
        
        max_v = np.max(fzw[0:mid_pos])
        
        if np.sum(fzw[0:mid_pos] == max_v) <= 3:
            # Create a mask for the slice and apply it
            mask = fzw[0:mid_pos] == max_v
            fzw_slice = fzw[0:mid_pos].copy()
            fzw_slice[mask] = max_v - 1
            fzw[0:mid_pos] = fzw_slice
        
        thre = np.max(fzw[0:mid_pos]) + 1
        
        candidates = np.where(fzw[mid_pos:] >= thre)[0]
        
        if len(candidates) > 0:
            cut_pos1 = mid_pos + candidates[0]
        else:
            cut_pos1 = None
        
        if cut_pos1 is not None:
            ins_pts1_r1 = cross1_r1[
                cross1_r1[:, 2] > np.min(cross1_r1[:, 2]) + cut_pos1 * grid_width, :
            ]
        else:
            ins_pts1_r1 = np.zeros((0, 3))
        
        ins_pts1 = apply_rotation(ins_pts1_r1, -theta1, 'x')

        # Check if loc has at least 2 rows before accessing loc[1,1]
        if loc.shape[0] >= 2:
            ct1 = t[
                (t[:, 2] > np.min(t[:, 2]) + grid_width * (loc[1, 1] + 1)) &
                (t[:, 2] < np.min(t[:, 2]) + grid_width * (loc[0, 0] - 2 + 1)), :
            ]
        else:
            # If loc doesn't have enough rows, skip this processing
            ct1 = np.zeros((0, 3))
        
        if ct1.shape[0] > 0:
            bin_xz, _ = bin_projection(ct1, grid_width, 1, 3)
            dwxz = drow_zbarh(bin_xz, -2, 'Dsum')
            
            mid_x = int(np.floor(dwxz.shape[0] / 2))
            cut_thre = 10
            
            candidates_t1 = np.where(dwxz[0:mid_x] > cut_thre)[0]
            cut_pos_t1 = candidates_t1[-1] if len(candidates_t1) > 0 else 0
            
            candidates_t2 = np.where(dwxz[mid_x:] > cut_thre)[0]
            cut_pos_t2 = mid_x + candidates_t2[0] if len(candidates_t2) > 0 else mid_x
            
            ct2 = ct1[
                (ct1[:, 0] > np.min(ct1[:, 0]) + grid_width * (cut_pos_t1 + 1)) &
                (ct1[:, 0] < np.min(ct1[:, 0]) + grid_width * (cut_pos_t2 + 1)), :
            ]
        else:
            ct2 = np.zeros((0, 3))
        
        if ct2.shape[0] > 0 and cut_pos1 is not None:
            ct2_r1 = apply_rotation(ct2, theta1, 'x')
            ct3_r1 = ct2_r1[
                ct2_r1[:, 2] > np.min(cross1_r1[:, 2]) + (cut_pos_t1 + 1) * grid_width, :
            ]
            ct3 = apply_rotation(ct3_r1, -theta1, 'x')
        else:
            ct3 = np.zeros((0, 3))
        
        ins_pts2 = np.vstack([ins_pts1, ct3]) if ct3.shape[0] > 0 else ins_pts1
        
        if ins_pts2.shape[0] > 5:
            dbscan_model = DBSCAN(eps=1, min_samples=5)
            labels = dbscan_model.fit_predict(ins_pts2)
            
            max_label_val = max_label(labels)
            ins = ins_pts2[labels == max_label_val, :]
            
            if ins.shape[0] > 0:
                length = np.max(ins[:, 2]) - np.min(ins[:, 2])
                ins_pts[0] = ins
                ins_len[0] = length
        
        k = [1]
    else:
        k = [0, 1]
    
    for i in k:
        # Check if we have enough rows in loc
        if loc.shape[0] <= i:
            continue

        if i >= len(l):
            continue

        cross_line = l[i]

        if cross_line.shape[0] == 0:
            continue

        dbscan_model = DBSCAN(eps=0.5, min_samples=1)
        labels = dbscan_model.fit_predict(cross_line)
        max_label_val = max_label(labels)
        _, theta1 = rot_with_axle(cross_line[labels == max_label_val, :], 3)

        cross_beg = np.min(t[:, 2]) + grid_width * (loc[i, 0] - 1 + 1)
        cross_end = np.min(t[:, 2]) + grid_width * (loc[i, 1] + 1)
        
        cross_end_pts = t[
            (t[:, 2] < cross_end) &
            (t[:, 2] > cross_beg - 0.2), :
        ]
        
        if cross_end_pts.shape[0] == 0:
            continue
        
        cross_end_pts_r = apply_rotation(cross_end_pts, theta1, 'z')
        
        cross_line_r3 = apply_rotation(cross_line, theta1, 'z')
        
        third_len = (np.max(cross_end_pts_r[:, 1]) - np.min(cross_end_pts_r[:, 1])) / 3
        min_y = np.min(cross_end_pts_r[:, 1])
        
        ins_in_tower = t[t[:, 2] < cross_beg, :]
        
        for j in range(3):
            third_len_pts = cross_line_r3[
                (cross_line_r3[:, 1] >= min_y + j * third_len) &
                (cross_line_r3[:, 1] <= min_y + (j + 1) * third_len), :
            ]
            third_len_pts = apply_rotation(third_len_pts, -theta1, 'z')
            
            ins, length = ins_extrat_partone(third_len_pts, ins_in_tower, grid_width)
            
            ins_pts[i * 3 + j] = ins
            ins_len[i * 3 + j] = length
    
    return ins_pts, is_cable, ins_len


def ins_extrat_partone(line, tower, grid_width):
    """
    Extract insulators from individual power lines
    
    Notes:
    ------
    Same as in ins_extract_zl.py
    """
    ins = np.zeros((0, 3))
    length = 0
    
    if line.shape[0] < 5:
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
        
        ins_tr = tower_r[
            (tower_r[:, 0] < np.max(line_r32[:, 0])) &
            (tower_r[:, 0] > np.min(line_r32[:, 0])) &
            (tower_r[:, 1] < np.max(line_r32[:, 1])) &
            (tower_r[:, 1] > np.min(line_r32[:, 1])) &
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
    
    Notes:
    ------
    Same as in ins_extract_zl.py
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
    
    cut_pos_ind = cut_pos_ind_arr[0]
    cut_pos = beg[cut_pos_ind]
    
    return cut_pos
