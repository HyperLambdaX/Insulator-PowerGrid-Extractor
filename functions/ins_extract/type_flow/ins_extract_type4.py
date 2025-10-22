"""Extract horizontal and vertical insulators for Type-3 and Type-4 towers."""
import numpy as np
from sklearn.cluster import DBSCAN
from ...max_label import max_label
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ..fit_plane.fit_plane1 import fit_plane1
from ..fit_plane.fit_plane2 import fit_plane2
from ..tc_split.split_overline4_mid1 import split_overline4_mid1
from ..tc_split.cut_over_line import cut_over_line
from ..extract_ins_in_tc_or_jc.extra_ins_with_line_h1 import extra_ins_with_line_h1
from ..remove_duplicate_points import remove_duplicate_points
from ...redirect.rot_with_axle import rot_with_axle


def ins_extract_type4(t, l, loc, grid_width, tower_type):
    """
    Insulator extraction for Type 3 and Type 4 towers
    
    Parameters:
    -----------
    t : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    l : list of numpy.ndarray
        List of power line point clouds
    loc : numpy.ndarray
        Cross arm locations of shape (M, 2) [start, end]
    grid_width : float
        Grid width for projection
    tower_type : int
        Tower type (3 or 4)
        
    Returns:
    --------
    ins_pts : list
        List of extracted insulator point clouds (12 elements)
    ins_len : numpy.ndarray
        Array of insulator lengths (12 elements)
        
    Notes:
    ------
    This function handles horizontal and vertical insulator extraction
    for single cross arm towers (Type 3) and tension resistant dry towers (Type 4).
    Includes nested helper function findMuta.
    """
    ins_pts = [np.array([[0, 0, 0]]) for _ in range(12)]
    ins_len = np.zeros(12)
    
    # ==================== First cross arm insulator extraction (horizontal) ====================

    cross_line1 = l[0] if len(l) > 0 else np.zeros((0, 3))
    
    if len(cross_line1) == 0:
        return ins_pts, ins_len
    
    cross_end = np.min(t[:, 2]) + grid_width * (loc[0, 0] + 1)
    
    cross_tower_pts1 = t[(t[:, 2] < cross_end - 1) & (t[:, 2] > np.min(cross_line1[:, 2])), :]
    
    if len(cross_tower_pts1) == 0:
        return ins_pts, ins_len
    
    fleft, fright = fit_plane2(cross_tower_pts1)
    
    fit_line = np.vstack([fleft, fright])
    
    cross_line_pts1 = split_overline4_mid1(cross_line1, grid_width)
    
    if len(cross_line_pts1) == 0:
        cross_line_pts1 = cross_line1.copy()
    
    mid_t_pos = np.min(cross_line_pts1[:, 1]) + (np.max(cross_line_pts1[:, 1]) - np.min(cross_line_pts1[:, 1])) / 2
    
    cross_cell = [
        cross_line_pts1[cross_line_pts1[:, 1] < mid_t_pos, :],
        cross_line_pts1[cross_line_pts1[:, 1] > mid_t_pos, :]
    ]
    
    for i in range(2):
        cross = cross_cell[i]
        
        if len(cross) == 0:
            continue
        
        try:
            ins, length = extra_ins_with_line_h1(cross, cross_tower_pts1, fit_line[i, :], 1, grid_width)
            
            ins_pts[i] = ins
        except:
            pass
    
    # ==================== First cross arm insulator extraction (vertical) ====================
    
    over_l = remove_duplicate_points(cross_line1, cross_line_pts1, 0.001)
    
    if tower_type == 3:
        over_lc = over_l[over_l[:, 2] < np.min(t[:, 2]) + grid_width * (loc[0, 0] + 1), :]
    else:
        over_lc = over_l.copy()
    
    if len(over_lc) == 0:
        over_lc = over_l.copy()

    if len(over_lc) >= 5:
        dbscan_model = DBSCAN(eps=2, min_samples=5)
        labels = dbscan_model.fit_predict(over_lc)
        max_lbl = max_label(labels)
        over_lc = over_lc[labels == max_lbl, :]
    
    if len(over_lc) > 0:
        cross_half_len = (np.max(over_lc[:, 1]) - np.min(over_lc[:, 1])) / 2
        
        for i in range(2):
            try:
                over1 = over_lc[
                    (over_lc[:, 1] > np.min(over_lc[:, 1]) + cross_half_len * i) &
                    (over_lc[:, 1] < np.min(over_lc[:, 1]) + cross_half_len * (i + 1)), :]
                
                if len(over1) == 0:
                    continue
                
                bin_yz, _ = bin_projection(over1, grid_width, 2, 3)
                zw = drow_zbarh(bin_yz, 1, 'wid')
                
                cut_indices = np.where(zw > 10)[0]
                if len(cut_indices) > 0:
                    cut_pos = cut_indices[-1]
                    
                    qlc1 = over1[over1[:, 2] > np.min(over1[:, 2]) + grid_width * (cut_pos + 1), :]
                    
                    ins_pts[2 + i] = qlc1
            except:
                pass
    
    # ==================== Second cross arm insulator extraction (horizontal) ====================
    
    if len(l) < 2:
        return ins_pts, ins_len
    
    cl2 = l[1]
    
    if len(cl2) == 0:
        return ins_pts, ins_len
    
    cross_beg = np.min(t[:, 2]) + grid_width * (loc[1, 1] + 1)
    cross_end = np.min(t[:, 2]) + grid_width * (loc[1, 0] + 1)
    
    cp2 = t[(t[:, 2] < cross_beg) & (t[:, 2] > cross_end), :]
    
    if len(cp2) == 0:
        return ins_pts, ins_len
    
    fleft, fright = fit_plane1(cp2, tower_type)
    fit_line = np.vstack([fleft, fright])
    
    mid_yt = np.min(cp2[:, 1]) + (np.max(cp2[:, 1]) - np.min(cp2[:, 1])) / 2
    
    half_len_y = np.abs(np.min(cl2[:, 1]) - mid_yt)
    
    over_in_one_cross = np.zeros((0, 3))
    
    for i in range(2):
        hl = cl2[
            (cl2[:, 1] >= np.min(cl2[:, 1]) + half_len_y * i) &
            (cl2[:, 1] <= np.min(cl2[:, 1]) + half_len_y * (i + 1)), :]
        
        if len(hl) < 5:
            continue
        
        dbscan_model = DBSCAN(eps=1, min_samples=5)
        labels = dbscan_model.fit_predict(hl)
        max_lbl = max_label(labels)
        hl_one = hl[labels == max_lbl, :]
        
        if len(hl_one) == 0:
            continue
        
        _, theta = rot_with_axle(hl_one, 3)
        rot_z = _rotz(np.rad2deg(theta))
        hlr3 = hl @ rot_z
        cp2r3 = cp2 @ rot_z
        
        mid_ytr = np.min(cp2r3[:, 1]) + (np.max(cp2r3[:, 1]) - np.min(cp2r3[:, 1])) / 2
        half_len_yr = np.abs(np.min(hlr3[:, 1]) - mid_ytr)
        
        for j in range(2):
            qlr3 = hlr3[
                (hlr3[:, 1] >= np.min(hlr3[:, 1]) + half_len_yr * j) &
                (hlr3[:, 1] <= np.min(hlr3[:, 1]) + half_len_yr * (j + 1)), :]
            
            rot_z_inv = _rotz(-np.rad2deg(theta))
            ql = qlr3 @ rot_z_inv
            
            if len(ql) == 0:
                continue
            
            qlc1, qlc2 = cut_over_line(ql, grid_width, fit_line[i, :], mid_yt)
            
            over_in_one_cross = np.vstack([over_in_one_cross, qlc2])
            
            try:
                ins, length = extra_ins_with_line_h1(qlc1, cp2, fit_line[i, :], 0, grid_width)
                
                ins_pts[4 + 2 * i + j] = ins
                ins_len[4 + 2 * i + j] = length
            except:
                pass
    
    # ==================== Second cross arm insulator extraction (longitudinal) ====================
    
    if len(over_in_one_cross) == 0:
        return ins_pts, ins_len
    
    mid_xcl = (np.max(cl2[:, 0]) - np.min(cl2[:, 0])) / 2
    
    for i in range(2):
        qlc3 = over_in_one_cross[
            (over_in_one_cross[:, 0] >= np.min(cl2[:, 0]) + mid_xcl * i) &
            (over_in_one_cross[:, 0] < np.min(cl2[:, 0]) + mid_xcl * (i + 1)), :3]
        
        if len(qlc3) < 5:
            continue
        
        dbscan_model = DBSCAN(eps=1, min_samples=5)
        labels = dbscan_model.fit_predict(qlc3)
        max_lbl = max_label(labels)
        qlc4 = qlc3[labels == max_lbl, :]
        
        if len(qlc4) == 0:
            continue
        
        bin_yz, _ = bin_projection(qlc4, grid_width, 2, 3)
        wid1 = drow_zbarh(bin_yz, -2, 'wid')
        
        cut_thre = 10
        
        try:
            beg_pos, end_pos = _find_muta(wid1, cut_thre)
        except:
            continue
        
        if len(beg_pos) > 0:
            wid2 = end_pos[-1] - beg_pos[0]
            
            try:
                start_idx = max(0, beg_pos[0] - wid2)
                end_idx = min(bin_yz.shape[1], end_pos[-1] + wid2 + 1)
                h_bin_yz = bin_yz[:, start_idx:end_idx]
            except:
                continue
            
            h_wid = drow_zbarh(h_bin_yz, 1, 'wid')
            cut_indices = np.where(h_wid > wid2)[0]
            
            if len(cut_indices) > 0:
                cut_pos_h2 = cut_indices[-1] + 1
            else:
                sum_indices = np.where(np.sum(h_bin_yz, axis=1) > 0)[0]
                if len(sum_indices) > 0:
                    cut_pos_h2 = sum_indices[0]
                else:
                    continue
            
            ins_in_o = qlc4[
                (qlc4[:, 2] > np.min(qlc4[:, 2]) + (cut_pos_h2 + 1) * grid_width) &
                (qlc4[:, 1] <= np.min(qlc4[:, 1]) + (end_pos[-1] + 1 + 1) * grid_width) &
                (qlc4[:, 1] >= np.min(qlc4[:, 1]) + (beg_pos[0] - 1 + 1) * grid_width), :]
            
            if len(ins_in_o) == 0:
                continue
            
            all_pts = np.vstack([t, cl2])
            ins = all_pts[
                (all_pts[:, 2] < np.min(t[:, 2]) + (loc[1, 0] - 1) * grid_width) &
                (all_pts[:, 2] > np.min(ins_in_o[:, 2])) &
                (all_pts[:, 1] < np.max(ins_in_o[:, 1])) & 
                (all_pts[:, 1] > np.min(ins_in_o[:, 1])) &
                (all_pts[:, 0] < np.max(ins_in_o[:, 0])) & 
                (all_pts[:, 0] > np.min(ins_in_o[:, 0])), :]
            
            if len(beg_pos) == len(end_pos) and len(beg_pos) == 2 and (beg_pos[1] - end_pos[0]) > 0:
                mid_pos = end_pos[0] + int(np.ceil((beg_pos[1] - end_pos[0]) / 2))
                
                ins1 = ins[
                    (ins[:, 1] >= np.min(qlc4[:, 1])) & 
                    (ins[:, 1] <= np.min(qlc4[:, 1]) + (mid_pos + 1) * grid_width), :]
                ins2 = ins[
                    (ins[:, 1] > np.min(qlc4[:, 1]) + (mid_pos + 1) * grid_width) & 
                    (ins[:, 1] <= np.max(qlc4[:, 1])), :]
                
                ins_pts[8 + 2 * i + 0] = ins1
                ins_pts[8 + 2 * i + 1] = ins2
            else:
                ins_pts[8 + 2 * i + 0] = ins
                ins_pts[8 + 2 * i + 1] = np.array([[0, 0, 0]])
    
    return ins_pts, ins_len


def _find_muta(bin_data, cut_thre):
    """
    Find mutation positions in histogram
    
    Parameters:
    -----------
    bin_data : numpy.ndarray
        Histogram bin data
    cut_thre : float
        Threshold for clipping
        
    Returns:
    --------
    beg_pos : numpy.ndarray
        Beginning positions of mutations
    end_pos : numpy.ndarray
        Ending positions of mutations
        
    Notes:
    ------
    This function identifies significant mutations (peaks) in the histogram.
    """
    bin_c = bin_data - cut_thre
    bin_c[bin_c < 0] = 0
    
    d_bin_c = np.zeros_like(bin_c)
    d_bin_c[1:] = bin_c[:-1]
    
    df = d_bin_c + bin_c
    
    beg_pos = np.where((bin_c == df) & (bin_c != 0))[0]
    end_pos = np.where((d_bin_c == df) & (d_bin_c != 0))[0]
    
    if len(beg_pos) > 1 and len(end_pos) > 1:
        gap1 = beg_pos[1:] - end_pos[:-1]
        keep_indices = np.where(gap1 > 2)[0]

        if len(keep_indices) > 0:
            beg_pos = np.concatenate([[beg_pos[0]], beg_pos[keep_indices + 1]])
            end_pos = end_pos[np.concatenate([keep_indices, [len(end_pos) - 1]])]
    
    beg_pos = beg_pos - 1
    end_pos = end_pos - 1
    
    del_gap = np.where((beg_pos < len(bin_c) / 4) | (end_pos > len(bin_c) / 4 * 3))[0]
    beg_pos = np.delete(beg_pos, del_gap)
    end_pos = np.delete(end_pos, del_gap)
    
    return beg_pos, end_pos


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
