"""Extract transverse and longitudinal insulators for Type-5 drum towers."""
import numpy as np
from sklearn.cluster import DBSCAN
from ...max_label import max_label
from ...project.bin_projection import bin_projection
from ...draw_results.drow_zbarh import drow_zbarh
from ..fit_plane.fit_plane1 import fit_plane1
from ..tc_split.cut_over_line import cut_over_line
from ..extract_ins_in_tc_or_jc.extra_ins_with_line_h1 import extra_ins_with_line_h1
from ...redirect.rot_with_axle import rot_with_axle


def ins_extract_type51(t, l, loc, grid_width, tower_type):
    """
    Insulator extraction for Type 5 tower (Tension type drum tower)
    
    Parameters:
    -----------
    t : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    l : list of numpy.ndarray
        List of power line point clouds - should have 3 elements
    loc : numpy.ndarray
        Cross arm locations of shape (M, 2) [start, end]
    grid_width : float
        Grid width for projection
    tower_type : int
        Tower type (should be 5)
        
    Returns:
    --------
    ins_pts : list
        List of extracted insulator point clouds (24 elements)
    ins_len : numpy.ndarray
        Array of insulator lengths (24 elements)
        
    Notes:
    ------
    This function handles transverse and longitudinal insulator extraction
    for tension type drum towers, processing 3 cross arms.
    Includes nested helper function findMuta.
    """
    # Initialize with empty arrays (not nested lists) to match assignment pattern
    ins_pts = [np.array([[0, 0, 0]]) for _ in range(24)]
    ins_len = np.zeros(24)
    
    # Take the last 3 cross arms
    loc = loc[-3:, :]
    
    # Loop over three cross arms
    for k in range(3):
        if len(l) <= k:
            continue
        
        cl = l[k]
        
        if len(cl) == 0:
            continue
        
        # ==================== Cut off the crossover line ====================
        cross_beg = np.min(t[:, 2]) + grid_width * (loc[k, 1] + 1)
        cross_end = np.min(t[:, 2]) + grid_width * (loc[k, 0] + 1)
        
        cp = t[(t[:, 2] < cross_end + 0.5) & (t[:, 2] > cross_end), :]
        
        if len(cp) == 0:
            continue
        
        fleft, fright = fit_plane1(cp, tower_type)

        fit_line = np.vstack([fleft, fright])
        
        # ==================== Transverse insulator extraction ====================
        mid_yt = np.min(cp[:, 1]) + (np.max(cp[:, 1]) - np.min(cp[:, 1])) / 2
        
        half_len_y = np.abs(np.min(cl[:, 1]) - mid_yt)
        
        over_in_one_cross = np.zeros((0, 3))
        
        l_counter = 1
        
        # Loop over two transmission directions
        for i in range(2):
            hl = cl[
                (cl[:, 1] >= np.min(cl[:, 1]) + half_len_y * i) &
                (cl[:, 1] <= np.min(cl[:, 1]) + half_len_y * (i + 1)), :]
            
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
            cp2r3 = cp @ rot_z
            
            mid_ytr = np.min(cp2r3[:, 1]) + (np.max(cp2r3[:, 1]) - np.min(cp2r3[:, 1])) / 2
            
            qlr_cell = [
                hlr3[hlr3[:, 1] < mid_ytr, :],
                hlr3[hlr3[:, 1] >= mid_ytr, :]
            ]
            
            # Loop over two halves
            for j in range(2):
                qlr = qlr_cell[j]
                
                if len(qlr) == 0:
                    continue
                
                rot_z_inv = _rotz(-np.rad2deg(theta))
                ql = qlr @ rot_z_inv
                
                if len(ql) >= 20:
                    dbscan_model = DBSCAN(eps=2, min_samples=20)
                    labels = dbscan_model.fit_predict(ql)
                    max_lbl = max_label(labels)
                    qlc = ql[labels == max_lbl, :]
                else:
                    qlc = ql.copy()
                
                if len(qlc) == 0:
                    continue
                
                qlc1, qlc2 = cut_over_line(qlc, grid_width, fit_line[i, :], mid_yt)
                
                # Add discrete points to QLC2
                if len(ql) >= 20:
                    qlc2 = np.vstack([qlc2, ql[labels != max_lbl, :]])
                
                ins, length = extra_ins_with_line_h1(qlc1, cp, fit_line[i, :], 0, grid_width)
                
                over_in_one_cross = np.vstack([over_in_one_cross, qlc2])
                
                ins_pts[k * 8 + i * 4 + j] = ins
                ins_len[k * 8 + i * 4 + j] = length
                
                l_counter += 1
        
          
        # ==================== Longitudinal insulator extraction ====================
        if len(over_in_one_cross) == 0:
            continue
        
        mid_xcl = (np.max(cl[:, 0]) - np.min(cl[:, 0])) / 2
        
        # Loop over two longitudinal directions
        for i in range(2):
            qlc3 = over_in_one_cross[
                (over_in_one_cross[:, 0] >= np.min(cl[:, 0]) + mid_xcl * i) &
                (over_in_one_cross[:, 0] < np.min(cl[:, 0]) + mid_xcl * (i + 1)), :3]
            
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
            
            # cutThre = 10
            cut_thre = 10
            
            try:
                beg_pos, end_pos = _find_muta(wid1, cut_thre)
            except:
                continue
            
            if len(beg_pos) > 0:
                wid2 = end_pos[-1] - beg_pos[0] + 1
                
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
                
                all_pts = np.vstack([t, cl])
                ins = all_pts[
                    (all_pts[:, 2] < np.min(t[:, 2]) + (loc[k, 0] - 1) * grid_width) &
                    (all_pts[:, 2] > np.min(ins_in_o[:, 2])) &
                    (all_pts[:, 1] < np.max(ins_in_o[:, 1])) & 
                    (all_pts[:, 1] > np.min(ins_in_o[:, 1])) &
                    (all_pts[:, 0] < np.max(ins_in_o[:, 0])) & 
                    (all_pts[:, 0] > np.min(ins_in_o[:, 0])), :]
                
                if ((len(beg_pos) == 2 and (beg_pos[1] - end_pos[0]) > 0) or
                    (len(beg_pos) == 1 and (end_pos[0] - beg_pos[0]) * grid_width > 1)):
                    mid_pos = end_pos[0] + int(np.ceil((beg_pos[-1] - end_pos[0]) / 2))
                    
                    ins1 = ins[
                        (ins[:, 1] >= np.min(qlc4[:, 1])) & 
                        (ins[:, 1] <= np.min(qlc4[:, 1]) + (mid_pos + 1) * grid_width), :]
                    ins2 = ins[
                        (ins[:, 1] > np.min(qlc4[:, 1]) + (mid_pos + 1) * grid_width) & 
                        (ins[:, 1] <= np.max(qlc4[:, 1])), :]
                    
                    ins_pts[k * 8 + 2 + 4 * i + 0] = ins1
                    ins_pts[k * 8 + 2 + 4 * i + 1] = ins2
                else:
                    ins_pts[k * 8 + 2 + 4 * i + 0] = ins
    
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
    Very similar to the findMuta function in InsExtractType4, but with slightly different filtering.
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
    
    gap2 = end_pos - beg_pos
    del_gap = np.where((gap2 <= 3) | (beg_pos < len(bin_c) / 4) | (end_pos > len(bin_c) / 4 * 3))[0]
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
