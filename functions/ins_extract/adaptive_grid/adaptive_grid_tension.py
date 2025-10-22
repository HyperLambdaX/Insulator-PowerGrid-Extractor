"""Select an optimal sampling grid for each insulator on tension towers."""
import numpy as np
from .calcu_v import calcu_v


def adaptive_grid_tension(all_ins, all_len):
    """
    Adaptive grid selection for tension towers
    Selects optimal grid width for each insulator based on verticality
    
    Parameters:
    -----------
    all_ins : list or 2D array
        Cell array containing insulators extracted at different grid widths
        Shape: (InsNum, GridNum)
    all_len : numpy.ndarray
        Array of insulator lengths for each grid width
        Shape: (InsNum, GridNum)
        
    Returns:
    --------
    fine_ins : numpy.ndarray
        Final selected insulator points with labels, shape (N, 4)
    fine_len : numpy.ndarray
        Final selected insulator lengths
    fine_index : numpy.ndarray
        Final selected grid width indices
    fine_ve : numpy.ndarray
        Final verticality values
    """
    ins_num = len(all_ins) if isinstance(all_ins, list) else all_ins.shape[0]
    
    if isinstance(all_ins, list):
        grid_num = len(all_ins[0]) if ins_num > 0 else 0
    else:
        grid_num = all_ins.shape[1]
    
    # Calculate the verticality of each insulator
    ve_i = np.zeros((ins_num, grid_num))
    
    for i in range(ins_num):
        for j in range(grid_num):
            try:
                if isinstance(all_ins, list):
                    ins_pts = all_ins[i][j] if isinstance(all_ins[i], list) else all_ins[i]
                else:
                    ins_pts = all_ins[i, j]
                
                if ins_pts is not None and isinstance(ins_pts, np.ndarray) and ins_pts.size > 0:
                    ve_i[i, j] = calcu_v(ins_pts)
            except:
                pass
    
    xy_ins_ind = (np.sum((ve_i < 0.8) & (ve_i > 0) & (all_len > 1), axis=1) 
                  - np.sum(ve_i > 0, axis=1) / 2 >= 0)
    
    mask_xy = (ve_i.flatten() < 0.8) & (all_len.flatten() > 2.5)
    if np.any(mask_xy):
        mean_len_xy = np.mean(all_len.flatten()[mask_xy])
    else:
        mean_len_xy = 0
    xy_ind = np.argmin(np.abs(all_len - mean_len_xy), axis=1)
    
    mask_z = (ve_i.flatten() >= 0.8) & (all_len.flatten() < 2.5)
    if np.any(mask_z):
        mean_len_z = np.mean(all_len.flatten()[mask_z])
    else:
        mean_len_z = 0
    z_ind = np.argmin(np.abs(all_len - mean_len_z), axis=1)
    
    res_ind = xy_ind.copy()
    res_ind[~xy_ins_ind] = z_ind[~xy_ins_ind]
    
    fine_len = all_len[np.arange(all_len.shape[0]), res_ind]
    
    fine_index = res_ind

    fine_ve = ve_i[np.arange(ve_i.shape[0]), res_ind]
    
    fine_ins = np.zeros((0, 4))
    
    label = 1
    
    for i in range(ins_num):
        if isinstance(all_ins, list):
            ins_cur = all_ins[i][fine_index[i]] if isinstance(all_ins[i], list) else all_ins[i]
        else:
            ins_cur = all_ins[i, fine_index[i]]
        
        if ins_cur is not None and isinstance(ins_cur, np.ndarray) and ins_cur.shape[0] > 1:
            labels_col = np.full((ins_cur.shape[0], 1), label)
            ins_with_label = np.hstack([ins_cur, labels_col])
            fine_ins = np.vstack([fine_ins, ins_with_label])
            
            label += 1
    
    return fine_ins, fine_len, fine_index, fine_ve
