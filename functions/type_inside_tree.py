"""Dispatch insulator extraction routines based on the detected tower type."""
import numpy as np
from .type_detect.type_detect import type_detect
from .project.bin_projection import bin_projection
from .type_detect.cross_location import cross_location
from .ins_extract.append_loc import append_loc
from .ins_extract.cluster_line.split_power_line_2d import split_power_line_2d
from .ins_extract.cluster_line.split_power_line_c4 import split_power_line_c4
from .ins_extract.cluster_line.split_power_line_c41 import split_power_line_c41
from .ins_extract.cluster_line.split_power_line_c5 import split_power_line_c5
from .ins_extract.type_flow.ins_extract_zl import ins_extract_zl
from .ins_extract.type_flow.ins_extract_zl1 import ins_extract_zl1
from .ins_extract.type_flow.ins_extract_type4 import ins_extract_type4
from .ins_extract.type_flow.ins_extract_type51 import ins_extract_type51


def type_inside_tree(t, l, grid_width):
    """
    Main function for insulator extraction based on tower type
    
    Parameters:
    -----------
    t : numpy.ndarray
        Tower point cloud of shape (N, 3) [X, Y, Z]
    l : numpy.ndarray
        Power line point cloud of shape (M, 3) [X, Y, Z]
    grid_width : float
        Grid width for projection and processing
        
    Returns:
    --------
    ins_pts : numpy.ndarray or list
        Extracted insulator points
    is_cable : int
        Flag indicating if cable-stayed type (0 or 1)
    ins_len : numpy.ndarray or list
        Lengths of extracted insulators
        
    Notes:
    ------
    This is the core algorithm function that dispatches to different
    extraction methods based on tower type.
    
    Tower types:
    1, 8: Wine glass tower, portal tower
    2: Cat head tower
    3: Single cross arm tower
    4: Tension resistant dry type tower
    5: Tension type drum tower
    6: DC drum tower
    """
    tower_type = type_detect(t, l)
    
    bin_yz, _ = bin_projection(t, grid_width, 1, 3)
    
    loc = cross_location(bin_yz, 3)
    
    ins_pts = np.zeros((0, 3))
    is_cable = 0
    ins_len = []
    
    if tower_type in [1, 8]:
        cross_line = split_power_line_2d(l, 0.1, 1, 10)
        
        print(f"Processing type {tower_type}: wine-glass or portal tower")
        ins_pts, ins_len = ins_extract_zl(t, cross_line, loc, grid_width, tower_type)
        
    elif tower_type == 2:
        cross_line = split_power_line_c4(l, 2, 0.6)
        
        print(f"Processing type {tower_type}: cat-head tower")
        ins_pts, is_cable, ins_len = ins_extract_zl1(t, cross_line, loc, grid_width)
        
    elif tower_type == 3:
        cross_line = split_power_line_c4(l, 2, 0.5)
        
        loc_p = append_loc(bin_yz, loc)
        
        print(f"Processing type {tower_type}: single cross-arm tower")
        ins_pts, ins_len = ins_extract_type4(t, cross_line, loc_p, grid_width, tower_type)
        
    elif tower_type == 4:
        lc = split_power_line_c5(l, 4, 0.5)
        
        cross_line = split_power_line_c41(lc, 2, 1)
        
        print(f"Processing type {tower_type}: tension cross-arm tower")
        ins_pts, ins_len = ins_extract_type4(t, cross_line, loc, grid_width, tower_type)
        
    elif tower_type == 5:
        cross_line = split_power_line_c4(l, 3, 0.8)
        
        print(f"Processing type {tower_type}: tension drum tower")
        ins_pts, ins_len = ins_extract_type51(t, cross_line, loc, grid_width, tower_type)
        pass      
    
    elif tower_type == 6:
        cross_line = split_power_line_2d(l, 0.1, 3, 10)
        
        print(f"Processing type {tower_type}: DC drum tower")
        ins_pts, ins_len = ins_extract_zl(t, cross_line, loc, grid_width, tower_type)
    
    else:
        print(f"Warning: unknown tower type {tower_type}")
        ins_pts = np.zeros((0, 3))
        ins_len = []
    
    return ins_pts, is_cable, ins_len
