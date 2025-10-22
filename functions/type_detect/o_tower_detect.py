"""Detect whether a tower silhouette contains the hole of an O-type tower."""
import numpy as np
from skimage import measure


def o_tower_detect(img, cut_loca, auto_threshold=None):
    """
    Detect binary image holes and calculate their area size.
    If the area is greater than the threshold, it is an O-shaped tower.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Binary image after tower reorientation and binarization
    cut_loca : float
        Partial power tower interception location (fraction, e.g., 1/2)
    auto_threshold : float, optional
        Hole area threshold. If not provided, calculated automatically
        
    Returns:
    --------
    is_o_tower : bool
        Returns True if it is an O-type tower, otherwise False
        
    """
    # Only judge part of the above
    cut_row = int(np.ceil(img.shape[0] * cut_loca))
    img = img[cut_row:, :]
    
    if img.size == 0:
        return False
    
    # Set first row to 1 to close boundaries
    img[0, :] = 1
    
    # Label the 1-regions (foreground)
    labeled_img = measure.label(img, connectivity=2, background=0)

    regions = measure.regionprops(labeled_img)

    # To find holes, we need to:
    # 1. Invert the image to turn holes (0s) into regions (1s)
    # 2. Only count regions that are NOT touching the image boundary
    #    (regions touching boundary are external background, not internal holes)

    img_inv = 1 - img
    labeled_holes = measure.label(img_inv, connectivity=2, background=0)
    hole_regions = measure.regionprops(labeled_holes)

    # Filter out regions that touch the image boundary
    # (these are external background, not internal holes)
    max_area = 0
    for region in hole_regions:
        # Check if region touches any boundary
        minr, minc, maxr, maxc = region.bbox
        touches_boundary = (
            minr == 0 or minc == 0 or
            maxr == img.shape[0] or maxc == img.shape[1]
        )

        if not touches_boundary:
            # This is a true internal hole
            if region.area > max_area:
                max_area = region.area
    
    if auto_threshold is None:
        auto_threshold = img.shape[0] * img.shape[1] * 0.07
    
    if max_area > auto_threshold and max_area > 300:
        is_o_tower = True
    else:
        is_o_tower = False
    
    return is_o_tower
