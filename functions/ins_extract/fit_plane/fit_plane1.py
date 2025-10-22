"""Fit left/right boundary lines for upper cross arms with helper routines."""
import os
import numpy as np
from ...redirect.rot_with_axle import rot_with_axle
from ...project.sin_pro import sin_pro
from .ransac_fitline import ransac_fitline
from .boundary_2d import boundary_2d


def fit_plane1(points, tower_type):
    """
    Fit left and right boundary lines for cross arm
    
    Parameters:
    -----------
    points : numpy.ndarray
        Cross arm tower points of shape (N, 3) [X, Y, Z]
    tower_type : int
        Tower type (3, 5 requires cutting middle, 1, 4 does not)
        
    Returns:
    --------
    fleft : numpy.ndarray
        Left boundary line parameters [slope, intercept]
    fright : numpy.ndarray
        Right boundary line parameters [slope, intercept]
        
    Notes:
    ------
    This function is more complex than fitPlane2, handling different tower types
    and including multiple nested helper functions.
    """
    if tower_type in [3, 5]:
        is_cut_mid = True
    elif tower_type in [1, 4]:
        is_cut_mid = False
    else:
        raise ValueError("Undefined tower type")
    
    points_rz, theta3 = rot_with_axle(points, 3)
    points_rzy, theta2 = rot_with_axle(points_rz, 2)
    
    # Swap X and Y columns
    points_rzy = points_rzy[:, [1, 0, 2]]
    
    # boundary(X, Y) returns boundary indices based on alpha shape algorithm
    try:
        k = boundary_2d(points_rzy[:, [0, 1]], shrink_factor=0.5)
        points_b = points_rzy[k, :]
    except Exception as e:
        # Fallback if boundary extraction fails
        print(f"Warning: boundary_2d failed ({e}), using all points")
        points_b = points_rzy.copy()
    
    pts = np.zeros_like(points_b)
    pts[:, 0] = points_b[:, 0] - np.min(points_b[:, 0])
    pts[:, 1] = points_b[:, 1] - np.min(points_b[:, 1])
    pts[:, 2] = points_b[:, 2] - np.min(points_b[:, 2])

    sam_gap = 0.1
    mid_x = np.min(pts[:, 0]) + (np.max(pts[:, 0]) - np.min(pts[:, 0])) / 2
    
    pts_l_ind = np.where(pts[:, 0] < mid_x)[0]
    pts_left_r = _extra_fit_pts(pts_l_ind, pts, points_b, sam_gap, is_cut_mid)
    
    pts_left = pts_left_r[:, [1, 0, 2]] @ _roty(-theta2 * 180 / np.pi) @ _rotz(-theta3 * 180 / np.pi)
    
    fleft = ransac_fitline(pts_left[:, [0, 1]], 10000, sam_gap)
    
    pts_r_ind = np.where(pts[:, 0] >= mid_x)[0]
    pts_right_r = _extra_fit_pts(pts_r_ind, pts, points_b, sam_gap, is_cut_mid)
    
    pts_right = pts_right_r[:, [1, 0, 2]] @ _roty(-theta2 * 180 / np.pi) @ _rotz(-theta3 * 180 / np.pi)
    
    fright = ransac_fitline(pts_right[:, [0, 1]], 10000, sam_gap)
    
    if np.mean(pts_left[:, 1]) > np.mean(pts_right[:, 1]):
        temp_f = fleft.copy()
        fleft = fright.copy()
        fright = temp_f
    
    return fleft, fright


def _extra_fit_pts(half_ind, pts, points, sam_gap, is_cut_mid):
    """
    Extract the point function to be fitted
    
    Parameters:
    -----------
    half_ind : numpy.ndarray
        Indices of half of the points
    pts : numpy.ndarray
        Normalized boundary points
    points : numpy.ndarray
        Original boundary points
    sam_gap : float
        Sampling gap for histogram
    is_cut_mid : bool
        Whether to cut the middle part
        
    Returns:
    --------
    half_pts : numpy.ndarray
        Extracted fitting points
    """
    pts_half = pts[half_ind, :]
    
    pts_r, _ = _rota_pts(pts_half)
    
    pts_r_move = np.zeros_like(pts_r)
    pts_r_move[:, 0] = pts_r[:, 0] - np.min(pts_r[:, 0])
    pts_r_move[:, 1] = pts_r[:, 1] - np.min(pts_r[:, 1])
    pts_r_move[:, 2] = pts_r[:, 2] - np.min(pts_r[:, 2])
    
    out_pts_ind = np.arange(len(pts_r_move))  # Default: all points
    
    if is_cut_mid:
        y_len = np.max(pts_r_move[:, 1]) - np.min(pts_r_move[:, 1])
        # Keep only points in outer quarters (remove middle half)
        out_pts_ind = np.where((pts_r_move[:, 1] <= np.min(pts_r_move[:, 1]) + y_len / 4) |
                               (pts_r_move[:, 1] >= np.max(pts_r_move[:, 1]) - y_len / 4))[0]
        pts_r_move = pts_r_move[out_pts_ind, :]
    
    if len(pts_r_move) == 0:
        # Fallback if all points removed
        return points[half_ind[:1], :] if len(half_ind) > 0 else np.zeros((1, 3))
    
    wid_histo, pts_in_w_ind = sin_pro(pts_r_move, 1, sam_gap)
    max_ind = np.argmax(wid_histo)
    
    s_range = 1
    
    tep_ind = np.arange(max(0, max_ind - s_range), min(len(wid_histo), max_ind + s_range + 1))
    len_dgde_ind = []
    for i in range(len(tep_ind)):
        len_dgde_ind.extend(np.where(pts_in_w_ind == tep_ind[i])[0].tolist())
    
    len_dgde_ind = np.array(len_dgde_ind)
    
    if is_cut_mid:
        dgde_ind = half_ind[out_pts_ind[len_dgde_ind]]
    else:
        dgde_ind = half_ind[len_dgde_ind]
    
    half_pts = points[dgde_ind, :]
    
    y_mid = np.min(points[:, 1]) + (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
    up_pts_len = np.sum(half_pts[:, 1] > y_mid)
    dw_pts_len = np.sum(half_pts[:, 1] < y_mid)
    is_ok = True
    if up_pts_len == 0 or dw_pts_len == 0:
        is_ok = False
    
    while not is_ok and s_range < len(wid_histo) / 3:
        s_range = s_range + 1
        
        # Expand to the left
        tep_ind_l = np.arange(max(0, max_ind - s_range), max_ind + 1)
        len_dgde_ind_l = []
        for i in range(len(tep_ind_l)):
            len_dgde_ind_l.extend(np.where(pts_in_w_ind == tep_ind_l[i])[0].tolist())
        
        len_dgde_ind_l = np.array(len_dgde_ind_l)
        
        if is_cut_mid:
            dgde_ind_l = half_ind[out_pts_ind[len_dgde_ind_l]]
        else:
            dgde_ind_l = half_ind[len_dgde_ind_l]
        
        half_pts_l = points[dgde_ind_l, :]
        up_pts_len_l = np.sum(half_pts_l[:, 1] > y_mid)
        dw_pts_len_l = np.sum(half_pts_l[:, 1] < y_mid)
        
        if up_pts_len_l > 1 and dw_pts_len_l > 1:
            is_ok = True
        
        # Expand to the right
        tep_ind_r = np.arange(max_ind, min(len(wid_histo), max_ind + s_range + 1))
        len_dgde_ind_r = []
        for i in range(len(tep_ind_r)):
            len_dgde_ind_r.extend(np.where(pts_in_w_ind == tep_ind_r[i])[0].tolist())
        
        len_dgde_ind_r = np.array(len_dgde_ind_r)
        
        if is_cut_mid:
            dgde_ind_r = half_ind[out_pts_ind[len_dgde_ind_r]]
        else:
            dgde_ind_r = half_ind[len_dgde_ind_r]
        
        half_pts_r = points[dgde_ind_r, :]
        up_pts_len_r = np.sum(half_pts_r[:, 1] > y_mid)
        dw_pts_len_r = np.sum(half_pts_r[:, 1] < y_mid)
        
        if up_pts_len_r > 1 and dw_pts_len_r > 1:
            if is_ok:  # Both left and right can expand, choose the side with more points
                if len(half_pts_r) >= len(half_pts_l):
                    half_pts = half_pts_r
                else:
                    half_pts = half_pts_l
            else:  # Can only expand to the right
                is_ok = True
                half_pts = half_pts_r
        elif is_ok:  # Can only expand to the left
            half_pts = half_pts_l
    
    return half_pts


def _rota_pts(pts):
    """
    Redirect point cloud function
    Rotate around the z-axis so that the point cloud is parallel to the y-axis
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Points to rotate of shape (N, 3)
        
    Returns:
    --------
    pts_r : numpy.ndarray
        Rotated points
    angle : float
        Rotation angle in radians
    """
    pts_2d = pts[:, [0, 1]]
    
    # Downsample using simple grid averaging (replacement for Open3D)
    pts_2d_down = _grid_downsample(pts_2d, voxel_size=0.2)
    
    center = np.mean(pts_2d_down, axis=0)
    m = pts_2d_down - center
    mm = (m.T @ m) / len(pts_2d_down)
    eigenvalues, eigenvectors = np.linalg.eig(mm)
    dire_vector = eigenvectors[:, 0]
    
    angle = np.arccos(np.abs(dire_vector[0]) / np.linalg.norm(dire_vector))
    if dire_vector[0] * dire_vector[1] < 0:
        angle = -angle
    
    pts_r = pts @ _rotz(angle * 180 / np.pi)
    
    return pts_r, angle


def _rotz(angle_deg):
    """
    Rotation matrix around Z axis
    
    Parameters:
    -----------
    angle_deg : float
        Rotation angle in degrees
        
    Returns:
    --------
    R : numpy.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def _roty(angle_deg):
    """
    Rotation matrix around Y axis
    
    Parameters:
    -----------
    angle_deg : float
        Rotation angle in degrees
        
    Returns:
    --------
    R : numpy.ndarray
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def _grid_downsample(points, voxel_size):
    """
    Simple grid-based downsampling (replacement for Open3D's voxel_down_sample)

    Parameters:
    -----------
    points : numpy.ndarray
        Points to downsample of shape (N, 2)
    voxel_size : float
        Size of the voxel grid

    Returns:
    --------
    downsampled : numpy.ndarray
        Downsampled points
    """
    if len(points) == 0:
        return points

    # Calculate grid indices for each point
    min_bound = np.min(points, axis=0)
    grid_indices = np.floor((points - min_bound) / voxel_size).astype(int)

    # Create unique voxel keys
    unique_voxels = {}
    for i, idx in enumerate(grid_indices):
        key = tuple(idx)
        if key not in unique_voxels:
            unique_voxels[key] = []
        unique_voxels[key].append(points[i])

    # Average points in each voxel
    downsampled = []
    for voxel_points in unique_voxels.values():
        downsampled.append(np.mean(voxel_points, axis=0))

    return np.array(downsampled)


def _visualize_boundary_points(pts, points_rzy=None, points_b=None, title="Boundary Points"):
    """
    Visualize boundary points for debugging

    Parameters:
    -----------
    pts : numpy.ndarray
        Normalized boundary points (N, 3)
    points_rzy : numpy.ndarray, optional
        Original rotated points before boundary extraction
    points_b : numpy.ndarray, optional
        Boundary points before normalization
    title : str
        Plot title

    Notes:
    ------
    This function is for debugging intermediate results in fit_plane1.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 6))

        # Plot 1: Original rotated points + boundary points
        if points_rzy is not None and points_b is not None:
            ax1 = fig.add_subplot(121, projection='3d')

            # Plot all rotated points in light blue
            ax1.scatter(points_rzy[:, 0], points_rzy[:, 1], points_rzy[:, 2],
                       c='lightblue', marker='.', s=1, alpha=0.3, label='All Points')

            # Plot boundary points in red
            ax1.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2],
                       c='red', marker='.', s=3, alpha=0.8, label=f'Boundary ({len(points_b)} pts)')

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            ax1.set_title('Original Points + Boundary')
            ax1.set_box_aspect([1, 1, 1])

        ax2 = fig.add_subplot(122, projection='3d')

        # Plot normalized boundary points in red
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='red', marker='.', s=3, label=f'Normalized Boundary ({len(pts)} pts)')

        ax2.set_xlabel('X (normalized)')
        ax2.set_ylabel('Y (normalized)')
        ax2.set_zlabel('Z (normalized)')
        ax2.legend()
        ax2.set_title(title)
        ax2.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Warning: matplotlib not available for visualization")
    except Exception as e:
        print(f"Warning: visualization failed: {e}")
