"""Rotate point clouds so their primary direction aligns with the X axis."""
import numpy as np
from scipy.spatial.transform import Rotation


def rot_with_axle(pts, rota_axle):
    """
    Reorient a 3D point cloud by projecting it onto the plane orthogonal to
    ``rota_axle`` and rotating it so the dominant direction is parallel to the
    X axis.
    
    According to field calibration, the power-line direction should be roughly
    parallel to the X axis after reorientation.
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Point cloud array of shape (N, 3) with columns [X, Y, Z]
    rota_axle : int
        Rotation axis (1, 2, or 3 for X, Y, Z respectively)
        
    Returns:
    --------
    pts_r : numpy.ndarray
        Rotated point cloud
    angle : float
        Rotation angle in radians
        
    Notes:
    ------
    Uses PCA to find main direction and aligns it with X-axis
    """
    axle = [0, 1, 2]
    axle.remove(rota_axle - 1)
    
    pts_2d = pts[:, axle]
    
    pts_2d_down = grid_downsample_2d(pts_2d, grid_size=0.2)
    
    center = np.mean(pts_2d_down, axis=0)
    
    M = pts_2d_down - center
    
    MM = (M.T @ M) / pts_2d_down.shape[0]
    
    eigenvalues, V = np.linalg.eig(MM)
    
    # Direction vector of point cloud (second eigenvector - corresponding to larger eigenvalue)
    # Sort by eigenvalue to ensure consistency
    idx = np.argsort(eigenvalues)[::-1]  # Sort descending
    V = V[:, idx]
    dire_vector = V[:, 0]  # First eigenvector (largest eigenvalue)
    
    # The angle between the direction vector and the projected x-axis
    angle = np.arccos(np.abs(dire_vector[0]) / np.linalg.norm(dire_vector))
    
    if rota_axle == 1:  # Rotate around X-axis
        if dire_vector[0] * dire_vector[1] < 0:
            angle = -angle
        pts_r = apply_rotation(pts, angle, 'x')
        
    elif rota_axle == 2:  # Rotate around Y-axis
        if dire_vector[0] * dire_vector[1] > 0:
            angle = -angle
        pts_r = apply_rotation(pts, angle, 'y')
        
    else:  # rota_axle == 3, Rotate around Z-axis
        if dire_vector[0] * dire_vector[1] < 0:
            angle = -angle
        pts_r = apply_rotation(pts, angle, 'z')
    
    return pts_r, angle


def grid_downsample_2d(pts_2d, grid_size=0.2):
    """
    Grid-average downsampling for 2D points.

    """
    pts = np.asarray(pts_2d)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts_2d must be an (N, 2) array")

    if pts.shape[0] == 0:
        return pts.copy()

    step = float(grid_size)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid_size must be a positive finite scalar")

    pts_f = pts.astype(np.float64, copy=False)
    min_bound = np.min(pts_f, axis=0)

    # to keep boundary points stable under floating point noise.
    scaled = (pts_f - min_bound) / step
    indices = np.floor(scaled + 1e-12).astype(np.int64)

    # Group points by voxel index and average within each voxel.
    unique_indices, inverse = np.unique(indices, axis=0, return_inverse=True)
    sums = np.zeros((unique_indices.shape[0], 2), dtype=np.float64)
    np.add.at(sums, inverse, pts_f)
    counts = np.bincount(inverse, minlength=sums.shape[0]).astype(np.float64)

    downsampled = sums / counts[:, None]
    return downsampled


def apply_rotation(pts, angle, axis):
    """
    Apply rotation to 3D points around specified axis
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Point cloud of shape (N, 3)
    angle : float
        Rotation angle in radians
    axis : str
        Rotation axis ('x', 'y', or 'z')
        
    Returns:
    --------
    pts_rotated : numpy.ndarray
        Rotated point cloud
    """
    # Convert angle from radians to degrees for scipy
    angle_deg = angle * 180 / np.pi
    
    # Create rotation object
    if axis == 'x':
        r = Rotation.from_euler('x', angle_deg, degrees=True)
    elif axis == 'y':
        r = Rotation.from_euler('y', angle_deg, degrees=True)
    else:  # axis == 'z'
        r = Rotation.from_euler('z', angle_deg, degrees=True)
    
    rotation_matrix = r.as_matrix()
    pts_rotated = pts @ rotation_matrix

    return pts_rotated
