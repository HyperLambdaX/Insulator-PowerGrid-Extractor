"""Measure the vertical alignment of a point cloud via PCA."""
import numpy as np


def calcu_v(points):
    """
    Calculate verticality of point cloud using PCA
    
    Parameters:
    -----------
    points : numpy.ndarray
        Point cloud array of shape (N, 3)
        
    Returns:
    --------
    ve : float
        Verticality measure (0 to 1), absolute value of Z component
        of the main eigenvector
    """
    if points.size == 0 or points.shape[0] == 0:
        return 0
    
    # Centralize points
    mean_points = np.mean(points, axis=0)
    
    points_centered = points - mean_points
    
    # Calculate covariance matrix
    # Note: numpy.cov expects features in rows, so transpose
    cov_matrix = np.cov(points_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    D, V = np.linalg.eig(cov_matrix)
    
    # Sort according to eigenvalues (descending)
    D_order = np.argsort(D)[::-1]
    
    V_sort = V[:, D_order]
    
    # Find the main eigenvector (direction vector)
    dir_vector = V_sort[:, 0]
    
    # Normalize direction vector
    length = np.linalg.norm(dir_vector)
    
    norm_vector = dir_vector / length if length > 0 else dir_vector
    
    # Calculate verticality (Z component)
    ve = np.abs(norm_vector[2])
    
    return ve
