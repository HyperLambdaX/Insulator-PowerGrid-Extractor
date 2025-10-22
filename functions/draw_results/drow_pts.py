"""3D point-cloud visualization helpers."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def drow_pts(*args):
    """
    Plot 3D point clouds with various input formats
    
    Parameters:
    -----------
    *args : variable arguments
        Can be:
        1. Single array with 4 columns [X,Y,Z,Label], optional class_num
        2. Multiple arrays with 3 columns [X,Y,Z]
        3. Pairs of (array, style_string)
        
    Notes:
    ------
    This is mainly used for visualization during development
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if len(args) == 0:
        return
    
    # Get first argument
    pts1 = np.array(args[0])
    
    # Case 1: Point cloud with labels (4 columns)
    if pts1.shape[1] == 4:
        if len(args) > 1 and isinstance(args[-1], (int, float)):
            class_num = int(args[-1])
        else:
            class_num = 0
        
        if class_num == 0:
            class_num = len(np.unique(pts1[:, 3]))
        
        color_map = plt.cm.hsv(np.linspace(0, 1, class_num))
        
        for i in range(class_num):
            color = color_map[i]
            cluster_uni = np.unique(pts1[:, 3])
            if i < len(cluster_uni):
                cur_cluster = cluster_uni[i]
                mask = pts1[:, 3] == cur_cluster
                ax.scatter(pts1[mask, 0], pts1[mask, 1], pts1[mask, 2], 
                          c=[color], marker='.', s=1)
    
    # Case 2: Multiple point clouds, all numeric
    elif len(args) > 1 and all(isinstance(arg, np.ndarray) and arg.dtype.kind in 'biufc' 
                                for arg in args):
        color_map = plt.cm.hsv(np.linspace(0, 1, len(args)))
        for i, pts in enumerate(args):
            if pts.size > 0 and pts.shape[1] >= 3:
                color = color_map[i]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                          c=[color], marker='.', s=1)
    
    else:
        num_pairs = len(args) // 2
        for i in range(num_pairs):
            k = i * 2
            pts = np.array(args[k])
            style = args[k + 1] if k + 1 < len(args) else '.r'
            
            if pts.size > 0 and pts.shape[1] >= 3:
                color = 'red' if 'r' in style else 'blue' if 'b' in style else 'green' if 'g' in style else 'black'
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                          c=color, marker='.', s=1)
    
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
