"""
Get the ID of each tower in the transmission line

"""
import os


def get_tower_id(filepath):
    """
    Get the number of each tower of the transmission line
    
    Parameters:
    -----------
    filepath : str
        Path to the directory containing tower data files
        
    Returns:
    --------
    tower_ids : list of str
        List of tower IDs extracted from filenames
        
    """
    try:
        read_dir = os.listdir(filepath)
    except FileNotFoundError:
        return []
    
    tower_ids = []

    for filename in read_dir:
        if 'Tower.txt' not in filename:
            continue

        # Find position of 'Tower' and extract prefix
        tower_pos = filename.find('Tower')
        if tower_pos != -1:
            tower_id = filename[:tower_pos]
            tower_ids.append(tower_id)

    # Sort tower IDs to ensure sequential processing
    tower_ids = sorted(tower_ids)

    return tower_ids

