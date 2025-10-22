"""
Main program - Entry point for insulator extraction algorithm

Usage:
  python main.py                              # Process all towers
  python main.py --tower-id 001               # Process only tower 001
  python main.py --tower-id 001 --visualize   # Process tower 001 and visualize
  python main.py --tower-id 001 --plot        # Process tower 001 and save as PCD file
  python main.py --tower-id 001 --plot --visualize  # Save PCD and use Open3D visualization
"""
import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functions.get_tower_id import get_tower_id
from functions.redirect.r_tower import r_tower
from functions.redirect.rot_with_axle import apply_rotation
from functions.type_inside_tree import type_inside_tree
from functions.type_detect.type_detect import type_detect
from functions.merge_cell3 import merge_cell3
from functions.ins_extract.adaptive_grid.adaptive_grid_tension import adaptive_grid_tension
from functions.draw_results.drow_pts import drow_pts


def parse_arguments():
    """
    Parse command line arguments

    Returns:
    --------
    args : argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Insulator Extraction Algorithm - Extract insulators from UAV LiDAR point cloud',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                      # Process all towers in Data directory
  python main.py --tower-id 001                       # Process only tower 001
  python main.py --tower-id 001 --visualize           # Process tower 001 and visualize
  python main.py --tower-id 001 --plot                # Save as PCD file (insulators highlighted in red)
  python main.py --tower-id 001 --plot --visualize    # Save PCD and visualize with Open3D
  python main.py --tower-id 001 --plot --output-dir ./results  # Specify output directory
        """
    )

    parser.add_argument(
        '--tower-id',
        type=str,
        default=None,
        help='Specify tower ID to process (e.g., 001, 002). If not specified, processes all towers'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Data directory path'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization'
    )

    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Save point cloud as PCD file with visualization (insulators highlighted in red)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for PCD files. Default: ./output'
    )

    return parser.parse_args()


def main():
    """
    Main function for insulator extraction

    Process:
    1. Read tower and line point clouds from Data directory
    2. Redirect (align) tower point cloud
    3. Process with multiple grid widths
    4. Use adaptive grid selection for optimal results
    5. Visualize and save results
    """
    args = parse_arguments()

    print("=" * 60)
    print("Insulator Extraction Algorithm")
    print("Insulator Extraction from UAV LiDAR Point Cloud")
    print("=" * 60)

    # Data path
    data_path = args.data_path

    if not os.path.exists(data_path):
        print(f"Error: Data directory does not exist: {data_path}")
        return

    if args.tower_id:
        # Process only specified tower
        tower_ids = [args.tower_id]
        print(f"\nProcessing mode: Single tower")
        print(f"Specified tower ID: {args.tower_id}")
    else:
        # Get tower IDs - process all towers
        tower_ids = get_tower_id(data_path)
        print(f"\nProcessing mode: Batch")
        print(f"Found {len(tower_ids)} tower data files")

    tower_num = len(tower_ids)

    print(f"Data path: {data_path}")
    if args.visualize:
        print(f"Visualization: Enabled")
    elif args.no_visualize:
        print(f"Visualization: Disabled")
    else:
        print(f"Visualization: Auto (enabled for single tower)")

    if args.plot:
        print(f"PCD output: Enabled")
        print(f"Output directory: {args.output_dir}")

    print("-" * 60)

    # Record processing results for all towers
    tower_results = []

    # Traverse the towers
    for i in range(tower_num):
        tower_id = tower_ids[i]
        print(f"\nProcessing tower [{i+1}/{tower_num}]: {tower_id}")

        # Initialize variables
        detected_tower_type = -1
        tower_type_name = "Unknown"
        fine_ins = np.zeros((0, 3))
        num_insulators = 0  # Number of insulator segments

        try:
            tower_file = os.path.join(data_path, f"{tower_id}Tower.txt")
            line_file = os.path.join(data_path, f"{tower_id}Line.txt")

            if not os.path.exists(tower_file) or not os.path.exists(line_file):
                print(f"  Warning: Files not found, skipping")
                continue

            # Try comma delimiter first (common format), then space
            try:
                tower_pts = np.loadtxt(tower_file, delimiter=',')
                line_pts = np.loadtxt(line_file, delimiter=',')
            except:
                tower_pts = np.loadtxt(tower_file)
                line_pts = np.loadtxt(line_file)

            print(f"  Tower points: {tower_pts.shape[0]}, Line points: {line_pts.shape[0]}")

            tower_pts_r, theta = r_tower(tower_pts)
            line_pts_r = apply_rotation(line_pts, theta, 'z')

            print(f"  Redirection angle: {theta * 180 / np.pi:.2f}°")

            # Detect tower type
            detected_tower_type = type_detect(tower_pts_r, line_pts_r)
            tower_type_names = {
                1: "Wine Glass Tower",
                2: "Cat Head Tower",
                3: "Single Cross-arm Tower",
                4: "Tension Cross Tower",
                5: "Tension Drum Tower",
                6: "DC Drum Tower",
                8: "Portal Tower"
            }
            tower_type_name = tower_type_names.get(detected_tower_type, f"Unknown Type {detected_tower_type}")
            print(f"  Detected tower type: {detected_tower_type} - {tower_type_name}")

            grid_widths = np.arange(0.05, 0.16, 0.01)
            grid_num = len(grid_widths)

            print(f"  Using {grid_num} grid widths for multi-scale processing")

            ins_pts_in_grids = []
            len_pts_in_grids = []

            for j, grid_width in enumerate(grid_widths):
                print(f"    Processing grid width [{j+1}/{grid_num}]: {grid_width:.3f}")

                ins_cell, is_cable, ins_len_raw = type_inside_tree(tower_pts_r, line_pts_r, grid_width)

                if isinstance(ins_cell, list):
                    # Already in list format (from ins_extract functions)
                    # Wrap as 2D cell for mergeCell3: [[cell_item1], [cell_item2], ...]
                    cell_wrapped = [ins_cell]  # Single row of cells
                elif isinstance(ins_cell, np.ndarray) and ins_cell.size > 0:
                    # Single array, wrap in list structure
                    cell_wrapped = [[ins_cell]]
                else:
                    # Empty result
                    cell_wrapped = [[np.zeros((0, 3))]]

                # Call mergeCell3 to get merged points and lengths
                merged_pts, merged_lens = merge_cell3(cell_wrapped)

                ins_pts_in_grids.append(cell_wrapped[0])  # Append the wrapped cell
                len_pts_in_grids.append(merged_lens)

            # Convert to proper format for adaptive grid processing
            # Need to ensure each grid has the same number of cells

            # First, determine the maximum number of cells across all grids
            max_cells = 0
            cell_counts = []
            for cell_list in ins_pts_in_grids:
                if isinstance(cell_list, list):
                    count = len(cell_list)
                else:
                    count = 1
                cell_counts.append(count)
                max_cells = max(max_cells, count)

            # Pad each grid to have the same number of cells
            ins_array_padded = []
            len_array_padded = []

            for i in range(grid_num):
                # Get the cell list for this grid
                cell_list = ins_pts_in_grids[i] if i < len(ins_pts_in_grids) else []
                if not isinstance(cell_list, list):
                    cell_list = [cell_list]

                # Get the length array for this grid
                lens = len_pts_in_grids[i] if i < len(len_pts_in_grids) else np.array([])
                if isinstance(lens, np.ndarray):
                    lens_list = lens.flatten().tolist()
                elif isinstance(lens, (list, tuple)):
                    lens_list = list(lens)
                else:
                    lens_list = [lens]

                # Pad to max_cells
                while len(cell_list) < max_cells:
                    cell_list.append(np.zeros((0, 3)))
                while len(lens_list) < max_cells:
                    lens_list.append(0.0)

                ins_array_padded.append(cell_list[:max_cells])
                len_array_padded.extend(lens_list[:max_cells])

            # Now reshape into proper format
            all_lens_flat = np.array(len_array_padded)

            # Only print if we have valid lengths
            valid_lens = all_lens_flat[all_lens_flat > 0]
            if len(valid_lens) > 0:
                print(f"\n  Multi-scale processing completed, extracted insulator length range: "
                      f"{np.min(valid_lens):.2f} - {np.max(valid_lens):.2f} m")
            else:
                print(f"\n  Multi-scale processing completed, but no valid insulators extracted")

            # Adaptive grid selection
            if max_cells > 0:
                # Reshape to (GridNum, NumCells) then transpose to (NumCells, GridNum)
                # ins_array_padded is already (GridNum, NumCells)
                ins_array = [[ins_array_padded[j][i] for j in range(grid_num)]
                             for i in range(max_cells)]

                # Reshape lengths similarly: (GridNum*NumCells,) -> (GridNum, NumCells) -> (NumCells, GridNum)
                len_array = all_lens_flat.reshape(grid_num, max_cells).T

                try:
                    fine_ins, fine_len, fine_grid, fine_ve = adaptive_grid_tension(ins_array, len_array)
                    # Count number of insulator segments (non-zero lengths)
                    num_insulators = np.count_nonzero(fine_len > 0)
                    print(f"  Adaptive grid selection completed, extracted {num_insulators} insulator segments, total {fine_ins.shape[0]} points")
                except Exception as e:
                    print(f"  Warning: Adaptive grid processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Use first grid result as fallback
                    # Find first non-empty cell
                    fine_ins = np.zeros((0, 3))
                    for cell in ins_pts_in_grids:
                        if isinstance(cell, list) and len(cell) > 0:
                            for item in cell:
                                if isinstance(item, np.ndarray) and item.shape[0] > 0:
                                    fine_ins = item
                                    break
                            if fine_ins.shape[0] > 0:
                                break
            else:
                # No valid cells extracted
                print(f"  Warning: Unable to extract valid insulator data")
                fine_ins = np.zeros((0, 3))

            # Decide whether to visualize:
            # 1. If --visualize is specified, always visualize
            # 2. If --no-visualize is specified, don't visualize
            # 3. If neither is specified, auto visualize for single tower, don't for batch
            should_visualize = args.visualize or (
                not args.no_visualize and args.tower_id is not None
            )

            # Save PCD files if --plot is enabled
            if args.plot:
                print(f"  Saving PCD point cloud file...")
                ins_pts_for_save = fine_ins[:, :3] if fine_ins.shape[0] > 0 else np.zeros((0, 3))
                save_colored_pcd(
                    tower_pts_r,
                    line_pts_r,
                    ins_pts_for_save,
                    args.output_dir,
                    tower_id
                )

            if fine_ins.shape[0] > 0:
                if should_visualize and not args.plot:
                    # Use matplotlib visualization if --plot is not enabled
                    print(f"  Visualizing results...")
                    visualize_results(tower_pts_r, line_pts_r, fine_ins[:, :3], tower_id)
                elif not args.plot:
                    print(f"  Extraction successful, extracted {fine_ins.shape[0]} insulator points")
            else:
                if not args.plot:
                    print(f"  Warning: No insulator points extracted")

            tower_results.append({
                'tower_id': tower_id,
                'tower_type': detected_tower_type,
                'tower_type_name': tower_type_name,
                'num_insulators': num_insulators,  # Number of insulator segments
                'insulator_points': fine_ins.shape[0] if fine_ins.shape[0] > 0 else 0,  # Number of insulator points
                'status': 'success'
            })

            print(f"\nTower {tower_id} processing completed")

        except Exception as e:
            print(f"  Error: Exception occurred while processing tower {tower_id}: {e}")
            import traceback
            traceback.print_exc()

            # Record failed tower
            tower_results.append({
                'tower_id': tower_id,
                'tower_type': -1,
                'tower_type_name': 'Processing Failed',
                'num_insulators': 0,
                'insulator_points': 0,
                'status': 'failed',
                'error': str(e)
            })
            continue

    # Output final statistics
    print("\n" + "=" * 60)
    print("Processing Completed - Final Statistics")
    print("=" * 60)

    if len(tower_results) > 0:
        print(f"\nTotal towers processed: {len(tower_results)}")
        print(f"Successfully processed: {sum(1 for r in tower_results if r['status'] == 'success')}")
        print(f"Processing failed: {sum(1 for r in tower_results if r['status'] == 'failed')}")

        # Group statistics by tower type
        print(f"\n{'='*60}")
        print("Detailed Results for Each Tower:")
        print(f"{'='*60}")
        print(f"{'Tower ID':<10} {'Type':<6} {'Type Name':<22} {'Segments':<12} {'Points':<12} {'Status'}")
        print(f"{'-'*60}")

        for result in tower_results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            tower_type_str = str(result['tower_type']) if result['tower_type'] != -1 else "N/A"
            print(f"{result['tower_id']:<10} {tower_type_str:<6} {result['tower_type_name']:<22} "
                  f"{result['num_insulators']:<12} {result['insulator_points']:<12} {status_symbol}")

        # Statistics by tower type
        print(f"\n{'='*60}")
        print("Statistics by Tower Type:")
        print(f"{'='*60}")

        type_stats = {}
        for result in tower_results:
            if result['status'] == 'success':
                t_type = result['tower_type']
                t_name = result['tower_type_name']
                if t_type not in type_stats:
                    type_stats[t_type] = {
                        'name': t_name,
                        'count': 0,
                        'total_insulators': 0,
                        'total_points': 0
                    }
                type_stats[t_type]['count'] += 1
                type_stats[t_type]['total_insulators'] += result['num_insulators']
                type_stats[t_type]['total_points'] += result['insulator_points']

        for t_type in sorted(type_stats.keys()):
            stats = type_stats[t_type]
            avg_insulators = stats['total_insulators'] / stats['count'] if stats['count'] > 0 else 0
            avg_points = stats['total_points'] / stats['count'] if stats['count'] > 0 else 0
            print(f"Type {t_type} ({stats['name']}): "
                  f"{stats['count']} towers, "
                  f"total {stats['total_insulators']} segments, "
                  f"total {stats['total_points']} points, "
                  f"avg {avg_insulators:.1f} segments/tower, "
                  f"avg {avg_points:.0f} points/tower")

        # Grand total
        total_insulators = sum(r['num_insulators'] for r in tower_results if r['status'] == 'success')
        total_points = sum(r['insulator_points'] for r in tower_results if r['status'] == 'success')
        successful_towers = sum(1 for r in tower_results if r['status'] == 'success')
        print(f"\nTotal insulator segments extracted: {total_insulators}")
        print(f"Total insulator points extracted: {total_points}")
        if successful_towers > 0:
            print(f"Average per tower: {total_insulators / successful_towers:.1f} insulator segments")
            print(f"Average per tower: {total_points / successful_towers:.0f} insulator points")

    print(f"\n{'='*60}")
    print("All towers processing completed")
    print(f"{'='*60}")


def save_colored_pcd(tower_pts, line_pts, ins_pts, output_path, tower_id='Unknown'):
    """
    Save colored point cloud to PCD format for CloudCompare visualization.

    Tower and line points are colored gray (uncolored), while detected insulator
    points are colored with bright distinct colors for each insulator instance.

    Parameters:
    -----------
    tower_pts : numpy.ndarray
        Tower points (N, 3)
    line_pts : numpy.ndarray
        Line points (N, 3)
    ins_pts : numpy.ndarray
        Insulator points (N, 3) or (N, 4) where 4th column is label
    output_path : str
        Output directory or file path
    tower_id : str
        Tower ID for filename
    """
    try:
        # Create output directory if it doesn't exist
        if os.path.isdir(output_path) or not output_path.endswith('.pcd'):
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f'{tower_id}_colored.pcd')
        else:
            output_file = output_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Prepare point lists and colors
        all_points = []
        all_colors = []

        # Define gray color for tower and line points (RGB values 0-255)
        gray_color = [128, 128, 128]  # Medium gray

        # Add tower points (gray)
        if len(tower_pts) > 0:
            all_points.append(tower_pts[:, :3])
            tower_colors = np.tile(gray_color, (len(tower_pts), 1))
            all_colors.append(tower_colors)

        # Add line points (gray)
        if len(line_pts) > 0:
            all_points.append(line_pts[:, :3])
            line_colors = np.tile(gray_color, (len(line_pts), 1))
            all_colors.append(line_colors)

        # Add insulator points with bright colors
        if len(ins_pts) > 0:
            # Check if ins_pts has label column (4th column)
            has_labels = ins_pts.shape[1] >= 4

            # Define bright distinct colors for insulators (RGB values 0-255)
            bright_colors = [
                [255, 0, 0],     # Red
                [0, 255, 0],     # Green
                [0, 0, 255],     # Blue
                [255, 255, 0],   # Yellow
                [255, 0, 255],   # Magenta
                [0, 255, 255],   # Cyan
                [255, 128, 0],   # Orange
                [128, 0, 255],   # Purple
                [255, 192, 203], # Pink
                [0, 128, 255],   # Light Blue
                [128, 255, 0],   # Lime
                [255, 0, 128],   # Rose
            ]

            if has_labels:
                # Process by labels
                labels = ins_pts[:, 3].astype(int)
                unique_labels = np.unique(labels[labels > 0])  # Exclude label 0 (background)

                ins_points = []
                ins_colors = []

                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    label_points = ins_pts[mask, :3]  # Extract X, Y, Z coordinates

                    # Assign color (cycle through bright_colors if more insulators than colors)
                    color = bright_colors[i % len(bright_colors)]
                    label_colors = np.tile(color, (len(label_points), 1))

                    ins_points.append(label_points)
                    ins_colors.append(label_colors)

                if ins_points:
                    all_points.extend(ins_points)
                    all_colors.extend(ins_colors)
            else:
                # No labels, use red color for all insulator points
                ins_color = [255, 0, 0]  # Red
                all_points.append(ins_pts[:, :3])
                ins_colors = np.tile(ins_color, (len(ins_pts), 1))
                all_colors.append(ins_colors)

        # Combine all points and colors
        if not all_points:
            print("Warning: No points to save in PCD file")
            return

        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)

        # Write PCD file
        num_points = len(combined_points)

        with open(output_file, 'w') as f:
            # PCD header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write("DATA ascii\n")

            # Write point data
            for i in range(num_points):
                x, y, z = combined_points[i]
                r, g, b = combined_colors[i].astype(np.uint8)

                # Convert RGB to packed format for PCD
                rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)

                f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed}\n")

        print(f"    Saved colored PCD file: {output_file}")
        print(f"    Total points: {num_points:,}")
        if len(ins_pts) > 0:
            has_labels = ins_pts.shape[1] >= 4
            if has_labels:
                unique_labels = np.unique(ins_pts[:, 3].astype(int))
                unique_labels = unique_labels[unique_labels > 0]
                print(f"    Insulator instances: {len(unique_labels)} (marked with different colors)")
            else:
                print(f"    Insulator points: {len(ins_pts)} (marked in red)")

    except Exception as e:
        print(f"Error saving colored PCD: {e}")
        import traceback
        traceback.print_exc()



def visualize_results(tower_pts, line_pts, ins_pts, tower_id='Unknown'):
    """
    Visualize tower, line, and insulator points

    Parameters:
    -----------
    tower_pts : numpy.ndarray
        Tower points
    line_pts : numpy.ndarray
        Line points
    ins_pts : numpy.ndarray
        Insulator points
    tower_id : str
        Tower ID for the title
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot tower points in green
    if tower_pts.shape[0] > 0:
        ax.scatter(tower_pts[:, 0], tower_pts[:, 1], tower_pts[:, 2],
                  c='green', marker='.', s=1, alpha=0.5, label='Tower')

    # Plot line points in blue
    if line_pts.shape[0] > 0:
        ax.scatter(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
                  c='blue', marker='.', s=1, alpha=0.5, label='Line')

    # Plot insulator points in red
    if ins_pts.shape[0] > 0:
        ax.scatter(ins_pts[:, 0], ins_pts[:, 1], ins_pts[:, 2],
                  c='red', marker='.', s=2, label='Insulator')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title(f'Insulator Extraction Results - Tower {tower_id}')

    # Set equal aspect ratio
    max_range = np.array([
        tower_pts[:, 0].max() - tower_pts[:, 0].min() if tower_pts.shape[0] > 0 else 1,
        tower_pts[:, 1].max() - tower_pts[:, 1].min() if tower_pts.shape[0] > 0 else 1,
        tower_pts[:, 2].max() - tower_pts[:, 2].min() if tower_pts.shape[0] > 0 else 1
    ]).max() / 2.0

    if tower_pts.shape[0] > 0:
        mid_x = (tower_pts[:, 0].max() + tower_pts[:, 0].min()) * 0.5
        mid_y = (tower_pts[:, 1].max() + tower_pts[:, 1].min()) * 0.5
        mid_z = (tower_pts[:, 2].max() + tower_pts[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
