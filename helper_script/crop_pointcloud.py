#!/usr/bin/env python3

import numpy as np
import os
import argparse
import glob

def read_pcd(filename):
    """Read PCD file using NumPy without Open3D dependency"""
    # Initialize variables
    points = []
    colors = []
    has_colors = False
    header_length = 0
    num_points = 0
    data_format = 'ascii'
    
    try:
        with open(filename, 'rb') as f:
            # Read header
            header = []
            line = f.readline().decode().strip()
            while line and not line.startswith('DATA'):
                header.append(line)
                if line.startswith('POINTS'):
                    num_points = int(line.split()[1])
                elif line.startswith('FIELDS'):
                    fields = line.split()[1:]
                    has_colors = 'rgb' in fields or ('r' in fields and 'g' in fields and 'b' in fields)
                elif line.startswith('DATA'):
                    data_format = line.split()[1]
                line = f.readline().decode().strip()
                header_length += 1
            
            # Handle ASCII format
            if data_format.lower() == 'ascii':
                data = np.loadtxt(f)
                if data.shape[1] >= 3:  # At least x,y,z
                    points = data[:, :3]
                    if data.shape[1] >= 6:  # Might have colors
                        colors = data[:, 3:6]
                        has_colors = True
            else:
                print(f"Binary PCD formats not supported in this simplified version")
                return None, None, False
    
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        return None, None, False
        
    return points, colors, has_colors

def write_pcd(filename, points, colors=None):
    """Write PCD file using NumPy without Open3D dependency"""
    with open(filename, 'w') as f:
        # Write header
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write("VERSION 0.7\n")
        
        if colors is not None:
            f.write("FIELDS x y z r g b\n")
            f.write("SIZE 4 4 4 4 4 4\n")
            f.write("TYPE F F F F F F\n")
            f.write("COUNT 1 1 1 1 1 1\n")
        else:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        # Write points (and colors if available)
        if colors is not None:
            # Combine points and colors
            data = np.hstack((points, colors))
            np.savetxt(f, data, fmt='%.6f')
        else:
            np.savetxt(f, points, fmt='%.6f')

def crop_point_cloud(input_file, output_file, x_min=None, x_max=None, 
                     y_min=None, y_max=None, z_min=None, z_max=None):
    """
    Read a point cloud, remove points outside specified bounds, and save the result
    """
    # Read the point cloud
    points, colors, has_colors = read_pcd(input_file)
    if points is None:
        return 0, 0
    
    # Create initial mask (all True)
    mask = np.ones(len(points), dtype=bool)
    
    # Apply filtering for each axis if specified
    if x_min is not None:
        mask = mask & (points[:, 0] >= x_min)
    if x_max is not None:
        mask = mask & (points[:, 0] <= x_max)
    if y_min is not None:
        mask = mask & (points[:, 1] >= y_min)
    if y_max is not None:
        mask = mask & (points[:, 1] <= y_max)
    if z_min is not None:
        mask = mask & (points[:, 2] >= z_min)
    if z_max is not None:
        mask = mask & (points[:, 2] <= z_max)
    
    # Filter points
    filtered_points = points[mask]
    
    # Fix: Only filter colors if they exist
    filtered_colors = None
    if has_colors and colors is not None and len(colors) > 0:
        filtered_colors = colors[mask]
    
    # Save the cropped point cloud
    write_pcd(output_file, filtered_points, filtered_colors)
    
    return len(points), len(filtered_points)

def main():
    parser = argparse.ArgumentParser(description='Crop point clouds along all axes')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing PCD files')
    parser.add_argument('--x_min', type=float, default=None, help='Minimum X value (default: no limit)')
    parser.add_argument('--x_max', type=float, default=None, help='Maximum X value (default: no limit)')
    parser.add_argument('--y_min', type=float, default=None, help='Minimum Y value (default: no limit)')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum Y value (default: no limit)')
    parser.add_argument('--z_min', type=float, default=None, help='Minimum Z value (default: no limit)')
    parser.add_argument('--z_max', type=float, default=None, help='Maximum Z value (default: no limit)')
    parser.add_argument('--suffix', type=str, default='_cropped', 
                        help='Suffix to add to output filenames (default: _cropped)')
    
    args = parser.parse_args()
    
    # Check if args.dir is a directory or a file
    if os.path.isfile(args.dir) and args.dir.endswith('.pcd'):
        # Single file mode
        pcd_files = [args.dir]
    else:
        # Directory mode
        pcd_files = glob.glob(os.path.join(args.dir, '*.pcd'))
    
    if not pcd_files:
        print(f"No PCD files found in {args.dir}")
        return
    
    print(f"Found {len(pcd_files)} PCD files. Processing with bounds:")
    if args.x_min is not None: print(f"  - X min: {args.x_min}")
    if args.x_max is not None: print(f"  - X max: {args.x_max}")
    if args.y_min is not None: print(f"  - Y min: {args.y_min}")
    if args.y_max is not None: print(f"  - Y max: {args.y_max}")
    if args.z_min is not None: print(f"  - Z min: {args.z_min}")
    if args.z_max is not None: print(f"  - Z max: {args.z_max}")
    
    for pcd_file in pcd_files:
        # Create output filename
        base_name = os.path.splitext(pcd_file)[0]
        output_file = f"{base_name}{args.suffix}.pcd"
        
        # Process the file
        original_count, remaining_count = crop_point_cloud(
            pcd_file, output_file, 
            args.x_min, args.x_max, 
            args.y_min, args.y_max, 
            args.z_min, args.z_max
        )
        
        print(f"Processed {pcd_file}:")
        print(f"  - Original points: {original_count}")
        print(f"  - Remaining points: {remaining_count}")
        print(f"  - Removed points: {original_count - remaining_count}")
        print(f"  - Saved to: {output_file}")
        
    print("Processing complete!")

if __name__ == "__main__":
    main()

    """
    # Example usage:
    # (grad_nbv) ok@ok:~/mr30_ws$ python3 src/helper_script/crop_pointcloud.py --dir /home/ok/mr30_ws/data/1604_sim/pcd/ --x_min 0.450 --y_min -0.317 --z_min 0.070 --x_max 0.980 --y_max 0.110 --z_max 0.457
    # """