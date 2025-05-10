#!/usr/bin/env python3
# filepath: /home/ok/mr30_ws/src/helper_script/extract_pointcloud_data.py
import rosbag
import os
import csv
import argparse
import math
from datetime import datetime
import tf.transformations
import numpy as np
import bisect
from collections import defaultdict, deque

def quaternion_to_euler(quaternion):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    # Extract quaternion components
    x, y, z, w = quaternion
    
    # Convert quaternion to Euler angles
    euler = tf.transformations.euler_from_quaternion([x, y, z, w])
    
    # Return roll, pitch, yaw in degrees
    return [math.degrees(angle) for angle in euler]

def build_transform_graph(tf_data):
    """Build a graph of transform relationships for path finding."""
    graph = defaultdict(set)
    for (parent, child) in tf_data:
        # Add bidirectional edges (we can transform in either direction)
        graph[parent].add(child)
        graph[child].add(parent)
    return graph

def find_transform_path(graph, start, goal):
    """Find a path from start frame to goal frame using BFS."""
    if start == goal:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def get_transform_between_frames(tf_data, from_frame, to_frame, timestamp):
    """Get transform between two frames at a specific timestamp."""
    # Check if direct transform exists
    key = (from_frame, to_frame)
    if key in tf_data and tf_data[key]:
        transforms = tf_data[key]
        timestamps = [tf['timestamp'].to_sec() for tf in transforms]
        msg_time = timestamp.to_sec()
        
        # Find index of the closest timestamp
        idx = bisect.bisect_left(timestamps, msg_time)
        
        # Adjust index if necessary
        if idx >= len(transforms):
            idx = len(transforms) - 1
        elif idx > 0 and idx < len(transforms):
            if abs(timestamps[idx] - msg_time) > abs(timestamps[idx-1] - msg_time):
                idx -= 1
        
        return transforms[idx], False  # Return transform and flag (not inverted)
    
    # Check reverse transform
    key = (to_frame, from_frame)
    if key in tf_data and tf_data[key]:
        transforms = tf_data[key]
        timestamps = [tf['timestamp'].to_sec() for tf in transforms]
        msg_time = timestamp.to_sec()
        
        # Find index of the closest timestamp
        idx = bisect.bisect_left(timestamps, msg_time)
        
        # Adjust index if necessary
        if idx >= len(transforms):
            idx = len(transforms) - 1
        elif idx > 0 and idx < len(transforms):
            if abs(timestamps[idx] - msg_time) > abs(timestamps[idx-1] - msg_time):
                idx -= 1
        
        return transforms[idx], True  # Return transform and flag (inverted)
    
    return None, False

def process_bag_files(input_dir, output_dir, pointcloud_topic=None, world_frame="world"):
    """
    Process each rosbag file in a directory to extract timestamp, step,
    and camera pose coordinates (x, y, z, roll, pitch, yaw) from TF data.
    
    Args:
        input_dir: Directory containing .bag files
        output_dir: Directory for output CSV files
        pointcloud_topic: ROS topic for pointcloud messages (if None, auto-detect)
        world_frame: World coordinate frame name
    """
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all bag files in the directory
    bag_files = [f for f in os.listdir(input_dir) if f.endswith('.bag')]
    
    if not bag_files:
        print(f"No bag files found in {input_dir}")
        return
    
    # Process each bag file
    for bag_file in sorted(bag_files):
        print(f"Processing {bag_file}...")
        bag_path = os.path.join(input_dir, bag_file)
        
        # Extract method name from filename (without extension)
        method = os.path.splitext(bag_file)[0]
        
        # Create output CSV path
        output_csv = os.path.join(output_dir, f"{method}.csv")
        
        try:
            # Open the bag file
            bag = rosbag.Bag(bag_path)
            
            # Get information about topics in the bag
            info = bag.get_type_and_topic_info()
            topics = info.topics
            print(f"  Available topics: {', '.join(topics.keys())}")
            
            # Auto-detect pointcloud topic if not specified
            if pointcloud_topic is None or pointcloud_topic not in topics:
                # Look for potential pointcloud topics
                potential_topics = []
                for topic, topic_info in topics.items():
                    msg_type = topic_info.msg_type
                    # Check for common pointcloud message types
                    if 'PointCloud' in msg_type:
                        potential_topics.append(topic)
                
                if potential_topics:
                    pointcloud_topic = potential_topics[0]
                    print(f"  Auto-detected pointcloud topic: {pointcloud_topic}")
                else:
                    print(f"  No pointcloud topic found in {bag_file}")
                    continue
            
            # First, collect all TF data
            tf_data = {}  # Dictionary to store all transforms by frame_id
            print(f"  Collecting TF data...")
            
            # Count TF messages
            tf_message_count = 0
            for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
                tf_message_count += 1
                for transform in msg.transforms:
                    frame_id = transform.header.frame_id
                    child_frame_id = transform.child_frame_id
                    
                    # Print some sample frame IDs
                    if tf_message_count <= 5:
                        print(f"    TF: {frame_id} -> {child_frame_id}")
                    
                    # Store transform data by parent to child frame
                    key = (frame_id, child_frame_id)
                    if key not in tf_data:
                        tf_data[key] = []
                    
                    tf_data[key].append({
                        'timestamp': transform.header.stamp,
                        'translation': transform.transform.translation,
                        'rotation': transform.transform.rotation
                    })
            
            print(f"  Collected {tf_message_count} TF messages with {len(tf_data)} unique transforms")
            
            # Sort TF data by timestamp
            for key in tf_data:
                tf_data[key].sort(key=lambda x: x['timestamp'].to_sec())
            
            # Build a graph of transform relationships
            transform_graph = build_transform_graph(tf_data)
            
            # Now, process pointcloud messages
            print(f"  Processing pointcloud messages from topic: {pointcloud_topic}")
            start_time = None
            step_counter = 0
            entries = []
            
            # Count total number of pointcloud messages
            pointcloud_count = bag.get_message_count(pointcloud_topic)
            print(f"  Found {pointcloud_count} pointcloud messages")
            
            for topic, msg, t in bag.read_messages(topics=[pointcloud_topic]):
                # Get timestamp from message header
                timestamp = msg.header.stamp
                
                if start_time is None:
                    start_time = timestamp
                
                # Increment step counter for each pointcloud
                step_counter += 1
                
                # Calculate total time in seconds since start
                total_time = (timestamp - start_time).to_sec()
                
                # Format timestamp for CSV
                timestamp_str = datetime.fromtimestamp(timestamp.to_sec()).strftime('%Y-%m-%d %H:%M:%S')
                
                # Extract camera position (x, y, z) from the TF data
                try:
                    # Get the frame_id from the pointcloud message
                    camera_frame = msg.header.frame_id
                    
                    # Clean frame names
                    clean_camera = camera_frame.lstrip('/')
                    clean_world = world_frame.lstrip('/')
                    
                    # Find transform path from camera to world
                    path = find_transform_path(transform_graph, clean_camera, clean_world)
                    
                    if path:
                        # Create a 4x4 identity matrix as starting point
                        transform_matrix = np.identity(4)
                        
                        # Apply each transform in the path
                        for i in range(len(path) - 1):
                            from_frame = path[i]
                            to_frame = path[i + 1]
                            
                            transform_data, inverted = get_transform_between_frames(tf_data, from_frame, to_frame, timestamp)
                            
                            if transform_data:
                                # Extract position and orientation
                                x = transform_data['translation'].x
                                y = transform_data['translation'].y
                                z = transform_data['translation'].z
                                
                                qx = transform_data['rotation'].x
                                qy = transform_data['rotation'].y
                                qz = transform_data['rotation'].z
                                qw = transform_data['rotation'].w
                                
                                # Create transform matrix
                                rot_matrix = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
                                trans_matrix = np.identity(4)
                                trans_matrix[0:3, 3] = [x, y, z]
                                link_transform = np.dot(rot_matrix, trans_matrix)
                                
                                # Invert if needed
                                if inverted:
                                    link_transform = np.linalg.inv(link_transform)
                                
                                # Apply to overall transform
                                transform_matrix = np.dot(transform_matrix, link_transform)
                        
                        # Extract final position and orientation
                        x, y, z = transform_matrix[0:3, 3]
                        q = tf.transformations.quaternion_from_matrix(transform_matrix)
                        roll, pitch, yaw = quaternion_to_euler(q)
                        
                        print(f"    Successfully computed transform from {camera_frame} to {world_frame}")
                    else:
                        print(f"    No transform path found between {camera_frame} and {world_frame}")
                        x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
                except Exception as e:
                    print(f"    Error extracting pose for step {step_counter}: {e}")
                    x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0
                
                # Create entry for this step
                entry = {
                    'timestamp': timestamp_str,
                    'total time': round(total_time, 3),
                    'step': step_counter,
                    'method': method,
                    'camera_x': x,
                    'camera_y': y,
                    'camera_z': z,
                    'camera_roll': roll,
                    'camera_pitch': pitch,
                    'camera_yaw': yaw
                }
                
                entries.append(entry)
                
                # Progress output for large bags
                if step_counter % 10 == 0:
                    print(f"  Processed {step_counter} messages")
            
            # Close the bag
            bag.close()
            
            # Write data to CSV for this bag file
            if entries:
                # CSV field names
                fieldnames = [
                    'timestamp', 'total time', 'step', 'method',
                    'camera_x', 'camera_y', 'camera_z',
                    'camera_roll', 'camera_pitch', 'camera_yaw'
                ]
                
                with open(output_csv, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(entries)
                
                print(f"  CSV file created: {output_csv} with {step_counter} entries")
            else:
                print(f"  No data extracted from {bag_file}")
            
        except Exception as e:
            print(f"Error processing {bag_file}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pointcloud and pose information from rosbag files")
    parser.add_argument("input_dir", help="Directory containing rosbag files")
    parser.add_argument("output_dir", help="Directory for output CSV files")
    parser.add_argument("--topic", default=None, help="Pointcloud topic name (auto-detect if not specified)")
    parser.add_argument("--world-frame", default="world", help="World frame name")
    
    args = parser.parse_args()
    
    process_bag_files(args.input_dir, args.output_dir, args.topic, args.world_frame)