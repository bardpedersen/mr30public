#!/usr/bin/env python3

import os
import numpy as np
import argparse
from skimage import measure
import trimesh
import xml.etree.ElementTree as ET
import subprocess
import tempfile
import shutil

def parse_sdf_file(sdf_file):
    """
    Parse an SDF/URDF/XACRO file and return mesh data or grid parameters.
    
    Args:
        sdf_file: Path to the input file
        
    Returns:
        Either (grid_data, grid_origin, grid_size) for raw SDF files
        Or a combined mesh for XML-based files
    """
    # First, check if it's a XACRO file that needs preprocessing
    if sdf_file.lower().endswith('.xacro'):
        print("Detected XACRO file, preprocessing...")
        urdf_content = process_xacro_file(sdf_file)
        if urdf_content:
            # Create a temporary file with the processed content
            with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as tmp_file:
                tmp_file.write(urdf_content.encode('utf-8'))
                tmp_path = tmp_file.name
            
            try:
                # Check if the processed content contains URDF or SDF
                if '<robot' in urdf_content:
                    print("Processed XACRO file contains URDF content")
                    mesh = parse_urdf(tmp_path)
                else:
                    print("Processed XACRO file contains SDF content")
                    mesh = parse_xml_sdf(tmp_path)
                return mesh, None, None
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        else:
            raise ValueError("Failed to preprocess XACRO file")
    
    # For other files, read first few characters to determine file type
    with open(sdf_file, 'r') as f:
        start = f.read(1024)  # Read more characters to better detect file type
    
    # Check if it's an XML file
    if start.startswith('<?xml') or start.strip().startswith('<'):
        # Determine if it's URDF or SDF based on root element
        if '<robot' in start:
            print("Detected URDF file")
            mesh = parse_urdf(sdf_file)
        else:
            print("Detected SDF file")
            mesh = parse_xml_sdf(sdf_file)
        return mesh, None, None  # Return mesh with dummy values for grid params
    else:
        # Assume it's a raw SDF file
        return parse_raw_sdf(sdf_file)

def resolve_mesh_uri(uri, sdf_file_path):
    """
    Resolve a mesh URI to an actual file path.
    
    Args:
        uri: URI string from SDF file
        sdf_file_path: Path to the SDF file (for relative paths)
    
    Returns:
        Absolute path to the mesh file
    """
    # Handle model:// URIs (common in Gazebo)
    if uri.startswith('model://'):
        # Try to find the model in standard Gazebo model paths
        model_name = uri.split('//')[1].split('/')[0]
        relative_path = '/'.join(uri.split('//')[1].split('/')[1:])
        
        # Check common Gazebo model paths
        gazebo_paths = [
            os.path.expanduser('~/.gazebo/models'),
            '/usr/share/gazebo/models',
            '/usr/local/share/gazebo/models'
        ]
        
        # Add ROS paths
        ros_package_path = os.environ.get('ROS_PACKAGE_PATH', '')
        if ros_package_path:
            for path in ros_package_path.split(':'):
                if path:
                    gazebo_paths.append(os.path.join(path, 'share', 'gazebo_models'))
        
        # Add the model directory from the SDF file location
        sdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
        model_dir = os.path.dirname(sdf_dir)
        if os.path.basename(sdf_dir) == 'model':
            gazebo_paths.append(os.path.dirname(model_dir))
        
        # Look for the model relative to the SDF file path
        # This is specific to the structure seen in the example
        package_dir = os.path.dirname(os.path.dirname(sdf_dir))
        models_dir = os.path.join(package_dir, 'models')
        if os.path.exists(models_dir):
            gazebo_paths.append(models_dir)
        
        # Search for the mesh file
        for gazebo_path in gazebo_paths:
            model_path = os.path.join(gazebo_path, model_name)
            if os.path.exists(model_path):
                mesh_path = os.path.join(model_path, relative_path)
                if os.path.exists(mesh_path):
                    return mesh_path
        
        print(f"Warning: Could not resolve model:// URI: {uri}")
        return None
    
    # Handle package:// URIs (ROS packages)
    elif uri.startswith('package://'):
        # Parse the URI to get package name and relative path
        package_path = uri[len('package://'):]
        package_name = package_path.split('/')[0]
        relative_path = '/'.join(package_path.split('/')[1:])
        
        # Try to find the ROS package
        # Method 1: Use rospack if available
        try:
            result = subprocess.run(['rospack', 'find', package_name],
                                  check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
            package_dir = result.stdout.strip()
            mesh_path = os.path.join(package_dir, relative_path)
            if os.path.exists(mesh_path):
                return mesh_path
        except (subprocess.SubprocessError, FileNotFoundError):
            # If rospack is not available, try other methods
            pass
        
        # Method 2: Check common ROS package paths
        ros_package_paths = [
            os.path.expanduser('~/catkin_ws/src'),
            os.path.expanduser('~/mr30_ws/src'),  # Add your workspace
            '/opt/ros/noetic/share',
            '/opt/ros/melodic/share',
            '/opt/ros/kinetic/share'
        ]
        
        # Add paths from ROS_PACKAGE_PATH environment variable
        if 'ROS_PACKAGE_PATH' in os.environ:
            ros_package_paths.extend(os.environ['ROS_PACKAGE_PATH'].split(':'))
        
        for package_path in ros_package_paths:
            potential_path = os.path.join(package_path, package_name, relative_path)
            if os.path.exists(potential_path):
                return potential_path
        
        # Method 3: Look relative to the current directory
        sdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
        workspace_root = os.path.dirname(os.path.dirname(sdf_dir))  # Go up two levels
        
        potential_paths = [
            os.path.join(workspace_root, package_name, relative_path),
            os.path.join(workspace_root, 'src', package_name, relative_path)
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                return path
        
        print(f"Warning: Could not resolve package:// URI: {uri}")
        return None
    
    # Handle file:// URIs
    elif uri.startswith('file://'):
        file_path = uri.replace('file://', '')
        if os.path.isabs(file_path):
            return file_path
        else:
            # Relative to the SDF file location
            sdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
            return os.path.normpath(os.path.join(sdf_dir, file_path))
    
    # Handle relative paths
    else:
        # Relative to the SDF file location
        sdf_dir = os.path.dirname(os.path.abspath(sdf_file_path))
        # Check if it's in the "meshes" directory at the same level as the SDF file
        mesh_dir = os.path.join(os.path.dirname(sdf_dir), 'meshes')
        if os.path.exists(os.path.join(mesh_dir, uri)):
            return os.path.join(mesh_dir, uri)
        # Check if it's relative to the SDF file
        return os.path.normpath(os.path.join(sdf_dir, uri))

def parse_xml_sdf(sdf_file):
    """
    Parse XML-based SDF file (Gazebo/ROS format)
    """
    try:
        tree = ET.parse(sdf_file)
        root = tree.getroot()
        
        # Extract mesh information from XML
        print("Detected XML-based SDF file")
        print("Building link hierarchy...")
        
        # Helper function to get transformation matrix from pose element
        def get_transform_from_pose(pose_elem):
            if pose_elem is not None and pose_elem.text:
                pose_values = [float(v) for v in pose_elem.text.split()]
                transform = np.eye(4)
                
                # Apply translation
                if len(pose_values) >= 3:
                    transform[0:3, 3] = pose_values[0:3]
                
                # Apply rotation
                if len(pose_values) >= 6:
                    roll, pitch, yaw = pose_values[3:6]
                    rotation = trimesh.transformations.euler_matrix(roll, pitch, yaw)
                    # Combine rotation with existing transform
                    transform = np.dot(transform, rotation)
                
                return transform
            return np.eye(4)  # Identity matrix if no pose
        
        # Find all models in the SDF
        models = root.findall(".//model")
        if not models:
            models = [root]  # Use root if no explicit model
            
        combined_mesh = None
        
        for model in models:
            model_name = model.get('name', 'unnamed_model')
            print(f"Processing model: {model_name}")
            
            # Get model pose
            model_transform = get_transform_from_pose(model.find("pose"))
            
            # Build a dictionary of all links and their local transforms
            links = model.findall(".//link")
            link_transforms = {}
            link_parent_frames = {}
            
            # First pass: collect all links and their local transforms
            for link in links:
                link_name = link.get('name', 'unnamed_link')
                pose_elem = link.find("pose")
                
                # Check if pose is relative to another frame
                frame = None
                if pose_elem is not None and 'frame' in pose_elem.attrib:
                    frame = pose_elem.attrib['frame']
                    link_parent_frames[link_name] = frame
                
                link_transforms[link_name] = get_transform_from_pose(pose_elem)
                print(f"  Found link: {link_name}" + (f" (relative to {frame})" if frame else ""))
            
            # Build joint parent-child relationships
            joints = model.findall(".//joint")
            joint_parents = {}
            joint_children = {}
            
            for joint in joints:
                parent_elem = joint.find("parent")
                child_elem = joint.find("child")
                
                if parent_elem is not None and child_elem is not None:
                    parent = parent_elem.text.strip()
                    child = child_elem.text.strip()
                    joint_parents[child] = parent
                    if parent not in joint_children:
                        joint_children[parent] = []
                    joint_children[parent].append(child)
                    print(f"  Found joint: {parent} -> {child}")
            
            # Find root links (those without parents)
            root_links = []
            for link_name in link_transforms:
                if link_name not in joint_parents and link_name not in link_parent_frames:
                    root_links.append(link_name)
            
            print(f"  Root links: {root_links}")
            
            # Function to compute global transform for a link
            link_global_transforms = {}
            
            def compute_global_transform(link_name, parent_transform=model_transform):
                if link_name in link_global_transforms:
                    return link_global_transforms[link_name]
                
                local_transform = link_transforms.get(link_name, np.eye(4))
                
                # If link has a parent frame specified, use that parent's transform
                if link_name in link_parent_frames:
                    parent_name = link_parent_frames[link_name]
                    if parent_name in link_global_transforms:
                        parent_transform = link_global_transforms[parent_name]
                    else:
                        # Compute parent transform first
                        parent_transform = compute_global_transform(parent_name, model_transform)
                
                # If link has a joint parent, use that parent's transform
                elif link_name in joint_parents:
                    parent_name = joint_parents[link_name]
                    if parent_name in link_global_transforms:
                        parent_transform = link_global_transforms[parent_name]
                    else:
                        # Compute parent transform first
                        parent_transform = compute_global_transform(parent_name, model_transform)
                
                # Compute global transform by combining parent and local transforms
                global_transform = np.dot(parent_transform, local_transform)
                link_global_transforms[link_name] = global_transform
                return global_transform
            
            # Compute global transforms for all links starting from root links
            for root_link in root_links:
                compute_global_transform(root_link)
            
            # Make sure all links have global transforms (in case there are disconnected links)
            for link_name in link_transforms:
                if link_name not in link_global_transforms:
                    compute_global_transform(link_name)
            
            # Now process all links using their correct global transforms
            for link in links:
                link_name = link.get('name', 'unnamed_link')
                link_global_transform = link_global_transforms.get(link_name, np.eye(4))
                
                print(f"  Processing link: {link_name}")
                
                # Process all visual elements in the link
                visuals = link.findall("visual")
                
                for visual in visuals:
                    visual_name = visual.get('name', 'unnamed_visual')
                    print(f"    Processing visual: {visual_name}")
                    
                    # Get visual pose relative to link
                    visual_transform = get_transform_from_pose(visual.find("pose"))
                    # Combine with link transform
                    visual_global_transform = np.dot(link_global_transform, visual_transform)
                    
                    # Find geometry element
                    geom_elem = visual.find("geometry")
                    if geom_elem is not None:
                        # Process mesh geometry
                        mesh_elem = geom_elem.find("mesh")
                        if mesh_elem is not None:
                            uri_elem = mesh_elem.find("uri")
                            if uri_elem is not None:
                                uri_text = uri_elem.text
                                print(f"      Mesh: {uri_text}")
                                
                                # Resolve and load mesh
                                mesh_path = resolve_mesh_uri(uri_text, sdf_file)
                                if mesh_path and os.path.exists(mesh_path):
                                    print(f"      Found mesh at: {mesh_path}")
                                    try:
                                        # Load mesh
                                        current_mesh = trimesh.load(mesh_path)
                                        
                                        # Apply scale if present
                                        scale_elem = mesh_elem.find("scale")
                                        if scale_elem is not None and scale_elem.text:
                                            scale_values = [float(v) for v in scale_elem.text.split()]
                                            if len(scale_values) == 3:
                                                scale_matrix = np.diag([scale_values[0], scale_values[1], scale_values[2], 1.0])
                                                # Apply scale first
                                                current_mesh.apply_transform(scale_matrix)
                                        
                                        # Apply the combined global transform
                                        current_mesh.apply_transform(visual_global_transform)
                                        
                                        # Add to combined mesh
                                        if combined_mesh is None:
                                            combined_mesh = current_mesh
                                        else:
                                            combined_mesh = trimesh.util.concatenate([combined_mesh, current_mesh])
                                            
                                    except Exception as e:
                                        print(f"      Error loading mesh: {e}")
                                else:
                                    print(f"      Could not find mesh file for URI: {uri_text}")
                        
                        # Process box geometry
                        box_elem = geom_elem.find("box")
                        if box_elem is not None:
                            size_elem = box_elem.find("size")
                            if size_elem is not None:
                                print(f"      Box")
                                try:
                                    # Parse box size
                                    size_values = [float(v) for v in size_elem.text.split()]
                                    if len(size_values) == 3:
                                        # Create box mesh
                                        box_mesh = trimesh.creation.box(extents=size_values)
                                        
                                        # Apply the combined global transform
                                        box_mesh.apply_transform(visual_global_transform)
                                        
                                        # Add to combined mesh
                                        if combined_mesh is None:
                                            combined_mesh = box_mesh
                                        else:
                                            combined_mesh = trimesh.util.concatenate([combined_mesh, box_mesh])
                                        
                                        print(f"      Created box with size: {size_values}")
                                except Exception as e:
                                    print(f"      Error creating box: {e}")
                        
                        # Process cylinder geometry
                        cylinder_elem = geom_elem.find("cylinder")
                        if cylinder_elem is not None:
                            radius_elem = cylinder_elem.find("radius")
                            length_elem = cylinder_elem.find("length")
                            if radius_elem is not None and length_elem is not None:
                                print(f"      Cylinder")
                                try:
                                    radius = float(radius_elem.text)
                                    length = float(length_elem.text)
                                    cylinder_mesh = trimesh.creation.cylinder(radius=radius, height=length)
                                    
                                    # Apply the combined global transform
                                    cylinder_mesh.apply_transform(visual_global_transform)
                                    
                                    # Add to combined mesh
                                    if combined_mesh is None:
                                        combined_mesh = cylinder_mesh
                                    else:
                                        combined_mesh = trimesh.util.concatenate([combined_mesh, cylinder_mesh])
                                        
                                    print(f"      Created cylinder with radius: {radius}, length: {length}")
                                except Exception as e:
                                    print(f"      Error creating cylinder: {e}")
                        
                        # Process sphere geometry
                        sphere_elem = geom_elem.find("sphere")
                        if sphere_elem is not None:
                            radius_elem = sphere_elem.find("radius")
                            if radius_elem is not None:
                                print(f"      Sphere")
                                try:
                                    radius = float(radius_elem.text)
                                    sphere_mesh = trimesh.creation.icosphere(radius=radius)
                                    
                                    # Apply the combined global transform
                                    sphere_mesh.apply_transform(visual_global_transform)
                                    
                                    # Add to combined mesh
                                    if combined_mesh is None:
                                        combined_mesh = sphere_mesh
                                    else:
                                        combined_mesh = trimesh.util.concatenate([combined_mesh, sphere_mesh])
                                        
                                    print(f"      Created sphere with radius: {radius}")
                                except Exception as e:
                                    print(f"      Error creating sphere: {e}")
        
        # If we found any meshes, return the combined mesh
        if combined_mesh:
            return combined_mesh
            
        # If we get here, we didn't find any valid geometry
        raise ValueError("No valid geometries found in the SDF file")
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        raise ValueError("The SDF file is not a valid XML file")

def parse_urdf(urdf_file):
    """
    Parse URDF file (ROS Unified Robot Description Format)
    
    Args:
        urdf_file: Path to the URDF file
    
    Returns:
        A trimesh object containing the combined mesh
    """
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        if root.tag != 'robot':
            raise ValueError("Not a valid URDF file: root element is not 'robot'")
            
        print("Building link hierarchy...")
        
        # Helper function to get transformation matrix from origin element
        def get_transform_from_origin(origin_elem):
            transform = np.eye(4)
            
            if origin_elem is not None:
                # Get xyz attributes (translation)
                if 'xyz' in origin_elem.attrib:
                    xyz = [float(v) for v in origin_elem.attrib['xyz'].split()]
                    if len(xyz) == 3:
                        transform[0:3, 3] = xyz
                
                # Get rpy attributes (rotation)
                if 'rpy' in origin_elem.attrib:
                    rpy = [float(v) for v in origin_elem.attrib['rpy'].split()]
                    if len(rpy) == 3:
                        roll, pitch, yaw = rpy
                        rotation = trimesh.transformations.euler_matrix(roll, pitch, yaw)
                        transform = np.dot(transform, rotation)
            
            return transform
            
        # Build a dictionary of all links and their transforms
        links = root.findall("link")
        link_transforms = {}
        
        for link in links:
            link_name = link.get('name', 'unnamed_link')
            link_transforms[link_name] = np.eye(4)  # Initialize with identity transform
            print(f"  Found link: {link_name}")
        
        # Build joint parent-child relationships
        joints = root.findall("joint")
        joint_parents = {}
        joint_children = {}
        joint_transforms = {}
        
        for joint in joints:
            joint_name = joint.get('name', 'unnamed_joint')
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            
            if parent_elem is not None and child_elem is not None:
                parent = parent_elem.get('link')
                child = child_elem.get('link')
                
                if parent and child:
                    joint_parents[child] = parent
                    if parent not in joint_children:
                        joint_children[parent] = []
                    joint_children[parent].append(child)
                    
                    # Get joint transform
                    origin_elem = joint.find("origin")
                    joint_transforms[joint_name] = get_transform_from_origin(origin_elem)
                    
                    print(f"  Found joint: {joint_name} - {parent} -> {child}")
        
        # Find root link (the one without parents)
        root_links = []
        for link_name in link_transforms:
            if link_name not in joint_parents:
                root_links.append(link_name)
        
        # If no root links found, use a fallback strategy
        if not root_links:
            print("WARNING: Could not find root link using parent-child relationships.")
            
            # Strategy 1: Use the first link as root
            if link_transforms:
                first_link = list(link_transforms.keys())[0]
                print(f"Using first link '{first_link}' as root.")
                root_links = [first_link]
            # Strategy 2: Look for links named "base_link" or containing "base"
            else:
                for link_name in link_transforms:
                    if "base_link" in link_name.lower() or "base" in link_name.lower():
                        print(f"Using '{link_name}' as root based on naming convention.")
                        root_links = [link_name]
                        break
        
        if not root_links:
            raise ValueError("Could not find root link in URDF. Please check the file structure.")
        
        print(f"  Root links: {root_links}")
        
        # Compute global transforms for all links
        link_global_transforms = {}
        
        def compute_global_transform(link_name, parent_transform=np.eye(4)):
            if link_name in link_global_transforms:
                return link_global_transforms[link_name]
            
            # Start with the link's local transform
            global_transform = np.dot(parent_transform, link_transforms[link_name])
            
            # Find the joint connecting this link to its parent
            for joint in joints:
                child_elem = joint.find("child")
                if child_elem is not None and child_elem.get('link') == link_name:
                    # Get the joint's transform
                    origin_elem = joint.find("origin")
                    joint_transform = get_transform_from_origin(origin_elem)
                    
                    # Apply the joint transform
                    global_transform = np.dot(parent_transform, joint_transform)
                    break
            
            # Store and return the global transform
            link_global_transforms[link_name] = global_transform
            
            # Recursively compute transforms for children
            if link_name in joint_children:
                for child_name in joint_children[link_name]:
                    compute_global_transform(child_name, global_transform)
                    
            return global_transform
        
        # Compute transforms starting from root links
        for root_link in root_links:
            compute_global_transform(root_link)
        
        # Process all links to build the mesh
        combined_mesh = None
        
        for link in links:
            link_name = link.get('name', 'unnamed_link')
            link_global_transform = link_global_transforms.get(link_name, np.eye(4))
            
            print(f"  Processing link: {link_name}")
            
            # Process all visual elements in the link
            visuals = link.findall("visual")
            
            for visual_idx, visual in enumerate(visuals):
                visual_name = visual.get('name', f'visual_{visual_idx}')
                print(f"    Processing visual: {visual_name}")
                
                # Get visual origin relative to link
                origin_elem = visual.find("origin")
                visual_transform = get_transform_from_origin(origin_elem)
                
                # Combine with link transform
                visual_global_transform = np.dot(link_global_transform, visual_transform)
                
                # Find geometry element
                geom_elem = visual.find("geometry")
                if geom_elem is not None:
                    # Process mesh geometry
                    mesh_elem = geom_elem.find("mesh")
                    if mesh_elem is not None:
                        filename = mesh_elem.get('filename')
                        if filename:
                            print(f"      Mesh: {filename}")
                            
                            # Resolve and load mesh
                            mesh_path = resolve_mesh_uri(filename, urdf_file)
                            if mesh_path and os.path.exists(mesh_path):
                                print(f"      Found mesh at: {mesh_path}")
                                try:
                                    # Load mesh
                                    current_mesh = trimesh.load(mesh_path)
                                    
                                    # Apply scale if present
                                    scale = mesh_elem.get('scale')
                                    if scale:
                                        scale_values = [float(v) for v in scale.split()]
                                        if len(scale_values) == 3:
                                            scale_matrix = np.diag([scale_values[0], scale_values[1], scale_values[2], 1.0])
                                            current_mesh.apply_transform(scale_matrix)
                                    
                                    # Apply the combined global transform
                                    current_mesh.apply_transform(visual_global_transform)
                                    
                                    # Add to combined mesh
                                    if combined_mesh is None:
                                        combined_mesh = current_mesh
                                    else:
                                        combined_mesh = trimesh.util.concatenate([combined_mesh, current_mesh])
                                        
                                except Exception as e:
                                    print(f"      Error loading mesh: {e}")
                            else:
                                print(f"      Could not find mesh file for path: {filename}")
                    
                    # Process box geometry
                    box_elem = geom_elem.find("box")
                    if box_elem is not None:
                        size = box_elem.get('size')
                        if size:
                            print(f"      Box")
                            try:
                                size_values = [float(v) for v in size.split()]
                                if len(size_values) == 3:
                                    box_mesh = trimesh.creation.box(extents=size_values)
                                    box_mesh.apply_transform(visual_global_transform)
                                    
                                    if combined_mesh is None:
                                        combined_mesh = box_mesh
                                    else:
                                        combined_mesh = trimesh.util.concatenate([combined_mesh, box_mesh])
                                    
                                    print(f"      Created box with size: {size_values}")
                            except Exception as e:
                                print(f"      Error creating box: {e}")
                    
                    # Process cylinder geometry
                    cylinder_elem = geom_elem.find("cylinder")
                    if cylinder_elem is not None:
                        radius = cylinder_elem.get('radius')
                        length = cylinder_elem.get('length')
                        if radius and length:
                            print(f"      Cylinder")
                            try:
                                radius = float(radius)
                                length = float(length)
                                cylinder_mesh = trimesh.creation.cylinder(radius=radius, height=length)
                                cylinder_mesh.apply_transform(visual_global_transform)
                                
                                if combined_mesh is None:
                                    combined_mesh = cylinder_mesh
                                else:
                                    combined_mesh = trimesh.util.concatenate([combined_mesh, cylinder_mesh])
                                
                                print(f"      Created cylinder with radius: {radius}, length: {length}")
                            except Exception as e:
                                print(f"      Error creating cylinder: {e}")
                    
                    # Process sphere geometry
                    sphere_elem = geom_elem.find("sphere")
                    if sphere_elem is not None:
                        radius = sphere_elem.get('radius')
                        if radius:
                            print(f"      Sphere")
                            try:
                                radius = float(radius)
                                sphere_mesh = trimesh.creation.icosphere(radius=radius)
                                sphere_mesh.apply_transform(visual_global_transform)
                                
                                if combined_mesh is None:
                                    combined_mesh = sphere_mesh
                                else:
                                    combined_mesh = trimesh.util.concatenate([combined_mesh, sphere_mesh])
                                
                                print(f"      Created sphere with radius: {radius}")
                            except Exception as e:
                                print(f"      Error creating sphere: {e}")
        
        # If we found any meshes, return the combined mesh
        if combined_mesh:
            return combined_mesh
            
        # If we get here, we didn't find any valid geometry
        raise ValueError("No valid geometries found in the URDF file")
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        raise ValueError("The URDF file is not a valid XML file")

def parse_raw_sdf(sdf_file):
    """Parse raw SDF grid file format"""
    with open(sdf_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header information
    header_info = lines[0].strip().split()
    nx, ny, nz = int(header_info[0]), int(header_info[1]), int(header_info[2])
    grid_origin = np.array([float(header_info[3]), float(header_info[4]), float(header_info[5])])
    grid_size = float(header_info[6])
    
    # Parse the SDF values
    grid_data = np.zeros((nx, ny, nz))
    data_index = 0
    
    for i in range(1, len(lines)):
        values = lines[i].strip().split()
        for val in values:
            if data_index < nx * ny * nz:
                x = data_index // (ny * nz)
                y = (data_index % (ny * nz)) // nz
                z = data_index % nz
                grid_data[x, y, z] = float(val)
                data_index += 1
    
    return grid_data, grid_origin, grid_size

def convert_sdf_to_mesh(grid_data, grid_origin, grid_size):
    """
    Convert SDF grid to a triangle mesh using marching cubes algorithm.
    
    Args:
        grid_data: 3D numpy array of SDF values
        grid_origin: Origin coordinates of the grid
        grid_size: Size of each voxel in the grid
        
    Returns:
        vertices: Mesh vertices
        faces: Mesh faces
    """
    # Generate mesh using marching cubes algorithm at isovalue 0
    # (the surface is defined where SDF = 0)
    vertices, faces, _, _ = measure.marching_cubes(grid_data, level=0)
    
    # Transform vertices to world coordinates
    vertices = vertices * grid_size + grid_origin
    
    return vertices, faces

def save_mesh_to_stl(mesh_data, output_file):
    """
    Save mesh data to an STL file.
    
    Args:
        mesh_data: Either a trimesh.Trimesh object or (vertices, faces) tuple
        output_file: Path to save the STL file
    """
    if isinstance(mesh_data, trimesh.Trimesh):
        mesh = mesh_data
    else:
        vertices, faces = mesh_data
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    mesh.export(output_file)

def process_xacro_file(xacro_file):
    """
    Process a XACRO file using the xacro command-line tool.
    
    Args:
        xacro_file: Path to the XACRO file
    
    Returns:
        Processed URDF content as string or None if processing failed
    """
    try:
        # Check if xacro is available
        subprocess.run(['which', 'xacro'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Error: xacro command not found. Please install ROS or xacro package.")
        return None
    
    try:
        # Get absolute path of the xacro file
        abs_xacro_path = os.path.abspath(xacro_file)
        xacro_dir = os.path.dirname(abs_xacro_path)
        
        # Process xacro file, ensuring we're in the right directory for relative includes
        current_dir = os.getcwd()
        try:
            # Change to the xacro file directory to help with relative paths
            os.chdir(xacro_dir)
            
            # First try with ROS environment
            try:
                result = subprocess.run([
                    'rosrun', 'xacro', 'xacro', abs_xacro_path,
                    'arms_interface:=PositionJointInterface',  # Add default parameters 
                    'grippers_interface:=PositionJointInterface',
                    'yumi_setup:=robot_centric',
                    '--inorder'
                ], 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=30)
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"Error using rosrun xacro with parameters, trying direct xacro command: {e}")
                
                # Fallback to direct xacro command
                result = subprocess.run(['xacro', abs_xacro_path], 
                                      check=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      timeout=30)  # Add timeout to avoid hanging
                return result.stdout
        finally:
            # Change back to original directory
            os.chdir(current_dir)
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing XACRO file: {e}")
        print(f"STDERR: {e.stderr}")
        
        # Try to provide more helpful error message based on common issues
        if "No such package" in e.stderr or "No such file" in e.stderr:
            print("Hint: The XACRO file uses ROS package paths that couldn't be resolved.")
            print("Make sure your ROS environment is properly set up with 'source /opt/ros/YOUR_ROS_DISTRO/setup.bash'")
            print("And any workspace containing required packages is also sourced.")
        
        return None
    except Exception as e:
        print(f"Unexpected error processing XACRO file: {e}")
        return None

def apply_transform_to_mesh(mesh, translation=None, rotation=None):
    """
    Apply translation and rotation transforms to a mesh.
    
    Args:
        mesh: A trimesh.Trimesh object or (vertices, faces) tuple
        translation: [x, y, z] translation values in meters
        rotation: [roll, pitch, yaw] rotation values in radians
        
    Returns:
        The transformed mesh
    """
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    if rotation is None:
        rotation = [0.0, 0.0, 0.0]
        
    # Create a transformation matrix
    transform = np.eye(4)
    
    # Apply translation
    transform[0:3, 3] = translation
    
    # Apply rotation (roll, pitch, yaw)
    if any(rotation):
        roll, pitch, yaw = rotation
        rotation_matrix = trimesh.transformations.euler_matrix(roll, pitch, yaw)
        transform = np.dot(transform, rotation_matrix)
    
    # Apply transform to mesh
    if isinstance(mesh, trimesh.Trimesh):
        mesh.apply_transform(transform)
    else:
        # If we have vertices and faces
        vertices, faces = mesh
        # Create a temporary trimesh object to apply the transform
        temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        temp_mesh.apply_transform(transform)
        # Return the transformed vertices and faces
        mesh = (temp_mesh.vertices, temp_mesh.faces)
    
    return mesh

def main():
    parser = argparse.ArgumentParser(description='Convert SDF/URDF/XACRO file to STL format')
    parser.add_argument('input_file', help='Input file path (SDF, URDF, or XACRO)')
    parser.add_argument('output_file', help='Output STL file path')
    
    # Add translation and rotation options
    parser.add_argument('--translate', '-t', nargs=3, type=float, default=[0.5, 0.2, 0.2],
                        metavar=('X', 'Y', 'Z'),
                        help='Translation to apply (x, y, z) in meters')
    parser.add_argument('--rotate', '-r', nargs=3, type=float, default=[0.0, 0.0, -1.507],
                        metavar=('ROLL', 'PITCH', 'YAW'),
                        help='Rotation to apply (roll, pitch, yaw) in radians')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process input file
    print(f"Reading file: {args.input_file}")
    result = parse_sdf_file(args.input_file)
    
    # Check result type to determine how to handle it
    if isinstance(result[0], trimesh.Trimesh):
        # We got a mesh directly
        print("Processing XML file with mesh references...")
        mesh = result[0]
    else:
        # We got grid data (only for raw SDF)
        print("Processing raw SDF grid data...")
        grid_data, grid_origin, grid_size = result
        print("Converting SDF to mesh using marching cubes...")
        vertices, faces = convert_sdf_to_mesh(grid_data, grid_origin, grid_size)
        mesh = (vertices, faces)
    
    # Apply transform if specified
    if any(args.translate) or any(args.rotate):
        print(f"Applying translation {args.translate} and rotation {args.rotate}...")
        mesh = apply_transform_to_mesh(mesh, args.translate, args.rotate)
    
    # Save as STL
    print(f"Saving mesh to STL file: {args.output_file}")
    save_mesh_to_stl(mesh, args.output_file)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()