#!/usr/bin/env python3

import torch
import rospy
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array, look_at_rotation
from utils.torch_utils import transform_from_rotation_translation

class GridPlanner:
    """
    Grid-based cylindrical viewpoint planner that moves around a target in a cylindrical pattern.
    """
    def __init__(
        self,
        start_pose: np.array,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.002]),
        grid_center: np.array = np.array([0.5, -0.4, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array(
            [
                [685.5028076171875, 0.0, 485.35955810546875],
                [0.0, 685.6409912109375, 270.7330627441406],
                [0.0, 0.0, 1.0],
            ],
        ),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
        camera_bounds_in: np.array = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        radius=0.25, 
        length=0.15, 
        num_points_x=4, 
        num_points_y=6, 
        length_start=-0.3,
        radius_start= np.pi/8,
        radius_stop= 7*np.pi/8
    ) -> None:
        """
        Initialize the planner
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Voxel grid parameters
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        
        # Initialize camera and target parameters
        self.camera_bounds_in = camera_bounds_in
        self.start_params(start_pose, target_params)
        
        # Initialize voxel grid
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
            width=image_size[0],
            height=image_size[1],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            target_params=self.target_params,
            device=self.device,
        )
        
        # Cylindrical grid parameters
        self.radius = radius
        self.length = length
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.length_start = length_start
        self.length_stop = length
        self.radius_start = radius_start
        self.radius_stop = radius_stop
        
        # Center of the cylindrical pattern
        self.cylinder_center = np.array(grid_center.cpu().numpy())
        
        # Initialize visualization and points
        self.rviz_visualizer = RvizVisualizer()
        self.viewpoint = start_pose
        self.points = self.create_points()
        self.current_point_idx = 0

    def start_params(self, start_pose: np.array, target_params: np.array) -> None:
        """
        Initialize the parameters for camera and target bounds
        """
        self.camera_params = torch.tensor(
            start_pose[:7],
            dtype=torch.float32,
            device=self.device,
        )
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        self.camera_bounds = torch.tensor(
            [
                [
                    start_pose[0] - self.camera_bounds_in[0],
                    start_pose[1] - self.camera_bounds_in[2],
                    start_pose[2] - self.camera_bounds_in[4],
                    target_params[0] - self.camera_bounds_in[0],
                    target_params[1] - self.camera_bounds_in[2],
                    target_params[2] - self.camera_bounds_in[4],
                ],
                [
                    start_pose[0] + self.camera_bounds_in[1],
                    start_pose[1] + self.camera_bounds_in[3],
                    start_pose[2] + self.camera_bounds_in[5],
                    target_params[0] + self.camera_bounds_in[1],
                    target_params[1] + self.camera_bounds_in[3],
                    target_params[2] + self.camera_bounds_in[5],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        
    def create_points(self):
        """Create cylindrical grid of viewpoints around the target"""
        x_values = np.linspace(self.length_start, self.length_stop, self.num_points_x)
        theta_values = np.linspace(self.radius_start, self.radius_stop, self.num_points_y)        
        points = []
        
        for x in x_values:
            for theta in theta_values:
                # Compute point on cylinder surface
                y = self.radius * np.cos(theta)
                z = self.radius * np.sin(theta)
                point = np.array([x, y, z]) + self.cylinder_center
                target_pos = self.target_params.cpu().numpy()
                orientation = look_at_rotation(point, target_pos)
                pose = list(point) + list(orientation)
                points.append(pose)
        return points
    
    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(
            viewpoint[3:], dtype=torch.float32, device=self.device
        )
        transform = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage
    
    def next_best_view(self, N):
        """
        Return the next viewpoint in the cylindrical sequence
        """
        if len(self.points) == 0:
            rospy.logwarn("No viewpoints available in grid planner")
            return None, 0, 0
        
        if self.current_point_idx >= len(self.points):
            rospy.loginfo("Completed all viewpoints in grid pattern, starting again")
            self.current_point_idx = 0
        
        # Get next viewpoint
        next_point = self.points[self.current_point_idx]
        self.current_point_idx += 1
        
        # Visualize the current and future viewpoints
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(next_point))
        self.rviz_visualizer.visualize_view_samples(numpy_to_pose_array(self.points))
        
        # Calculate gain for this viewpoint
        camera_position = torch.tensor(
            next_point[:3], 
            dtype=torch.float32, 
            device=self.device
        )
        loss, gain_image = self.voxel_grid.compute_gain(
            camera_position, self.target_params
        )
        loss = loss.detach().cpu().numpy()
        
        # Display gain image
        self.rviz_visualizer.visualize_gain_image(gain_image)
        
        # Update viewpoint
        self.viewpoint = next_point
        self.camera_params = torch.tensor(
            next_point,
            dtype=torch.float32,
            device=self.device,
        )
        rospy.loginfo(f"Moving to viewpoint {self.current_point_idx}/{len(self.points)}")
        
        # Return the viewpoint, its gain, and total points
        
        # Plot metrics
        return self.viewpoint, loss, len(self.points)
    
    def visualize(self):
        """
        Visualize the voxel grid and planned viewpoints
        """
        # Visualize voxel grid
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        
        # Visualize camera bounds
        camera_bounds_np = self.camera_bounds[:, :3].cpu().numpy()
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds_np)
        
        # Visualize all planned viewpoints
        self.rviz_visualizer.visualize_view_samples(numpy_to_pose_array(self.points))