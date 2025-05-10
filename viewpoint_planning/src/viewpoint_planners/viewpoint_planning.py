import rospy
import numpy as np

import csv
import torch
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from perception.perceiver import Perceiver
from viewpoint_planners.move_camera import MoveCamera
from viewpoint_planners.grid_planner import GridPlanner 
from abb_control.arm_control_client import ArmControlClient
from viewpoint_planners.random_planner import RandomPlanner
from viewpoint_planners.sample_planner import SamplePlanner
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from viewpoint_planners.frontier_planner import FrontierNBVPlanner  
from viewpoint_planners.gradientnbv_planner import GradientNBVPlanner

# Import zivid capture from script, make use of yaml file
import os
import sys
zivid_scripts_path = os.path.join(os.path.expanduser('~/mr30_ws/src/zivid-ros/zivid_samples/scripts'))
sys.path.append(zivid_scripts_path)
from sample_capture_with_settings_from_yml import Sample
from utils.py_utils import numpy_to_pose
# Now to capture image with Zivid only use these lines, zivid node needs to be running from moveit.launch
"""
s = Sample()
s.capture()
"""

class ViewpointPlanning:
    def __init__(self):
        self.robot = rospy.get_param('/viewpoint_planning/robot', True)
        self.method = rospy.get_param('/viewpoint_planning/method', 'gradient_nbv')
        log_dir_str= rospy.get_param('/viewpoint_planning/log_dir', '~/mr30_ws/data/logs')
        self.log_dir = Path(os.path.expanduser(log_dir_str))

        if self.robot:
            self.arm_control = ArmControlClient()
            self.camera_bounds = np.array([1.2, 0.4, 0.5, 1.0, 0.05, 1.1]) # -x, +x, -y, +y, -z, +z from the target
        else:
            self.move_camera = MoveCamera()
            self.camera_bounds = np.array([1, 1, 0.5, 1.5, 0.2, 1])

        self.inner_bounds = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) # -x, +x, -y, +y, -z, +z from the target
        self.target_bounds = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])           
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler(10) # number of samples
        self.N = 1
        self.config()
        # Gradient-based planner
        self.gradient_planner = GradientNBVPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=5,
            camera_bounds_in=self.camera_bounds,
            target_bounds_in=self.target_bounds,
            inner_bounds_in=self.inner_bounds,
        )
        self.gradient_planner.visualize()
        # Random planner
        self.random_planner = RandomPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
            camera_bounds_in=self.camera_bounds,
            inner_bounds_in=self.inner_bounds, 
        )
        self.sample_planner = SamplePlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=20, 
            camera_bounds_in=self.camera_bounds,
            target_bounds_in=self.target_bounds,
            inner_bounds_in=self.inner_bounds,
        )
        self.frontier_planner = FrontierNBVPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=20,  # Number of viewpoint samples to evaluate
            camera_bounds_in=self.camera_bounds,
            target_bounds_in=self.target_bounds,
            inner_bounds_in=self.inner_bounds,
        )
        self.grid_planner = GridPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            camera_bounds_in=self.camera_bounds,
            radius=0.40,  
            length=0.25,  
            length_start=-0.2, 
            num_points_x=4,  
            num_points_y=7,
        )
                
        # Setup logging
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.method}_log_{timestamp}.csv"
        self.initialize_log_file()
        self.start_time = datetime.datetime.now()

    def run(self):
        if self.method == "random":
            sucsess = self.run_random()
        elif self.method == "sample_nbv":
            sucsess = self.run_five_dof_sampler()
        elif self.method == "frontier_nbv":
            sucsess = self.run_frontier_nbv()
        elif self.method == "grid":
            sucsess = self.run_grid()
        else:
            sucsess = self.run_gradient_nbv()

        if sucsess:
            self.N += 1

    def config(self):
        # Configure target
        self.target_position = np.array([0.7, 0, 0.25]) # where the object is

        # Configure initial camera viewpoint
        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(
            self.target_position,
            distance=0.3 # starting distance from the object
        )
        if self.robot:
            self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        else:
            self.move_camera.move(numpy_to_pose(self.camera_pose))
        # Configure scene
        self.grid_size = np.array([0.4, 0.4, 0.4]) # size of the voxel grid in meters
        self.grid_center = self.target_position
        # Configure camera
        camera_info = self.perceiver.get_camera_info()
        while camera_info == None:
            camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)

    def run_gradient_nbv(self):
        
        if self.method == "grad_nbv_pso":
            self.camera_pose, loss, iters = self.gradient_planner.next_best_view_pso(None, self.N)
        elif self.method == "grad_nbv_reset":
            self.camera_pose, loss, iters = self.gradient_planner.next_best_view_with_camera_reset(None, self.N)
        elif self.method == "grad_nbv_subop":
            self.camera_pose, loss, iters = self.gradient_planner.next_best_view_suboptimal(None, self.N)
        else:
            self.camera_pose, loss, iters = self.gradient_planner.next_best_view(None, self.N)
        
        for attempt in range(4):
            if self.robot:
                is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
            else:
                is_success = self.move_camera.move(numpy_to_pose(self.camera_pose))

            if is_success:
                rospy.sleep(5.0)
                depth_image, points, semantics = self.perceiver.run()
                coverage = self.gradient_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
                print("Target coverage: ", coverage, "Loss: ", loss, "Iters: ", iters)
                self.gradient_planner.visualize()

                # Save metrics
                self.save_metrics("gradient_nbv", coverage, loss, iters)
                self.plot_voxel_metrics(loss)
                return is_success
            else:
                self.rotate_camera()
        return False

    def run_frontier_nbv(self):
        self.camera_pose, score, _ = self.frontier_planner.next_best_view(self.N)
        for attempt in range(4):
            if self.robot:
                is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
            else:
                is_success = self.move_camera.move(numpy_to_pose(self.camera_pose))
            if is_success:
                rospy.sleep(5.0)
                depth_image, points, semantics = self.perceiver.run()
                coverage = self.frontier_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
                print("Target coverage: ", coverage, "Frontier score: ", score)
                self.frontier_planner.visualize()
                
                # Save metrics
                self.save_metrics("frontier_nbv", coverage, score)
                self.plot_voxel_metrics()
                return is_success
            else:
                self.rotate_camera()
        return False
    
    # note: this does not move the target
    def run_five_dof_sampler(self):
        self.camera_pose, loss, iters = self.sample_planner.next_best_view(self.N)
        for attempt in range(4):
            if self.robot:
                is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
            else:
                is_success = self.move_camera.move(numpy_to_pose(self.camera_pose))
            if is_success:
                rospy.sleep(5.0)
                depth_image, points, semantics = self.perceiver.run()
                coverage = self.sample_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
                print("Target coverage: ", coverage, "Loss: ", loss, "Iters: ", iters)
                self.sample_planner.visualize()

                # Save metrics
                self.save_metrics("five_dof_sampler", coverage, loss, iters)
                self.plot_voxel_metrics(loss)
                return is_success
            else:
                self.rotate_camera()
        return False

    def run_random(self):
        self.camera_pose, loss, _ = self.random_planner.next_best_view(self.N)
        for attempt in range(4):
            if self.robot:
                is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
            else:
                is_success = self.move_camera.move(numpy_to_pose(self.camera_pose))
            if is_success:
                rospy.sleep(5.0)
                depth_image, points, semantics = self.perceiver.run()
                coverage = self.random_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
                print("Target coverage: ", coverage, "Loss: ", loss)
                self.random_planner.visualize()

                # Save metrics
                self.save_metrics("random", coverage, loss)
                self.plot_voxel_metrics(loss)
                return is_success
            else:
                self.rotate_camera()
        return False

    def run_grid(self):
        """Run the grid-based cylindrical viewpoint planner"""
        self.camera_pose, loss, total_points = self.grid_planner.next_best_view(self.N)
        for attempt in range(4):
            if self.robot:
                is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
            else:
                is_success = self.move_camera.move(numpy_to_pose(self.camera_pose))
            if is_success:
                rospy.sleep(5.0)
                depth_image, points, semantics = self.perceiver.run()
                coverage = self.grid_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
                print("Target coverage:", coverage, "Loss:", loss, 
                    f"Viewpoint: {self.grid_planner.current_point_idx}/{total_points}")
                self.grid_planner.visualize()

                # Save metrics
                self.save_metrics("grid", coverage, loss, self.grid_planner.current_point_idx)
                self.plot_voxel_metrics(loss)
                return is_success
            else:
                self.rotate_camera()
        return False

    def initialize_log_file(self):
        """Create CSV file with headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total time', 'step', 'method', 'coverage', 
                'loss', 'iterations', 'camera_x', 'camera_y', 'camera_z'
            ])

    def save_metrics(self, method, coverage, loss=None, iterations=None, camera_pose=None):
        """
        Save metrics to CSV file
        
        :param method: Planning method used (string)
        :param coverage: Target coverage percentage
        :param loss: Loss value from optimization (optional)
        :param iterations: Number of iterations performed (optional)
        :param camera_pose: Final camera pose (optional)
        """
        if camera_pose is None:
            camera_pose = self.camera_pose
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time = "{:.3f}".format((datetime.datetime.now() - self.start_time).total_seconds())
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                total_time,
                self.N,
                method,
                coverage,
                loss if loss is not None else "N/A",
                iterations if iterations is not None else "N/A",
                camera_pose[0],  # x position
                camera_pose[1],  # y position
                camera_pose[2],  # z position
            ])

    def rotate_camera(self):
        """
        Rotate the camera around the z-axis by 90 degrees
        """        
        # Get current quaternion
        qx = self.camera_pose[3]
        qy = self.camera_pose[4]
        qz = self.camera_pose[5]
        qw = self.camera_pose[6]
        
        # Create quaternion for 90-degree rotation around Z
        rot_w = 0.7071068  # cos(π/4)
        rot_z = 0.7071068  # sin(π/4)
        
        # Quaternion multiplication
        new_qw = rot_w * qw - rot_z * qz
        new_qx = rot_w * qx - rot_z * qy
        new_qy = rot_w * qy + rot_z * qx
        new_qz = rot_w * qz + rot_z * qw
        
        # Normalize the quaternion
        norm = np.sqrt(new_qw**2 + new_qx**2 + new_qy**2 + new_qz**2)
        new_qw /= norm
        new_qx /= norm
        new_qy /= norm
        new_qz /= norm
        
        # Update the camera pose quaternion
        self.camera_pose[3] = new_qx
        self.camera_pose[4] = new_qy
        self.camera_pose[5] = new_qz
        self.camera_pose[6] = new_qw

    def plot_voxel_metrics(self, loss=None):
        """
        Plot metrics about the voxel grid state
        """
        # Define tracking history length
        N = self.N

        # Determine which planner is active based on the method
        if self.method == "random":
            planner = self.random_planner
        elif self.method == "sample_nbv":
            planner = self.sample_planner
        elif self.method == "frontier_nbv":
            planner = self.frontier_planner
        elif self.method == "grid":
            planner = self.grid_planner
        else:
            planner = self.gradient_planner
        
        # Get voxel grid data from the active planner
        voxel_grid = planner.voxel_grid.voxel_grid
        device = planner.device
        
        # Count occupied voxels (occupancy > 0.5)
        occupied_voxels = torch.sum(voxel_grid[..., 1] > 0.5).item()
        free_voxels = torch.sum(voxel_grid[..., 1] < 0.5).item()
        total_voxels = planner.voxel_grid.voxel_dims.prod().item()
        unknown_voxels = total_voxels - occupied_voxels - free_voxels
        
        # Calculate coverage of target region
        if hasattr(planner.voxel_grid, 'target_bounds') and planner.voxel_grid.target_bounds is not None:
            bounds = planner.voxel_grid.target_bounds
            target_voxels = voxel_grid[
                bounds[0]:bounds[3],
                bounds[1]:bounds[4],
                bounds[2]:bounds[5],
                1
            ]
            observed_target_voxels = torch.sum((target_voxels != 0.5)).item()
            target_total_voxels = target_voxels.numel()
            coverage_percentage = observed_target_voxels / target_total_voxels * 100
        else:
            coverage_percentage = 0
            observed_target_voxels = 0
            target_total_voxels = 0
        
        # Calculate uncertain voxels (for occupancy and semantics)
        uncertain_occ_voxels = torch.sum((voxel_grid[..., 1] > 0.4) & (voxel_grid[..., 1] < 0.6)).item()
        
        # Get current camera position and target position
        if hasattr(planner, 'camera_params'):
            camera_pos = planner.camera_params[:3].detach()
        else:
            # Use viewpoint if camera_params not available
            camera_pos = torch.tensor(planner.viewpoint[:3], dtype=torch.float32, device=device)
        
        if hasattr(self, 'target_position'):
            target_pos = torch.tensor(
                self.target_position, 
                dtype=torch.float32, 
                device=device
            )
        elif hasattr(planner, 'target_params'):
            target_pos = planner.target_params
        else:
            # Fallback
            target_pos = torch.zeros(3, dtype=torch.float32, device=device)
        
        if loss is None:
           loss, _ = planner.voxel_grid.compute_gain(camera_pos, target_pos)
           loss = loss.detach().cpu().numpy()
        print("Loss: plot ", loss)
            
        # Initialize history if it doesn't exist
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {
                'occupied_voxels': [],
                'free_voxels': [],
                'unknown_voxels': [],
                'coverage_percentage': [],
                'loss_values': [],
                'viewpoints': [camera_pos.cpu().numpy()],
                'viewpoints_target': [target_pos.cpu().numpy() if torch.is_tensor(target_pos) else target_pos]
            }
        
        # Store current metrics at the current index
        curr_viewpoint = camera_pos.cpu().numpy()
        curr_target = target_pos.cpu().numpy() if torch.is_tensor(target_pos) else target_pos
        self.metrics_history['occupied_voxels'].append(occupied_voxels)
        self.metrics_history['free_voxels'].append(free_voxels)
        self.metrics_history['unknown_voxels'].append(unknown_voxels)
        self.metrics_history['coverage_percentage'].append(coverage_percentage)
        self.metrics_history['loss_values'].append(loss)
        self.metrics_history['viewpoints'].append(curr_viewpoint)
        self.metrics_history['viewpoints_target'].append(curr_target)
        
        # Create the plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Get the data in correct order (oldest first)
        x = list(range(self.metrics_history['occupied_voxels'].__len__()))

        # Plot 1: Voxel counts and coverage
        ax = axs[0, 0]
        ax.plot(x, self.metrics_history['occupied_voxels'], 'r-o', label='Occupied Voxels')
        ax2 = ax.twinx()
        ax2.plot(x, self.metrics_history['free_voxels'], 'g-o', label='Free Voxels')
        ax.set_title('Voxel Occupancy')
        ax.set_xlabel('Viewpoint Number')
        ax.set_ylabel('Voxel Count, occ', color='r')
        ax2.set_ylabel('Voxel Count, free', color='g')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Information gain (negative loss)
        ax = axs[0, 1]
        ax.plot(x, [-l for l in self.metrics_history['loss_values']], 'g-o')
        ax.set_title('Information Gain (Negative Loss)')
        ax.set_xlabel('Viewpoint Number')
        ax.set_ylabel('Gain')
        ax.grid(True)
        
        # Plot 3: Current viewpoint statistics
        ax = axs[1, 0]
        
        # Format positions for display
        camera_pos_str = f"({camera_pos[0].item():.2f}, {camera_pos[1].item():.2f}, {camera_pos[2].item():.2f})"
        
        if torch.is_tensor(target_pos):
            target_pos_str = f"({target_pos[0].item():.2f}, {target_pos[1].item():.2f}, {target_pos[2].item():.2f})"
        else:
            target_pos_str = f"({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})"
        
        stats = [
            f'Total voxels: {total_voxels}',
            f'Occupied voxels: {occupied_voxels} ({occupied_voxels/total_voxels*100:.2f}%)',
            f'Free voxels: {free_voxels} ({free_voxels/total_voxels*100:.2f}%)',
            f'Uncertain voxels (occ): {uncertain_occ_voxels} ({uncertain_occ_voxels/total_voxels*100:.2f}%)',
            f'Target coverage: {coverage_percentage:.2f}%',
            f'Target observed: {observed_target_voxels} / {target_total_voxels}',
            f'Current loss: {loss.item():.6f}',
            f'Camera pos: {camera_pos_str}',
            f'Target pos: {target_pos_str}'
        ]
        ax.axis('off')
        ax.text(0.05, 0.95, '\n'.join(stats), transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: the coverage percentage
        ax = axs[1, 1]
        ax.plot(x, self.metrics_history['coverage_percentage'], 'r-o', label='Target Coverage (%)')
        ax.set_title('Target Coverage')
        ax.set_ylabel('Coverage (%)', color='r')
        ax.set_ylim(0, 100)
        plt.tight_layout()
        if N in [10, 25, 50, 100, 250]:
            plt.show()
        else:
            plt.close()