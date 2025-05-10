import os

# Set required environment variable for deterministic CUDA operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import numpy as np
import random

"""
# Set seeds for reproducibility
torch.manual_seed(0)        # PyTorch RNG
np.random.seed(0)           # NumPy RNG
random.seed(0)              # Python's built-in RNG

# Configure PyTorch for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True,warn_only=True)
"""

import torch.nn as nn
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from scene_representation.voxel_grid import VoxelGrid

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation


class GradientNBVPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint using gradient-based optimization
    """

    def __init__(
        self,
        start_pose: np.array,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.003]),
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
        num_samples: int = 1, # nuumber of optimization steps
        camera_bounds_in: np.array = np.array([0.2, 0.4, 0.2, 0.2, 0.15, 0.3]),
        target_bounds_in: np.array = np.array([0.2, 0.2, 0.3, 0.2, 0.2, 0.2]),
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
        inner_bounds_in: np.array = None,
    ) -> None:
        """
        Initialize the planner
        :param grid_size: size of the voxel grid in meters
        :param voxel_size: size of the voxels in meters
        :param grid_center: center of the voxel grid in meters
        :param image_size: size of the image in pixels
        :param num_pts_per_ray: number of points sampled per ray
        :param num_features: number of features per voxel
        """
        super(GradientNBVPlanner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.camera_bounds_in = camera_bounds_in
        self.target_bounds_in = target_bounds_in
        self.inner_bounds_in = inner_bounds_in
        self.start_pose = start_pose
        self.prev_loss = 0
        self.optimization_params(start_pose, target_params)
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
        self.num_samples = num_samples
        self.rviz_visualizer = RvizVisualizer()

    def optimization_params(
        self, start_pose: np.array, target_params: np.array
    ) -> None:
        """
        Initialize the optimization parameters
        """
        self.camera_params = nn.Parameter(
            torch.tensor(
                [
                    start_pose[0],
                    start_pose[1],
                    start_pose[2],
                    target_params[0],
                    target_params[1],
                    target_params[2],
                ],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
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
                    target_params[0] - self.target_bounds_in[0],
                    target_params[1] - self.target_bounds_in[2],
                    target_params[2] - self.target_bounds_in[4],
                ],
                [
                    start_pose[0] + self.camera_bounds_in[1],
                    start_pose[1] + self.camera_bounds_in[3],
                    start_pose[2] + self.camera_bounds_in[5],
                    target_params[0] + self.target_bounds_in[1],
                    target_params[1] + self.target_bounds_in[3],
                    target_params[2] + self.target_bounds_in[5],
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Set inner bounds (if provided)
        if self.inner_bounds_in is not None:
            self.inner_bounds = torch.tensor(
                [
                    [
                        target_params[0] - self.inner_bounds_in[0],
                        target_params[1] - self.inner_bounds_in[2],
                        target_params[2] - self.inner_bounds_in[4],
                    ],
                    [
                        target_params[0] + self.inner_bounds_in[1],
                        target_params[1] + self.inner_bounds_in[3],
                        target_params[2] + self.inner_bounds_in[5],
                    ],
                ],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.inner_bounds = None

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.03) #self.parameters() = self.camera_params

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H x W)
        :param semantics: confidence scores and class ids (H x W x 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
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

    def loss(self, target_pos: np.array) -> torch.tensor:
        """
        Compute the loss for the current viewpoint
        :return: loss
        """
        if target_pos is not None:
            self.target_params = torch.tensor(
                target_pos, dtype=torch.float32, device=self.device
            )
        else:
            self.target_params = self.camera_params[3:]
        loss, gain_image = self.voxel_grid.compute_gain(
            self.camera_params[:3], self.target_params
        )
        return loss, gain_image

    def enforce_bounds(self, params):
        """Enforce both inner and outer bounds on parameters"""
        # First clamp to outer bounds
        params = torch.clamp(params, self.camera_bounds[0], self.camera_bounds[1])
        
        # Then enforce inner bounds if they exist
        if self.inner_bounds is not None:
            camera_pos = params[:3]
            
            # Check if camera is inside the inner bounds
            inside_x = (camera_pos[0] > self.inner_bounds[0, 0]) & (camera_pos[0] < self.inner_bounds[1, 0])
            inside_y = (camera_pos[1] > self.inner_bounds[0, 1]) & (camera_pos[1] < self.inner_bounds[1, 1])
            inside_z = (camera_pos[2] > self.inner_bounds[0, 2]) & (camera_pos[2] < self.inner_bounds[1, 2])
            
            if inside_x & inside_y & inside_z:
                # Camera is inside the keep-out zone, project it to the nearest face
                distances = torch.zeros(6, device=self.device)
                distances[0] = camera_pos[0] - self.inner_bounds[0, 0]  # distance to -x face
                distances[1] = self.inner_bounds[1, 0] - camera_pos[0]  # distance to +x face
                distances[2] = camera_pos[1] - self.inner_bounds[0, 1]  # distance to -y face
                distances[3] = self.inner_bounds[1, 1] - camera_pos[1]  # distance to +y face
                distances[4] = camera_pos[2] - self.inner_bounds[0, 2]  # distance to -z face
                distances[5] = self.inner_bounds[1, 2] - camera_pos[2]  # distance to +z face
                
                # Find nearest face
                min_dist, min_idx = torch.min(distances, dim=0)
                
                # Project to the nearest face
                if min_idx == 0:
                    camera_pos[0] = self.inner_bounds[0, 0]
                elif min_idx == 1:
                    camera_pos[0] = self.inner_bounds[1, 0]
                elif min_idx == 2:
                    camera_pos[1] = self.inner_bounds[0, 1]
                elif min_idx == 3:
                    camera_pos[1] = self.inner_bounds[1, 1]
                elif min_idx == 4:
                    camera_pos[2] = self.inner_bounds[0, 2]
                elif min_idx == 5:
                    camera_pos[2] = self.inner_bounds[1, 2]
                
                # Update params with projected position
                params[:3] = camera_pos
                
        return params

    def next_best_view(self, target_pos=None, N=1) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of samples
        """
        #self.plot_optimization_landscape()
        #self.plot_optimization_landscape(camera=False)
        for _ in range(self.num_samples):
            self.optimizer.zero_grad()
            loss, gain_image = self.loss(target_pos)
            loss.backward()
            self.optimizer.step()
            self.camera_params.data = self.enforce_bounds(self.camera_params.data)

        viewpoint = self.get_viewpoint()
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_gain_image(gain_image)
        loss = loss.detach().cpu().numpy()
        return viewpoint, loss, self.num_samples

    def next_best_view_with_camera_reset(self, target_pos=None, N=1) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of samples
        """
        #self.plot_optimization_landscape()
        #self.plot_optimization_landscape(camera=False)
        for _ in range(self.num_samples):
            self.optimizer.zero_grad()
            loss, gain_image = self.loss(target_pos)
            loss.backward()
            self.optimizer.step()
            self.camera_params.data = self.enforce_bounds(self.camera_params.data)

        viewpoint = self.get_viewpoint()
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_gain_image(gain_image)
        loss = loss.detach().cpu().numpy()

        if loss - self.prev_loss < 0.1:
            print("Resetting camera position")
            target_params = self.camera_params[3:].clone().detach()
            
            # Generate random position within a sphere around the target
            # Start with a random direction vector
            rand_dir = torch.randn(3, device=self.device)
            rand_dir = rand_dir / torch.norm(rand_dir)  # Normalize to unit vector
            
            # Choose random distance between 0.3 and 0.6 meters
            distance = 0.3 + 0.3 * torch.rand(1, device=self.device).item()
            
            # Calculate new camera position
            target_pos = torch.tensor([
                target_params[0].item(),
                target_params[1].item(), 
                target_params[2].item()
            ], device=self.device)
            
            new_camera_pos = target_pos + rand_dir * distance
            
            # Update camera parameters
            self.camera_params.data = torch.tensor(
                [
                    new_camera_pos[0].item(),
                    new_camera_pos[1].item(), 
                    new_camera_pos[2].item(),
                    target_params[0].item(),
                    target_params[1].item(),
                    target_params[2].item()
                ], 
                dtype=torch.float32, 
                device=self.device
            )
            self.prev_loss = 0
        else:
            self.prev_loss = loss

        return viewpoint, loss, self.num_samples

    def initialize_pso_particles(self, num_particles=10):
        """
        Initialize particles for PSO optimization. This should be called once.
        
        :param num_particles: Number of particles to initialize
        """
        # Store current camera parameters
        original_camera_params = self.camera_params.clone().detach()
        
        # Initialize particles
        self.pso_particles = []
        
        # First particle at current position
        self.pso_particles.append(original_camera_params.clone())
        
        # Get camera bounds for uniform distribution
        target_params = original_camera_params[3:].clone().detach()    
        camera_min_bounds = self.camera_bounds[0, :3]  # Min x,y,z for camera position
        camera_max_bounds = self.camera_bounds[1, :3]  # Max x,y,z for camera position
        # Initialize remaining particles with uniform random positions across the entire camera bounds
        for _ in range(1, num_particles):
            # Generate random position uniformly within camera bounds
            random_camera_pos = torch.zeros(3, device=self.device)
            for j in range(3):  # For each x,y,z dimension
                random_camera_pos[j] = camera_min_bounds[j] + (camera_max_bounds[j] - camera_min_bounds[j]) * torch.rand(1, device=self.device)
            
            # Create particle with random camera position but fixed target
            particle = torch.cat([random_camera_pos, target_params])
            self.pso_particles.append(particle)
        
        # Restore original camera parameters
        self.camera_params.data = original_camera_params

        # Set flag indicating particles have been initialized
        self.pso_initialized = True
        self.num_pso_particles = num_particles
        
    def next_best_view_pso(self, target_pos=None, N=1) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint using particle swarm optimization with gradient guidance.
        Each particle uses gradient-based movement while being influenced by the global best solution.
        
        :param target_pos: Optional target position to look at
        :param N: Number of optimization steps
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of iterations performed
        """        
        # Track iteration number for history
        N -= 1
        if not hasattr(self, 'last_pso_N_value'):
            self.last_pso_N_value = 9999
        print(f"PSO iteration {N}, last iteration {self.last_pso_N_value}")
        
        # Initialize particles if not done already
        if not hasattr(self, 'pso_initialized') or not self.pso_initialized:
            self.initialize_pso_particles(20)  # Default 10 particles

        # Initialize PSO history for plotting
        if not hasattr(self, 'pso_history'):
            self.pso_history = {
                'iterations': [],
                'particle_losses': [],
                'best_losses': [],
                'best_particle_ids': []
            }
            self.particle_trajectories = {}  # Dictionary to store trajectory for each particle
        
        # Update target parameters if specified
        if target_pos is not None:
            target_params = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
            for i in range(len(self.pso_particles)):
                # Update only target position, keep camera position
                self.pso_particles[i] = torch.cat([self.pso_particles[i][:3], target_params])
            if 'global_best' in locals():
                global_best = torch.cat([global_best[:3], target_params])
        
        # Run gradient-based optimization for particles
        global_best_loss = float('inf')
        global_best = None
        global_best_gain_image = None
        particle_losses = []
        particle_id = 0
        
        # Process each particle
        for i, particle in enumerate(self.pso_particles):
            print(f"Processing particle {i+1}/{self.num_pso_particles}")
            # Set particle as current camera position
            self.camera_params.data = particle

            traj_id = f"particle_{i+1}"
            if traj_id not in self.particle_trajectories:
                self.particle_trajectories[traj_id] = []
            best_loss = float('inf')
            # Multiple gradient steps per PSO iteration
            for _ in range(self.num_samples):
                self.optimizer.zero_grad()
                loss, gain_image = self.loss(target_pos)
                loss.backward()
                self.optimizer.step()
                self.camera_params.data = self.enforce_bounds(self.camera_params.data)
                current_loss_value = loss.detach().cpu().numpy()
                current_gain_image = gain_image

                if current_loss_value < best_loss:
                    best_loss = current_loss_value
                    
                if current_loss_value < global_best_loss:
                    global_best_loss = current_loss_value
                    global_best = self.camera_params.clone().detach()
                    global_best_gain_image = current_gain_image
                    particle_id = i+1

            # Store final loss for this particle
            particle_losses.append(best_loss)
            pos = particle[:3].detach().cpu().numpy()
            self.particle_trajectories[traj_id].append(pos)
            print(f"Particle {i+1} loss: {best_loss:.4f}")
            
            # Update particle position after gradient steps
            self.pso_particles[i] = self.camera_params.clone().detach()
        print(f"Global best loss: {global_best_loss:.4f} at particle {particle_id}")
        
        viewpoints = []
        for particle in self.pso_particles:
            camera_pos = particle[:3]
            target_pos = particle[3:]
            # Calculate orientation quaternion from camera to target
            quat = look_at_rotation(camera_pos, target_pos)
            quat = quat.detach().cpu().numpy()
            # Combine position and orientation
            viewpoint = np.zeros(7)
            viewpoint[:3] = particle[:3].detach().cpu().numpy()
            viewpoint[3:] = quat
            viewpoints.append(viewpoint)

        # Update PSO history
        if N == self.last_pso_N_value:
            self.pso_history['iterations'][N] = N
            self.pso_history['particle_losses'][N] = particle_losses
            self.pso_history['best_losses'][N] = global_best_loss
            self.pso_history['best_particle_ids'][N] = particle_id
        else:
            self.pso_history['iterations'].append(N)
            self.pso_history['particle_losses'].append(particle_losses)
            self.pso_history['best_losses'].append(global_best_loss)
            self.pso_history['best_particle_ids'].append(particle_id)

        # Set the best parameters found
        self.camera_params.data = global_best
        viewpoint = self.get_viewpoint()
        
        # Visualize the selected viewpoint and gain
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_gain_image(global_best_gain_image)
        
        # Plot metrics at certain intervals
        if N in [10, 25, 50]:
            self.plot_pso_metrics()
        
        self.last_pso_N_value = N
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_view_samples(numpy_to_pose_array(viewpoints))
        self.visualize_particle_trajectories()
        return viewpoint, global_best_loss, self.num_pso_particles * self.num_samples
    
    def visualize_particle_trajectories(self):
        """
        Visualize the trajectories of all particles in the PSO algorithm
        """
        if not hasattr(self, 'particle_trajectories') or not self.particle_trajectories:
            print("No particle trajectories available")
            return

        # Create unique marker for each particle's path
        for particle_id, trajectory in self.particle_trajectories.items():
            if not trajectory:
                continue
            
            # Publish to a unique topic for this particle
            topic = f"/viewpoint_planning/particle_path_{particle_id}"
            
            # Use RViz visualizer to publish this path
            self.rviz_visualizer.publish_pose_array(topic, trajectory)

    def plot_pso_metrics(self):
        """
        Plot metrics from the particle swarm optimization
        """
        fig = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(1, 2, figure=fig)
        
        # Plot 1: Particle losses across iterations
        ax1 = fig.add_subplot(gs[0, 0])
        if hasattr(self, 'pso_history') and len(self.pso_history['iterations']) > 1:
            # Create particle loss plot
            iterations = self.pso_history['iterations']
            
            # For each particle, plot its loss across iterations
            num_particles = len(self.pso_history['particle_losses'][0])
            for p in range(num_particles):
                particle_loss_values = [losses[p] if p < len(losses) else float('nan') 
                                    for losses in self.pso_history['particle_losses']]
                ax1.plot(iterations, particle_loss_values, 'o-', alpha=0.5, 
                        label=f'Particle {p+1}')
            
            # Plot best loss across iterations
            best_losses = self.pso_history['best_losses']
            ax1.plot(iterations, best_losses, 'r-*', linewidth=2, label='Best Loss')
            
            ax1.set_title('Particle Losses Across Iterations')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "Not enough history data yet", 
                    horizontalalignment='center', verticalalignment='center')
            ax1.set_title('PSO Particle History')
            ax1.axis('off')
        
        # Plot 2: Best particle IDs and their losses
        ax2 = fig.add_subplot(gs[0, 1])
        if hasattr(self, 'pso_history') and len(self.pso_history['iterations']) > 1:
            iterations = self.pso_history['iterations']
            best_losses = self.pso_history['best_losses']
            best_particle_ids = self.pso_history['best_particle_ids']
            
            # Create bar chart of which particle was selected at each iteration
            ax2.bar(iterations, best_particle_ids, alpha=0.7, color='skyblue')
            
            # Add loss trend line on secondary y-axis
            ax2_twin = ax2.twinx()
            ax2_twin.plot(iterations, best_losses, 'r-o', label='Best Loss')
            
            ax2.set_title('Best Particle ID and Loss per Iteration')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Particle ID')
            ax2_twin.set_ylabel('Loss', color='r')
            
            # Combine legends
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2_twin.legend(loc='upper right')
            
            ax2.set_ylim(0.5, self.num_pso_particles + 0.5)  # Set y-limits for bar chart
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Not enough history data yet", 
                    horizontalalignment='center', verticalalignment='center')
            ax2.set_title('Best Particle History')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def next_best_view_suboptimal(self, target_pos=None, N=1, initial_temp=0.7, 
                                    min_temp=0.01, r_min=0.03, r_max=0.15, plot_metrics=True) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint using suboptimal optimization,
        occasionally accepting worse solutions to escape local minima
        
        :param target_pos: Optional target position to look at
        :param N: Number of optimization steps
        :param initial_temp: Initial temperature for simulated annealing-like approach
        :param min_temp: Minimum temperature threshold
        :param r_min: Minimum radius for random shell sampling
        :param r_max: Maximum radius for random shell sampling
        :param plot_metrics: Whether to plot optimization metrics after completion
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of samples
        """
        N -= 1
        if not hasattr(self, 'last_N_value'):
            self.last_N_value = 9999
        print(N, self.last_N_value)
        # Initialize temperature if not already set
        if not hasattr(self, 'current_temp') or self.current_temp is None:
            self.current_temp = initial_temp
            self.iteration_count = N
            
            # Initialize SA history for plotting
            if not hasattr(self, 'sa_history'):
                self.sa_history = {
                    'iterations': [],
                    'temperatures': [],
                    'acceptance_rates': [],
                    'grad_losses': [],
                    'noise_losses': [],
                    'selected_losses': [],
                    'improvements': [],
                    'accept_prob': [],
                }
        else:
            self.iteration_count = N

        #self.plot_optimization_landscape()
        #self.plot_optimization_landscape(camera=False)
        # Run optimization with simulated annealing acceptance
        noise_selections = 0
        gradient_selections = 0
        self.num_samples = 1
        for step in range(self.num_samples):
            # Run gradient step to get candidate 1
            self.optimizer.zero_grad()
            prev_loss, _ = self.loss(target_pos)
            prev_loss.backward()
            self.optimizer.step()
            self.camera_params.data = self.enforce_bounds(self.camera_params.data)
            
            # Evaluate gradient-based solution
            grad_loss, grad_gain_image = self.loss(target_pos)
            grad_loss_value = grad_loss.detach().cpu().numpy()
            grad_params = self.camera_params.clone().detach()
            
            # Store gradient position for visualization
            grad_camera_pos = grad_params[:3].clone().detach().cpu().numpy()
            
            # Generate random direction in spherical shell
            camera_pos = grad_params[:3]  # Extract current camera position
            target_pos_tensor = grad_params[3:]  # Keep target position the same
            direction = torch.randn(3, device=self.device)
            direction = direction / torch.norm(direction)

            # For uniform distribution in a shell, we need cube root sampling
            u = torch.rand(1, device=self.device).item()
            radius = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1/3)

            # Calculate new camera position
            new_camera_pos = camera_pos + direction * radius

            # Create new parameter tensor with updated camera position but same target
            noise_params = torch.cat([new_camera_pos, target_pos_tensor])
            self.camera_params.data = noise_params
            self.camera_params.data = self.enforce_bounds(self.camera_params.data)
            
            # Evaluate noisy solution
            noise_loss, noise_gain_image = self.loss(target_pos)
            noise_loss_value = noise_loss.detach().cpu().numpy()
            
            # Store noise position for visualization
            noise_camera_pos = self.camera_params[:3].clone().detach().cpu().numpy()
            
            # Check if noise improved over gradient
            is_improvement = noise_loss_value < grad_loss_value
            
            # Accept worse solutions with probability based on temperature
            delta = grad_loss_value - noise_loss_value  # negative when noise is worse
            accept_prob = np.exp(delta / self.current_temp)
            
            # Make selection decision
            rand_val = np.random.random()
            if rand_val < accept_prob:
                # Accept noise solution
                selected_params = self.camera_params.clone().detach()
                selected_loss = noise_loss_value
                selected_gain_image = noise_gain_image
                if N != self.last_N_value:
                    noise_selections += 1
            else:
                # Keep gradient-based solution
                selected_params = grad_params
                selected_loss = grad_loss_value
                selected_gain_image = grad_gain_image
            
            # Update camera params to selected solution
            self.camera_params.data = selected_params
    
        # Update global SA history
        if N == self.last_N_value:
            self.sa_history['iterations'][N] = self.iteration_count
            self.sa_history['temperatures'][N] = self.current_temp
            self.sa_history['acceptance_rates'][N] = 1 if rand_val < accept_prob else 0
            self.sa_history['grad_losses'][N] = grad_loss_value
            self.sa_history['noise_losses'][N] = noise_loss_value
            self.sa_history['accept_prob'][N] = accept_prob
            self.sa_history['selected_losses'][N] = selected_loss
            self.sa_history['improvements'][N] = np.mean(is_improvement)
        else:
            self.sa_history['iterations'].append(self.iteration_count)
            self.sa_history['temperatures'].append(self.current_temp)
            self.sa_history['acceptance_rates'].append(1 if rand_val < accept_prob else 0)
            self.sa_history['grad_losses'].append(grad_loss_value)
            self.sa_history['noise_losses'].append(noise_loss_value)
            self.sa_history['accept_prob'].append(accept_prob)
            self.sa_history['selected_losses'].append(selected_loss)
            self.sa_history['improvements'].append(np.mean(is_improvement))
        
        # Set final parameters to selected solution
        self.camera_params.data = selected_params
    
        # Logarithmic cooling schedule: T = T0/log_2(k+1)
        self.current_temp = initial_temp / np.log2(self.iteration_count + 2)  # Add 2 to avoid division by zero
        self.current_temp = max(self.current_temp, min_temp)  # Don't go below min temp
        
        # Get final viewpoint
        viewpoint = self.get_viewpoint()
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(viewpoint))
        self.rviz_visualizer.visualize_gain_image(selected_gain_image)
        
        # Plot metrics if requested
        if N in [10, 25, 50]:
            self.plot_sa_metrics()
        
        self.last_N_value = N
        return viewpoint, selected_loss, self.num_samples

    def plot_sa_metrics(self):
        """
        Plot metrics from the simulated annealing optimization
        
        :param run_metrics: Dictionary containing metrics from the current run
        """
        fig = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(1, 2, figure=fig)
        
        # Plot 1: Global SA history (temperature and acceptance rate)
        ax1 = fig.add_subplot(gs[0, 0])
        if hasattr(self, 'sa_history') and len(self.sa_history['iterations']) > 1:
            ax1.plot(self.sa_history['iterations'], self.sa_history['temperatures'], 'r-o', label='Temperature')
            ax1_2 = ax1.twinx()
            ax1_2.plot(self.sa_history['iterations'], self.sa_history['acceptance_rates'], 'g-o', label='Noise Acceptance Rate')
            #ax1_2.plot(self.sa_history['iterations'], self.sa_history['improvements'], 'b--', label='Improvement Rate')
            ax1.set_title('SA Temperature Schedule & Acceptance')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Temperature', color='r')
            ax1_2.set_ylabel('Rate', color='g')
            ax1_2.set_ylim(0, 1.05)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "Not enough history data yet", 
                    horizontalalignment='center', verticalalignment='center')
            ax1.set_title('SA Global History')
            ax1.axis('off')
        
        # Plot 2: Global loss history with current loss values (not averages)
        ax2 = fig.add_subplot(gs[0, 1])
        if hasattr(self, 'sa_history') and len(self.sa_history['iterations']) > 1:
            # Use selected losses directly instead of averages
            ax2.plot(self.sa_history['iterations'], self.sa_history['selected_losses'], 'r-o', label='Selected Loss')
            ax2.plot(self.sa_history['iterations'], self.sa_history['grad_losses'],'b-o', label='Current Gradient Loss')
            ax2.plot(self.sa_history['iterations'], self.sa_history['noise_losses'],'g-o', label='Current Noise Loss')
            ax2.plot(self.sa_history['iterations'], self.sa_history['accept_prob'], 'y--', label='Acceptance Probability')

            ax2.set_title('Loss History Across Iterations')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Not enough history data yet", 
                    horizontalalignment='center', verticalalignment='center')
            ax2.set_title('Loss History')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
            
    def get_viewpoint(self) -> np.array:
        """
        Get the current viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        quat = look_at_rotation(self.camera_params[:3], self.camera_params[3:])
        quat = quat.detach().cpu().numpy()
        viewpoint = np.zeros(7)
        viewpoint[:3] = self.camera_params.detach().cpu().numpy()[:3]
        viewpoint[3:] = quat
        return viewpoint

    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        return voxel_points, sem_conf_scores, sem_class_ids

    def visualize(self):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        # Visualize target
        # target = self.target_params.detach().cpu().numpy()
        target = self.camera_params.detach().cpu().numpy()[3:]
        rois = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        # Visualize camera bounds
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
        if hasattr(self, 'inner_bounds') and self.inner_bounds is not None:
            inner_bounds = self.inner_bounds.cpu().numpy()
            # Make sure your RvizVisualizer has this method
            self.rviz_visualizer.visualize_inner_bounds(inner_bounds)

    def plot_optimization_landscape(self, resolution=20, plot_range=0.5, alpha=0.7, vmin=-16, vmax=0, camera=True):
        """
        Visualize how the gradient-based optimization actually progresses through the information gain landscape,
        showing the historical trajectory instead of simulated steps.
        """        
        # Store the original camera parameters to restore later
        original_params = self.camera_params.detach().clone()
        
        # Get current camera position and target
        if camera:
            camera_center = original_params[:3].detach().clone().cpu()

            # Get historical trajectory if it exists
            if hasattr(self, 'metrics_history') and 'viewpoints' in self.metrics_history and len(self.metrics_history['viewpoints']) > 0:
                traj_positions = np.array(self.metrics_history['viewpoints'])
                # Add current position if not already in history
                if len(traj_positions) == 0 or not np.array_equal(traj_positions[-1], camera_center.numpy()):
                    curr_pos = camera_center.numpy()
                    traj_positions = np.vstack([traj_positions, curr_pos])
            else:
                # Only use current position
                traj_positions = np.array([camera_center.numpy()])

        else:
            camera_center = self.target_params.detach().clone().cpu()

            # Get historical trajectory if it exists
            if hasattr(self, 'metrics_history') and 'viewpoints_target' in self.metrics_history and len(self.metrics_history['viewpoints_target']) > 0:
                traj_positions = np.array(self.metrics_history['viewpoints_target'])
                # Add current position if not already in history
                if len(traj_positions) == 0 or not np.array_equal(traj_positions[-1], camera_center.numpy()):
                    curr_pos = camera_center.numpy()
                    traj_positions = np.vstack([traj_positions, curr_pos])
            else:
                # Only use current position
                traj_positions = np.array([camera_center.numpy()])


        # Get the starting position
        midpoint = traj_positions[0]
        
        # Create coordinate ranges centered on the midpoint
        x = np.linspace(midpoint[0] - plot_range, midpoint[0] + plot_range, resolution)
        y = np.linspace(midpoint[1] - plot_range, midpoint[1] + plot_range, resolution)
        z = np.linspace(midpoint[2] - plot_range, midpoint[2] + plot_range, resolution)
        
        # Create 2D slices for visualization
        xy_gains = np.zeros((resolution, resolution))
        xz_gains = np.zeros((resolution, resolution))
        yz_gains = np.zeros((resolution, resolution))
        
        # Calculate gains for each slice
        print("Computing XY plane...")
        for i, xi in tqdm(enumerate(x), total=resolution, desc="XY Plane"):
            for j, yj in enumerate(y):
                camera_pos = torch.tensor([xi, yj, camera_center[2]], device=self.device)
                loss, _ = self.voxel_grid.compute_gain(camera_pos, self.target_params)
                xy_gains[i, j] = -loss.item()  # Convert loss to gain
        
        print("Computing XZ plane...")
        for i, xi in tqdm(enumerate(x), total=resolution, desc="XZ Plane"):
            for k, zk in enumerate(z):
                camera_pos = torch.tensor([xi, camera_center[1], zk], device=self.device)
                loss, _ = self.voxel_grid.compute_gain(camera_pos, self.target_params)
                xz_gains[i, k] = -loss.item()
        
        print("Computing YZ plane...")
        for j, yj in tqdm(enumerate(y), total=resolution, desc="YZ Plane"):
            for k, zk in enumerate(z):
                camera_pos = torch.tensor([camera_center[0], yj, zk], device=self.device)
                loss, _ = self.voxel_grid.compute_gain(camera_pos, self.target_params)
                yz_gains[j, k] = -loss.item()
        
        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY Plane Plot
        im0 = axes[0].imshow(xy_gains.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], 
                        cmap='viridis', vmin=vmin, vmax=vmax, alpha=alpha)
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_title('XY Plane (Z={:.2f})'.format(camera_center[2]))
        
        # Plot historical trajectory in XY plane
        if len(traj_positions) > 1:
            axes[0].plot(traj_positions[:, 0], traj_positions[:, 1], 'r-', linewidth=2)

        # Scatter plot with smaller points for older positions, bigger for newer ones
        for pos in traj_positions:
            axes[0].scatter(pos[0], pos[1], color='white')
        
        # Mark current position and target
        axes[0].plot(camera_center[0], camera_center[1], 'mo', markersize=10, label='Current')
        
        # XZ Plane
        im1 = axes[1].imshow(xz_gains.T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], 
                        cmap='viridis', vmin=vmin, vmax=vmax, alpha=alpha)
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Z Position')
        axes[1].set_title('XZ Plane (Y={:.2f})'.format(camera_center[1]))
        
        # Plot historical trajectory in XZ plane
        if len(traj_positions) > 1:
            axes[1].plot(traj_positions[:, 0], traj_positions[:, 2], 'r-', linewidth=2)
        
        # Scatter plot for XZ trajectory
        for pos in traj_positions:
            axes[1].scatter(pos[0], pos[2], color='white')
            
        axes[1].plot(camera_center[0], camera_center[2], 'mo', markersize=10)
        
        # YZ Plane
        im2 = axes[2].imshow(yz_gains.T, origin='lower', extent=[y[0], y[-1], z[0], z[-1]], 
                        cmap='viridis', vmin=vmin, vmax=vmax, alpha=alpha)
        axes[2].set_xlabel('Y Position')
        axes[2].set_ylabel('Z Position')
        axes[2].set_title('YZ Plane (X={:.2f})'.format(camera_center[0]))
        
        # Plot historical trajectory in YZ plane
        if len(traj_positions) > 1:
            axes[2].plot(traj_positions[:, 1], traj_positions[:, 2], 'r-', linewidth=2)
        
        # Scatter plot for YZ trajectory
        for pos in traj_positions:
            axes[2].scatter(pos[1], pos[2], color='white')
            
        axes[2].plot(camera_center[1], camera_center[2], 'mo', markersize=10)
        
        # Add colorbars
        fig.colorbar(im0, ax=axes[0], label='Information Gain')
        fig.colorbar(im1, ax=axes[1], label='Information Gain')
        fig.colorbar(im2, ax=axes[2], label='Information Gain')
        
        # Add a legend to the first plot
        axes[0].legend(loc='upper right')
        
        # Add info text
        num_points = len(traj_positions)
        if camera:
            plt.figtext(0.5, 0.01, 
                        f"Historical trajectory showing {num_points} viewpoints\n" +
                        f"Red line shows actual path followed by the camera", 
                        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        else:
            plt.figtext(0.5, 0.01, 
                        f"Historical trajectory showing {num_points} viewpoints\n" +
                        f"Red line shows actual path followed by the target", 
                        ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.tight_layout()
        
        # Make sure we don't modify camera parameters
        self.camera_params.data = original_params
        
        print("Displaying plots...")
        plt.show()

"""
Understanding the Information Gain Plot Visualization
The plots from plot_gain_slices() show you a visual representation of the information gain landscape - essentially a map of where the camera should move to gather the most new information about the target object. Here's how to interpret what you're seeing:

The Three Plot Panels
Each visualization consists of three slices of the 3D information gain landscape:

Left panel (XY Plane): A top-down view showing horizontal movement options
Middle panel (XZ Plane): A side view showing forward/backward and up/down movement options
Right panel (YZ Plane): Another side view showing left/right and up/down movement options
Color Interpretation
The coloring uses the 'viridis' colormap:

Bright yellow/white areas: High information gain - moving the camera here would reveal a lot of new information
Dark blue/purple areas: Low information gain - moving here would reveal little new information
Green/teal areas: Medium information gain
Key Markers
Red X: Your current camera position
Red Star: The target object's position
How to Read the Plot
Identifying the Best Direction: Look for bright yellow areas - these show where the camera should move to maximize new information. The gradient-based optimizer automatically moves toward these bright regions.

Understanding Occlusions: Dark areas often indicate:

Regions where the view would be blocked by known objects
Areas that have already been well-observed (low uncertainty)
Positions with poor lines of sight to uncertain regions
Distance Effects: Notice how gain typically varies with distance from the target:

Too close: Limited field of view, may see fewer uncertain voxels
Too far: Details become less visible, lower information density
Optimal distance: Balanced view of the target and surrounding uncertainties
Example Interpretation
If you see:

A bright region above and to the left of your current position in the XY plane
The XZ plane shows higher gain at a slightly higher elevation
The YZ plane confirms the upward trend
This would indicate that moving the camera up and to the left would likely give you the best next view.

Why This Matters
This visualization shows you exactly what the gradient-based algorithm is "seeing" when it makes decisions. The optimization is simply moving the camera in the direction where the landscape gets brighter (higher gradient), allowing you to understand why it chooses particular movements.

The plot essentially reveals the mathematical objective function that drives the next-best-view algorithm.
"""