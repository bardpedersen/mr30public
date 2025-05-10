import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import transform_from_rotation_translation


class FrontierNBVPlanner:
    """
    Class to plan next best view using frontier exploration
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
        num_features: int = 5,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
        camera_bounds_in: np.array = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        target_bounds_in: np.array = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        inner_bounds_in = None,
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.camera_bounds_in = camera_bounds_in
        self.target_bounds_in = target_bounds_in
        self.start_params(start_pose, target_params)
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
            num_features=num_features, # ROIs, occ_prob, sem_conf, sem_class, frontier
            target_params=self.target_params,
            device=self.device,
        )
        self.voxel_grid.voxel_grid[..., 4] = 0   # Initialize frontier status to 0 (not frontier)
        self.num_samples = num_samples
        self.inner_bounds = None
        if inner_bounds_in is not None:
            self.inner_bounds = [[target_params[0] - inner_bounds_in[0],
            target_params[1] - inner_bounds_in[2],
            target_params[2] - inner_bounds_in[4]],
            [target_params[0] + inner_bounds_in[1],
            target_params[1] + inner_bounds_in[3],
            target_params[2] + inner_bounds_in[5]]]
        self.view_sampler = ViewpointSampler(num_samples, inner_bounds_in=self.inner_bounds)
        self.viewpoint = start_pose
        self.rviz_visualizer = RvizVisualizer()

    def start_params(self, start_pose: np.array, target_params: np.array) -> None:
        """
        Initialize the parameters for random sampling
        """
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

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H, W)
        :param semantics: confidence scores and class ids (H, W, 2)
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
        self.voxel_grid.update_frontier_status()
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage

    def next_best_view(self, N) -> Tuple[np.array, float, int]:
        """
        Sample a new viewpoint, based on the one with most information gain or highest coverage
        """
        view_samples = self.view_sampler.sample_five_dof(
            camera_limits=self.camera_bounds[:, :3].cpu().numpy(),
            target_limits=self.camera_bounds[:, 3:].cpu().numpy(),
        )
        self.rviz_visualizer.visualize_view_samples(
            numpy_to_pose_array(view_samples[:, :7])
        )
        # calculate the gain for each viewpoint
        number_front = []
        number_front_image_list = []
        
        for i in range(view_samples.shape[0]):
            viewpoint = view_samples[i]
            camera_position = torch.tensor(
                viewpoint[:3], 
                dtype=torch.float32, 
                device=self.device
            )
            visible_count, vis_data = self.voxel_grid.compute_frontier_visibility(
                camera_position, self.target_params
            )
            visible_count = visible_count.detach().cpu().numpy()
            number_front.append(visible_count)
            number_front_image_list.append(vis_data)

        # select the viewpoint with the highest number of frontiers
        index = np.argmax(number_front)
        self.viewpoint = view_samples[index]
        vis_data = number_front_image_list[index]
        self.rviz_visualizer.visualize_gain_image(vis_data)
        self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(self.viewpoint))
        return self.viewpoint, number_front[index], self.num_samples

    def visualize(self):
        """
        Visualize the voxel grid as a point cloud in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        frontier_points = self.voxel_grid.get_frontier_points()
        frontier_points = frontier_points.cpu().numpy()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        self.rviz_visualizer.visualize_frontiers(frontier_points)
        camera_bounds_np = self.camera_bounds[:, :3].cpu().numpy()
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds_np)
        if hasattr(self, 'inner_bounds') and self.inner_bounds is not None:
            self.rviz_visualizer.visualize_inner_bounds(self.inner_bounds)
