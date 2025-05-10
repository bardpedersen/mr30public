#pragma once

#include "rosbag_reader.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_points/types/point_cloud_gpu.hpp>
#include <gtsam_points/util/covariance_estimation.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/nearest_neighbor_search.hpp>
#include <gtsam_points/factors/integrated_icp_factor.hpp>
#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/factors/integrated_colored_gicp_factor.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>  
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>

struct ICPConfig {
    int type;
    float max_correspondence_distance;
    int num_threads;
    int k_neighbors;
    double randomsampling_rate;
    bool sequential;
};

struct FilterConfig {
    bool enabled;
    float far_threshold;
    float near_threshold;
};

struct VisualizerConfig {
    bool enabled;
    float point_size;
};

class PointCloudStitcher {
public:
    PointCloudStitcher(bool verbose);
    void stitchPointClouds(const std::vector<FrameData>& frame_data, 
                           const std::string& output_path,
                           bool use_pose,
                           bool use_between_factor,
                           const FilterConfig& filter_config,
                           const ICPConfig& icp_config,
                           const VisualizerConfig& visualizer_config);

private:

    gtsam_points::LevenbergMarquardtExtParams lm_params_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;

    void optimizeFrames(const std::vector<FrameData>& frame_data, 
                        const std::string& output_path,
                        bool use_pose,
                        bool use_between_factor,
                        const FilterConfig& filter_config,
                        const ICPConfig& icp_config,
                        const VisualizerConfig& visualizer_config);
    
    void addICPFactor(const std::shared_ptr<gtsam_points::PointCloudCPU>& prev_frame,
                        const std::shared_ptr<gtsam_points::PointCloudCPU>& current_frame,
                        const ICPConfig& icp_config,
                        gtsam::NonlinearFactorGraph& graph,
                        size_t frame_idx,
                        size_t prev_idx);

    void saveResults(const gtsam::Values& values, 
                     const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& filtered_point_clouds,
                     const std::string& output_path,
                     bool stitched,
                     bool colored);
                     
    void visualizePointClouds(const gtsam::Values& values, 
                            const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& filtered_point_clouds,
                            const VisualizerConfig& visualizer_config,
                            bool stiched);

    std::vector<Eigen::Vector4f> generateColors(int count);

    gtsam::Pose3 convertTransformToEigen(const geometry_msgs::Transform& transform);
    
    void filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_cloud,
                        const FilterConfig& filter_config);
};