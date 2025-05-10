#include "pointcloud_stitcher.hpp"

PointCloudStitcher::PointCloudStitcher(bool verbose) {
    ROS_INFO("Verbose mode: %s", verbose ? "true" : "false");
    if (verbose) {
        lm_params_.set_verbose();
    }
    prior_noise_ = gtsam::noiseModel::Isotropic::Precision(6, 1e10);
}

void PointCloudStitcher::stitchPointClouds(const std::vector<FrameData>& frame_data, const std::string& output_path, 
                                            bool use_pose,
                                            bool use_between_factor,
                                            const FilterConfig& filter_config,
                                            const ICPConfig& icp_config,
                                            const VisualizerConfig& visualizer_config) {
    // Ros info the parameters used
    ROS_INFO("use_pose: %s", use_pose ? "true" : "false");
    if (filter_config.enabled) {
        ROS_INFO("Filtering points, removing closer than %f and further than %f", filter_config.near_threshold, filter_config.far_threshold);
    }
    if (icp_config.type == 0) {
        ROS_INFO("No ICP for optimization");
    }
    else if (icp_config.type == 1) {
        ROS_INFO("Using GICP for optimization");
        ROS_INFO("k_neighbors: %d", icp_config.k_neighbors);
        ROS_INFO("randomsampling_rate: %f", icp_config.randomsampling_rate);
    }
    else if (icp_config.type == 2) {
        ROS_INFO("Using Color GICP for optimization");
        ROS_INFO("k_neighbors: %d", icp_config.k_neighbors);
        ROS_INFO("randomsampling_rate: %f", icp_config.randomsampling_rate);
    }
    else {
        ROS_INFO("Using ICP for optimization");
    }
    ROS_INFO("num_threads: %d", icp_config.num_threads);
    ROS_INFO("max_correspondence_distance: %f", icp_config.max_correspondence_distance);
    ROS_INFO("sequential: %s", icp_config.sequential ? "true" : "false");
    ROS_INFO("Visualizer enabled: %s", visualizer_config.enabled ? "true" : "false");
    ROS_INFO("Point size: %f", visualizer_config.point_size);
    optimizeFrames(frame_data, 
                    output_path, 
                    use_pose, 
                    use_between_factor,
                    filter_config,
                    icp_config,
                    visualizer_config);
}

void PointCloudStitcher::optimizeFrames(const std::vector<FrameData>& frame_data, 
                                        const std::string& output_path, 
                                        bool use_pose,
                                        bool use_between_factor,
                                        const FilterConfig& filter_config,
                                        const ICPConfig& icp_config,
                                        const VisualizerConfig& visualizer_config) {

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_values;
    size_t frame_idx = 0;
    std::mt19937 mt;
    
    // Vector to store filtered point clouds
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> filtered_point_clouds;
    std::vector<std::shared_ptr<gtsam_points::PointCloudCPU>> list_of_points;
    gtsam::Pose3 current_pose;
    gtsam::Pose3 prev_pose;

    // Iterate over all frames (point clouds with transforms)
    for (const auto& frame : frame_data) {
        // Filter the point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = frame.point_cloud;
        filterPointCloud(pcl_cloud, filter_config);

        std::vector<Eigen::Vector3d> points;
        std::vector<double> intensities;  // To store computed intensity values
        
        points.reserve(pcl_cloud->size());
        intensities.reserve(pcl_cloud->size());
        for (const auto& p : pcl_cloud->points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                points.push_back(Eigen::Vector3d(p.x, p.y, p.z));

                double intensity = 0.2989 * p.r + 0.5870 * p.g + 0.1140 * p.b;
                intensities.push_back(intensity);
            }
        }

        if (points.empty()) {
            ROS_WARN("Skipping frame %zu: No valid points", frame_idx);
            continue;
        }

        filtered_point_clouds.push_back(pcl_cloud); // Store the filtered point cloud
        // if no valid pose for frame, dont use pose        
        bool check_pose = frame.transform.translation.x != 0 && frame.transform.translation.y != 0 && frame.transform.translation.z != 0; 
        if (!check_pose) {
            ROS_WARN("No valid pose for frame %zu", frame_idx);
        }

        auto current_frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
        current_frame->add_intensities(intensities);
        current_frame = gtsam_points::random_sampling(current_frame, icp_config.randomsampling_rate, mt);
        current_frame->add_covs(gtsam_points::estimate_covariances(current_frame->points, current_frame->size(), icp_config.k_neighbors, icp_config.num_threads));
        list_of_points.push_back(current_frame);

        std::shared_ptr<gtsam_points::PointCloudCPU> prev_frame;
        
        // Add first pose as prior since frame_idx = 0
        if (frame_idx == 0) {
            // alwasys use the first pose as prior to get correct ground truth reference
            current_pose = convertTransformToEigen(frame.transform);
            if ((check_pose && use_pose)) { 
                initial_values.insert(frame_idx, current_pose);
            }     
            else {
                initial_values.insert(frame_idx, gtsam::Pose3());
            }
            auto prior_factor = gtsam::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                0, current_pose, prior_noise_);
            graph.add(prior_factor);
        }
        else {
            // Add current pose as initial value and use the transform if found
            if (check_pose && use_pose) {
                current_pose = convertTransformToEigen(frame.transform);
                initial_values.insert(frame_idx, current_pose);
            }     
            else {
                initial_values.insert(frame_idx, gtsam::Pose3());
            }

            if (icp_config.sequential) {
                // Add ICP factors between current frame and all previous frames
                for (size_t prev_idx = 0; prev_idx < frame_idx; ++prev_idx) {
                    prev_frame = list_of_points[prev_idx];
                    addICPFactor(prev_frame, current_frame, icp_config, graph, frame_idx, prev_idx);
                }
            }
            else {
                // Add ICP factors between the current frame and previous frames
                addICPFactor(prev_frame, current_frame, icp_config, graph, frame_idx, frame_idx - 1);
            }

            if (use_between_factor) {
                // add between factor between current frame and previous frame
                gtsam::Vector6 sigmas;
                sigmas << 0.05, 0.05, 0.05,    // Translation (x,y,z) - less constrained
                            0.001, 0.001, 0.001;  // Rotation (roll,pitch,yaw) - constrain much more
                auto odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
                current_pose = convertTransformToEigen(frame.transform);
                auto odom_ = prev_pose.inverse() * current_pose; // need to check if this is correct
                auto between_factor = gtsam::BetweenFactor<gtsam::Pose3>(
                    frame_idx - 1, frame_idx, odom_, odom_noise_);
                graph.add(between_factor);
            }
        }
        frame_idx++;
        prev_frame = current_frame;
        prev_pose = current_pose;
    }

    // Optimize the graph with all frames
    try {
        gtsam_points::LevenbergMarquardtOptimizerExt optimizer(
            graph, initial_values, lm_params_);
        auto final_values = optimizer.optimize();
        if (visualizer_config.enabled) {
            visualizePointClouds(final_values, filtered_point_clouds, visualizer_config, true);
            visualizePointClouds(final_values, filtered_point_clouds, visualizer_config, false);
            }
        graph.print("Final factor graph:\n");
        saveResults(final_values, filtered_point_clouds, output_path, true, true); // Use filtered point clouds
        //saveResults(final_values, filtered_point_clouds, output_path, true, false); // Use filtered point clouds
        //saveResults(final_values, filtered_point_clouds, output_path, false, true); 
    } catch (const std::exception& e) {
        ROS_ERROR("Final optimization failed: %s", e.what());
    }
    ros::shutdown();
}

// Helper function to add an ICP factor to the graph, make code less repetitive
void PointCloudStitcher::addICPFactor(const std::shared_ptr<gtsam_points::PointCloudCPU>& prev_frame,
                                         const std::shared_ptr<gtsam_points::PointCloudCPU>& current_frame,
                                         const ICPConfig& icp_config,
                                         gtsam::NonlinearFactorGraph& graph,
                                         size_t frame_idx,
                                         size_t prev_idx) {
    
    if (icp_config.type == 0) {
        return; // No ICP factor
    }
    // add GICP factor
    else if (icp_config.type == 1) {
        gtsam_points::NearestNeighborSearch::Ptr target_tree = std::make_shared<gtsam_points::KdTree>(prev_frame->points, prev_frame->size());
        auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
            prev_idx, frame_idx, prev_frame, current_frame, target_tree);
        icp_factor->set_max_correspondence_distance(icp_config.max_correspondence_distance);
        icp_factor->set_num_threads(icp_config.num_threads);
        graph.add(icp_factor);
    }
    // add color GICP
    else if (icp_config.type == 2) {
        gtsam_points::NearestNeighborSearch::Ptr target_tree = std::make_shared<gtsam_points::KdTree>(prev_frame->points, prev_frame->size());
        gtsam_points::IntensityGradients::Ptr target_gradients = gtsam_points::IntensityGradients::estimate(prev_frame, 10, 50);
        auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedColoredGICPFactor>(
            prev_idx, frame_idx, prev_frame, current_frame, target_tree, target_gradients);
        icp_factor->set_max_correspondence_distance(icp_config.max_correspondence_distance);
        icp_factor->set_num_threads(icp_config.num_threads);
        graph.add(icp_factor);
    }
    // add ICP factor
    else {
        auto icp_factor = gtsam::make_shared<gtsam_points::IntegratedICPFactor>(prev_idx, frame_idx, prev_frame, current_frame);
        icp_factor->set_max_correspondence_distance(icp_config.max_correspondence_distance);
        icp_factor->set_num_threads(icp_config.num_threads);
        graph.add(icp_factor);
    }
}

void PointCloudStitcher::saveResults(
    const gtsam::Values& values, 
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& filtered_point_clouds,
    const std::string& output_path,
    bool stitched,
    bool colored) {
    
    // Save stitched point cloud with each cloud in a separate color
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr stitched_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<Eigen::Vector4f> colors = generateColors(filtered_point_clouds.size());

    for (size_t i = 0; i < filtered_point_clouds.size(); ++i) {
        const auto& pcl_cloud = filtered_point_clouds[i];
        pcl::PointCloud<pcl::PointXYZRGB> transformed_cloud;

        if (stitched) {
            const auto& optimized_pose = values.at<gtsam::Pose3>(i);
            Eigen::Matrix4f transform = optimized_pose.matrix().cast<float>();
            pcl::transformPointCloud(*pcl_cloud, transformed_cloud, transform);

        } else {
            transformed_cloud = *pcl_cloud;
        }

        // Color each point in the transformed cloud
        Eigen::Vector4f color = colors[i % colors.size()];
        uint8_t r = colored ? static_cast<uint8_t>(color(0) * 255) : 255;
        uint8_t g = colored ? static_cast<uint8_t>(color(1) * 255) : 255;
        uint8_t b = colored ? static_cast<uint8_t>(color(2) * 255) : 255;
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                        static_cast<uint32_t>(g) << 8  |
                        static_cast<uint32_t>(b));
        for (auto& point : transformed_cloud.points) {
            point.rgb = *reinterpret_cast<float*>(&rgb);
        }

        *stitched_cloud += transformed_cloud;
    }

    std::string suffix = stitched ? "_stitched" : "_original";
    std::string color_suffix = colored ? "_colored" : "_uncolored";
    pcl::io::savePCDFile(output_path + ".pcd", *stitched_cloud); //+ suffix + color_suffix
    ROS_INFO("Saved stitched pointcloud to %s", (output_path + suffix + color_suffix).c_str());
}

void PointCloudStitcher::visualizePointClouds(
    const gtsam::Values& values, 
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& filtered_point_clouds,
    const VisualizerConfig& visualizer_config,
    bool stiched) {
    
    auto viewer = guik::LightViewer::instance();

    // Disable the grid (ground)
    viewer->set_draw_xy_grid(false);

    // Define a list of colors
    std::vector<Eigen::Vector4f> colors = generateColors(values.size());

    for (size_t i = 0; i < values.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud = filtered_point_clouds[i];
        pcl::PointCloud<pcl::PointXYZRGB> visualize_cloud;

        if (stiched) {
            const auto& optimized_pose = values.at<gtsam::Pose3>(i);
            Eigen::Matrix4f transform = optimized_pose.matrix().cast<float>();
            pcl::transformPointCloud(*current_cloud, visualize_cloud, transform);
        } else {
            visualize_cloud = *current_cloud;
        }

        // Convert pcl::PointCloud to Eigen::Matrix
        Eigen::Matrix<float, 3, Eigen::Dynamic> eigen_points(3, visualize_cloud.size());
        for (size_t j = 0; j < visualize_cloud.size(); ++j) {
            eigen_points(0, j) = visualize_cloud.points[j].x;
            eigen_points(1, j) = visualize_cloud.points[j].y;
            eigen_points(2, j) = visualize_cloud.points[j].z;
        }

        // Use a different color for each point cloud
        Eigen::Vector4f color = colors[i % colors.size()];

        // Create a ShaderSetting and set the color
        guik::ShaderSetting shader_setting;
        shader_setting.set_color(color);
        shader_setting.set_point_size(visualizer_config.point_size); 
        viewer->update_drawable("frame_" + std::to_string(i), 
            std::make_shared<glk::PointCloudBuffer>(eigen_points), shader_setting);
    }

    viewer->spin();
    viewer->clear_drawables();
}

// Generate a palette with 'count' unique colors
std::vector<Eigen::Vector4f> PointCloudStitcher::generateColors(int count) {
    std::vector<Eigen::Vector4f> colors;
    for (int i = 0; i < count; i++) {
        float hue = (360.0f * i) / count; // evenly distributed hues
        float r, g, b;
        int j = static_cast<int>(hue / 60.0f) % 6;
        float f = (hue / 60.0f) - j;
        float p = 0.0f;
        float v = 1.0f;
        float q = 1.0f - f;
        float t = -f;
        switch (j) {
            case 0: r = v, g = t, b = p; break;
            case 1: r = q, g = v, b = p; break;
            case 2: r = p, g = v, b = t; break;
            case 3: r = p, g = q, b = v; break;
            case 4: r = t, g = p, b = v; break;
            case 5: r = v, g = p, b = q; break;
            default: r = g = b = 0.0f; break;
        }
        colors.push_back(Eigen::Vector4f(r, g, b, 1.0f));
    }
    return colors;
}

// Helper function to convert a geometry_msgs/Transform to a GTSAM Pose3
gtsam::Pose3 PointCloudStitcher::convertTransformToEigen(
    const geometry_msgs::Transform& transform) {
    
    // Create translation vector
    gtsam::Point3 translation(
        transform.translation.x,
        transform.translation.y,
        transform.translation.z);
    
    // Create rotation from quaternion
    gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(
        transform.rotation.w,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z);
    
    // Return the GTSAM pose
    return gtsam::Pose3(rotation, translation);
}

// Function to filter the point cloud
void PointCloudStitcher::filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_cloud,
                                        const FilterConfig& filter_config) {

    // Perform PassThrough filter to remove points that are far away
    if (filter_config.enabled) {
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(pcl_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(filter_config.near_threshold, filter_config.far_threshold);
        pass.filter(*pcl_cloud);
    }

    bool remove_sparse_points = true;
    // Remove sparse points
    if (remove_sparse_points) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(pcl_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*pcl_cloud);
    }

    bool moving_least_squares = false;
    // Apply Moving Least Squares filter
    if (moving_least_squares) {
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_smoothed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
        mls.setComputeNormals(true);
        mls.setInputCloud(pcl_cloud);
        mls.setPolynomialOrder(true);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(0.03);
        mls.process(*cloud_smoothed);
        pcl_cloud = cloud_smoothed;
    }
    
    // Perform RANSAC to remove outliers
    bool use_ransac = false;
    if (use_ransac) {
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE); // Adjust the model 
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.005); // Adjust the threshold
        seg.setInputCloud(pcl_cloud);
        seg.segment(*inliers, *coefficients);
        
        // Extract inliers
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(pcl_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*pcl_cloud);
    }
}