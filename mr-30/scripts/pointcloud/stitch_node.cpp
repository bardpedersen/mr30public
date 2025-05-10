#include "rosbag_reader.hpp"
#include "pointcloud_stitcher.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "synchronized_pointcloud_stitcher");
    ros::NodeHandle nh("~");
    
    // Define parameters
    std::string bag_path, output_path, output_path_image, point_cloud_topic;
    bool use_pose, use_between_factor, filter_far_points, verbose, icp_sequential, enable_visualizer, camera_noise, use_perlin, robot_noise;
    float far_point_threshold, near_point_threshold, point_size, max_correspondence_distance, camera_base_stddev, camera_scale_factor, camera_mean, perline_scale, perlin_amplitude, robot_mean, robot_tran_stddev, robot_rot_stddev;
    int icp_type, num_threads, k_neighbors, perlin_nr_layers, number_of_frames;
    double randomsampling_rate;

    // Get parameters from ros params
    nh.getParam("output_path", output_path);
    nh.getParam("output_path_image", output_path_image);
    nh.getParam("use_pose", use_pose);
    nh.getParam("use_between_factor", use_between_factor);
    nh.getParam("filter_far_points", filter_far_points);
    nh.getParam("far_point_threshold", far_point_threshold);
    nh.getParam("near_point_threshold", near_point_threshold);
    nh.getParam("verbose", verbose);
    nh.getParam("icp_type", icp_type);
    nh.getParam("icp_sequential", icp_sequential);
    nh.getParam("max_correspondence_distance", max_correspondence_distance);
    nh.getParam("num_threads", num_threads);
    nh.getParam("k_neighbors", k_neighbors);
    nh.getParam("randomsampling_rate", randomsampling_rate);
    nh.getParam("enable_visualizer", enable_visualizer);
    nh.getParam("point_size", point_size);
    nh.getParam("camera_noise", camera_noise);
    nh.getParam("camera_mean", camera_mean);
    nh.getParam("camera_base_stddev", camera_base_stddev);
    nh.getParam("camera_scale_factor", camera_scale_factor);
    nh.getParam("use_perlin", use_perlin);
    nh.getParam("perline_scale", perline_scale);
    nh.getParam("perlin_amplitude", perlin_amplitude);
    nh.getParam("perlin_nr_layers", perlin_nr_layers);
    nh.getParam("robot_noise", robot_noise);
    nh.getParam("robot_mean", robot_mean);
    nh.getParam("robot_tran_stddev", robot_tran_stddev);
    nh.getParam("robot_rot_stddev", robot_rot_stddev);
    nh.getParam("number_of_frames", number_of_frames);
    nh.getParam("point_cloud_topic", point_cloud_topic);

    // Get bag path from ros params
    if (!nh.getParam("bag_path", bag_path)) {
        ROS_ERROR("No bag path provided!");
        return 1;
    }

    // Read from rosbag
    RosbagReader reader;
    cameraNoiseConfig camera_config = {camera_noise, camera_mean, camera_base_stddev, camera_scale_factor, use_perlin, perline_scale, perlin_amplitude, perlin_nr_layers};
    robotNoiseConfig robot_config = {robot_noise, robot_mean, robot_tran_stddev, robot_rot_stddev};
    auto frame_data = reader.readBag(bag_path, output_path_image, camera_config, robot_config, number_of_frames, point_cloud_topic);

    // Check if any frames were read
    ROS_INFO("Read %zu frames from bag", frame_data.size());
    if (frame_data.empty()) {
        ROS_ERROR("No valid frames found in bag!");
        return 1;
    }

    FilterConfig filter_config = {filter_far_points, far_point_threshold, near_point_threshold};
    ICPConfig icp_config = {icp_type, max_correspondence_distance, num_threads, k_neighbors, randomsampling_rate, icp_sequential};
    VisualizerConfig visualizer_config = {enable_visualizer, point_size};

    // Stitch the point clouds
    PointCloudStitcher stitcher(verbose);
    stitcher.stitchPointClouds(frame_data, 
                                output_path, 
                                use_pose, 
                                use_between_factor,
                                filter_config,
                                icp_config,
                                visualizer_config);

    return 0;
}