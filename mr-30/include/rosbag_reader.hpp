#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>



struct FrameData {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud;
    geometry_msgs::Transform transform;
    ros::Time timestamp;
};

struct cameraNoiseConfig {
    bool enabled;
    float mean;
    float base_stddev;
    float scale_factor;
    bool use_perlin;
    float perline_scale;
    float perlin_amplitude;
    int perlin_nr_layers; 
};

struct robotNoiseConfig {
    bool enabled;
    float mean;
    float tran_stddev;
    float rot_stddev;
};

struct TrajectoryMetrics {
    int num_points;          
    double distance_traveled; 
    double time_elapsed;   
};

class RosbagReader {
public:
    RosbagReader();
    std::vector<FrameData> readBag(const std::string& bag_file,
                                    const std::string& output_path_image, 
                                    const cameraNoiseConfig& camera_config, 
                                    const robotNoiseConfig& robot_config,
                                    const int number_of_frames,
                                    const std::string& point_cloud_topic);

    void robotNoise(geometry_msgs::Transform& transform, const robotNoiseConfig& config);
    void cameraNoise(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const cameraNoiseConfig& config);
    void cameraPerlinNoise(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const cameraNoiseConfig& config);
    void saveDepthAsImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std::string& output_path_image, const std::string& filename);

    void saveAmplifiedDifferenceImage(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& noisy_cloud,
        const std::string& output_path_image,
        const std::string& filename);

    cv::Mat addColorScale(const cv::Mat& image, float min_val, float max_val);
    cv::Scalar colorFromJet(float normalized);

    void calculateTrajectoryMetrics(const std::vector<FrameData>& frame_data);

private:
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
    std::default_random_engine generator; // Random number generator

};