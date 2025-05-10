#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>

// Function to read point clouds from a ROS bag
std::vector<pcl::PointCloud<pcl::PointXYZ>> readPointCloudsFromBag(
    const std::string& bag_file,
    const std::vector<std::string>& topics)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>> pointclouds;
    
    try {
        rosbag::Bag bag;
        bag.open(bag_file, rosbag::bagmode::Read);
        
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        
        for (const rosbag::MessageInstance& m : view) {
            sensor_msgs::PointCloud2::ConstPtr pc2_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (pc2_msg) {
                pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
                pcl::fromROSMsg(*pc2_msg, pcl_cloud);
                pointclouds.push_back(pcl_cloud);
            }
        }
        
        bag.close();
    }
    catch (const rosbag::BagException& e) {
        ROS_ERROR_STREAM("Failed to read bag: " << e.what());
    }
    
    return pointclouds;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "stitch");
    ros::NodeHandle nh;

    // Configuration parameters
    const std::string bag_file = "/home/ubuntu/Downloads/bag1_gradientnbv.bag";
    const std::vector<std::string> topics = {"/camera/depth/color/points_pose"};

    // Get point clouds from bag
    auto clouds = readPointCloudsFromBag(bag_file, topics);

    // Process point clouds (example: print statistics)
    for (size_t i = 0; i < clouds.size(); ++i) {
        std::cout << "Cloud " << i + 1 << " contains "
                  << clouds[i].size() << " points" << std::endl;
    }

    return 0;
}