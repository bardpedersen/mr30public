cmake_minimum_required(VERSION 3.0.2)
project(mr-30)

set(CMAKE_CXX_STANDARD 17)
set(OpenGL_GL_PREFERENCE "GLVND")

option(BUILD_GPU "Build GPU" ON)

find_package(catkin REQUIRED COMPONENTS
  random_strawberry_creator
  realsense2_description
  realsense_gazebo_plugin
  rospy
  roscpp
  rosbag
  std_msgs
  sensor_msgs
  yumi_control
  yumi_description
  yumi_hw
  yumi_launch
  yumi_motion_api
  yumi_moveit_config
  yumi_support 
  pcl_conversions
  pcl_ros
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
)

find_package(OpenCV REQUIRED)
find_package(gtsam_points REQUIRED)
find_package(Iridescence REQUIRED)
find_package(PCL REQUIRED
  common
  io
  filters  
  )
find_package(Boost REQUIRED COMPONENTS filesystem system)

if (BUILD_GPU)
  find_package(CUDAToolkit REQUIRED)
endif()

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS})


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(pointcloud_processing
  scripts/pointcloud/rosbag_reader.cpp
  scripts/pointcloud/pointcloud_stitcher.cpp
)

target_link_libraries(pointcloud_processing
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  gtsam_points::gtsam_points
  Iridescence::Iridescence
)

add_executable(stitch scripts/pointcloud/stitch_node.cpp)
target_link_libraries(stitch 
  ${catkin_LIBRARIES}
  pointcloud_processing
  )
  
add_executable(pointcloud_evaluator scripts/pointcloud/pointcloud_evaluator.cpp)
target_link_libraries(pointcloud_evaluator 
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  )

###################################################################
# Tests
###################################################################
#add_executable(readpointcloud_test scripts/pointcloud/_readpointcloud_test.cpp)
#target_link_libraries(readpointcloud_test 
#  ${catkin_LIBRARIES}
#  ${PCL_LIBRARIES}
#  )

#add_executable(stitch_test scripts/pointcloud/_stitch_test.cpp)
#target_link_libraries(stitch_test 
#  gtsam_points::gtsam_points
#  Iridescence::Iridescence
#  )