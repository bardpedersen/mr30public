# ROS workspace used for MR-30

The source code used for my master thesis at NMBU. The goal of the thesis was to create a system that can autonomously scan a strawberry plant and create a 3D model of the plant. The system consists of a UR robot, a Yumi robot, a realsense camera and a Zivid camera. The system is controlled by ROS and uses MoveIt for motion planning. The system is simulated in Gazebo and the point clouds are stitched together using GTSAM. For any questions, please contact me at email.

## How to use

This will launch the gazebo world with the robot, the realsense camera and the strawberry:
```bash
roslaunch mr-30 moveit.launch
```

To launch real realsense camera:
```bash
roslaunch mr-30 realsense.launch
```

To record a rosbag with the right topics:
```bash
rosbag record -O bag1_nbv.bag /tf /tf_static /camera/depth/color/points_pose
```

To stich the point clouds from the rosbag:
```bash
roslauch mr-30 stitch.launch
```
Here are alot of parameters, changing how it is stich and if noise is added.

To test the quality of the stiched point cloud:
```bash
rosrun mr-30 pointcloud_evaluator
```

Turn sdf to stl.
```bash
./sdf2stl.py robot_model.sdf output.stl
```

## Folder structure

abb_controll: This package is used to control the ABB robot. from gradientnbv.

abb_driver: industrial_core, yumi_ros: These folders are used to control the ABB robot.

common: from gradientnbv

extra: contains all things that are not ros package/code

helper: This package is used to automate repetitive tasks. Like evaluating the stitched point cloud.

industrial_core: This package is used to control the UR robot.

mr-30: I created this package to control all the elements. I will create scripts to control the robot in this package.

realsense_gazebo_plugin: This package is used to simulate the realsense camera in gazebo.

realsense2_description, realsense_gazebo_plugin: These packages are used to simulate the realsense camera in gazebo.

universal_robot: This package is used to control the UR robot with moveit and simulate the robot in gazebo.

Universal_Robots_ROS_Driver: This package is used to control the UR real life robot.

ur_msgs: This package is used to control the UR robot with moveit and simulate the robot in gazebo.

view_point_planner: This package is used to plan the view points for the robot. From gradientnbv.

yumi_ros: This package is used to control and simulate the Yumi robot.

zivid_ros: This package is used to control the Zivid camera

## Installation

### Conda environment

Create a conda environment from the following file:
```bash
conda env create -f extra/install_conda_pkgs.yaml
```
### Newer version of cmake for ubunut 20.04

This is needed if gtsam_points shall use cuda

https://apt.kitware.com/

### ROS packages

ros-noetic-full-desktop
ros-noetic-moveit*
ros-noetic-rospy-message-converter

Note: may be missing some packages

### GTSAM

#### Eigen from git on ubuntu 20.04
```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout 1fd5ce1

mkdir build && cd build
cmake .. 
make -j$(nproc)
sudo make install
```
#### GTSAM 
```bash
git clone https://github.com/borglab/gtsam
cd gtsam
git checkout 4.2a9

mkdir build && cd build
cmake .. \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF

make -j$(nproc)
sudo make install
```

#### GTSAM Points
```bash
git clone https://github.com/koide3/gtsam_points
git checkout tags/v1.0.6
cd gtsam_points

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON
make -j$(nproc)
sudo make install
```

#### Iridescence
Only for visualization
```bash
sudo apt install -y libglm-dev libglfw3-dev libpng-dev
git clone https://github.com/koide3/iridescence --recursive
mkdir iridescence/build && cd iridescence/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

Remember to modify in file /usr/local/lib/cmake/iridescence/iridescence-targets.cmake
add_library(Iridescence::iridescence SHARED IMPORTED) -> add_library(Iridescence::iridescence SHARED IMPORTED GLOBAL)

## Moveit

To launch moveit with everything:
```bash
roslaunch mr-30 moveit.launch
```

If the urdf is updater or change the way the robot is controlled, the moveit package must be updated. To do this, run the following command:
```bash
cd yumi_ws/src/yumi_ros/yumi_description/urdf/
```
```bash
rosrun xacro xacro yumi.urdf.xacro arms_interface:=PositionJointInterface grippers_interface:=EffortJointInterface yumi_setup:=default -o yumi.urdf
```
Then source the workspace and run again

## Git repositorys used

### Yumi ROS
https://github.com/bhomaidan1990/yumi_ros?tab=readme-ov-file  

### UR robot
https://github.com/UniversalRobots/Universal_Robots_ROS_Driver  
https://github.com/ros-industrial/universal_robot

### Realsense
https://github.com/issaiass/realsense2_description  
https://github.com/issaiass/realsense_gazebo_plugin

### Zivid
https://github.com/zivid/zivid-ros  
https://github.com/sam-xl/zivid-ros/tree/zivid_description

### NBV
https://github.com/akshaykburusa/gradientnbv  

### gtsam_points
https://github.com/borglab/gtsam  
https://github.com/koide3/gtsam_points  
https://gitlab.com/libeigen/eigen.git  

