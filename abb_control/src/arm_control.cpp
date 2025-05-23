/*
 Author: Akshay Kumar Burusa
 Email: akshaykumar.burusa@wur.nl
 */

#include <abb_control/arm_control.hpp>
#include <utils/utils.hpp>

namespace arm_control {

/* Constructor.
 * Inputs: Name of MoveIt planning group */
ArmControl::ArmControl(std::string planning_group)
    : move_group_(planning_group) {
      ros::NodeHandle nh("~");
      nh.setParam("/move_group/trajectory_execution/allowed_start_tolerance", 0.02);
      nh.setParam("/move_group/path_simplifier", true);
      
    }

// Destructor.
ArmControl::~ArmControl() {}

/* Move end-effector to goal pose.
 * Output: Status of pose update */
bool ArmControl::MoveToGoal(geometry_msgs::Pose goal_pose) {
  // Set planning pose.
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  move_group_.setPlannerId("LazyPRMstar");
  move_group_.setPlanningTime(5.0);
  move_group_.setMaxVelocityScalingFactor(0.7);  // Smoother motion
  move_group_.setGoalPositionTolerance(0.001);   // More precise positioning
  move_group_.setGoalOrientationTolerance(0.01); // More precise orientation
  move_group_.setPoseTarget(goal_pose);
  // Plan trajectory.
  bool success = (move_group_.plan(my_plan) ==
                  moveit::planning_interface::MoveItErrorCode::SUCCESS);
  ROS_INFO("Visualizing (pose goal) %s", success ? "SUCCEEDED" : "FAILED");
  // Execute trajectory.
  if (success) {
    ros::Duration(0.005).sleep();
    success = (move_group_.execute(my_plan) ==
               moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO("Execution (pose goal) %s", success ? "SUCCEEDED" : "FAILED");
  }
  // Return status.
  return success;
}

/* Move end-effector to goal pose using Cartesian path.
 * Input: List of waypoints
 * Output: Status of pose update */
bool ArmControl::MoveToGoalCartesian(geometry_msgs::Pose goal_pose) {
  std::vector<geometry_msgs::Pose> waypoints;
  waypoints.push_back(goal_pose);

  moveit_msgs::RobotTrajectory trajectory;
  const double jump_threshold = 0.0;
  const double eef_step = 0.01; // 1 cm

  double fraction = move_group_.computeCartesianPath(waypoints, eef_step, trajectory);
  if (fraction < 1.0) {
    ROS_WARN("Cartesian path planning only succeeded for %.2f%% of the path", fraction * 100.0);
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  bool success = (move_group_.execute(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  ROS_INFO("Execution (Cartesian path) %s", success ? "SUCCEEDED" : "FAILED");
  return success;
}

/* Service to move end-effector to goal pose.
 * Input: ArmGoal request, ArmGoal response
 * Output: Status of pose update */
bool ArmControl::MoveToGoalSrv(abb_control::ArmGoal::Request &req,
                               abb_control::ArmGoal::Response &res) {
  res.success = MoveToGoalCartesian(req.goal_pose);
  if (!res.success) {
     res.success = MoveToGoal(req.goal_pose);
  }

  return res.success;
}

/* Set collision constraints.
 * Input: Array of sizes of collision object, Array of poses of collision object
 */
void ArmControl::AddCollisionObjects(std::vector<geometry_msgs::Vector3> dims,
                                     geometry_msgs::PoseArray obstacle_poses) {
  // Vector of collision objects.
  std::vector<moveit_msgs::CollisionObject> collision_objects;
  // Define collision objects.
  for (int i = 0; i < dims.size(); i++) {
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = "world";
    // The id of the object is used to identify it.
    collision_object.id = "box" + std::to_string(i);
    // Define a box to add to the world.
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    // Add collision around object
    // Re-define box
    primitive.dimensions[0] = dims[i].x;
    primitive.dimensions[1] = dims[i].y;
    primitive.dimensions[2] = dims[i].z;
    // Add as collision object.
    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(obstacle_poses.poses[i]);
    collision_object.operation = collision_object.ADD;
    collision_objects.push_back(collision_object);
  }
  ROS_INFO("Add an object into the world");
  planning_scene_interface_.applyCollisionObjects(collision_objects);
  // Sleep so we have time to see the object in RViz
  sleep(1.0);
}

} // namespace arm_control