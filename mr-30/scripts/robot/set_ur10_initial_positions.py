#!/usr/bin/env python
# filepath: /home/ok/mr30_ws/src/mr-30/scripts/set_ur10_initial_positions.py

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
import time

def set_initial_position():
    rospy.init_node('set_ur10_initial_positions')
    
    # UR10 joint names
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    # Non-colliding joint positions
    positions = [1.57, -1.2, 1.4, -1.87, -1.57, 0.0]
    
    # Wait for the controller to be available
    client = actionlib.SimpleActionClient('/eff_joint_traj_controller/follow_joint_trajectory', 
                                        FollowJointTrajectoryAction)
    rospy.loginfo("Waiting for joint trajectory action...")
    client.wait_for_server()
    rospy.loginfo("Connected to joint trajectory action")
    
    # Create a trajectory goal
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = joint_names
    
    # Create a trajectory point with desired positions
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start = rospy.Duration(3.0)
    
    # Add the point to the trajectory
    goal.trajectory.points.append(point)
    
    # Send the goal
    rospy.loginfo("Sending initial joint positions goal...")
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration(5.0))
    
    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("UR10 initial positions set successfully")
    else:
        rospy.logerr("Failed to set UR10 initial positions")

if __name__ == '__main__':
    try:
        set_initial_position()
    except rospy.ROSInterruptException:
        pass