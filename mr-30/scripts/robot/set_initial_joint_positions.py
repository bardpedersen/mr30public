#!/usr/bin/env python
import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import actionlib

def set_initial_pose():
    rospy.init_node('set_initial_pose')
    
    # Create action clients for both arms
    left_client = actionlib.SimpleActionClient(
        '/yumi/joint_traj_pos_controller_l/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    right_client = actionlib.SimpleActionClient(
        '/yumi/joint_traj_pos_controller_r/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    
    # Wait for action servers
    left_client.wait_for_server()
    right_client.wait_for_server()
    
    # Define the joint angles you want (these are the ones from your launch file)
    left_positions = [-0.2, -2.2, -2.5, 0.0, -3.8, 1.3, 0.9] # [-1.41, -2.1, 0.30, 0.0, 0.0, 0.0, 0.71] so left arm is easier to filter out
    right_positions = [1.41, -2.1, 0.30, 0.0, 0.0, 0.0, -0.71]
    
    # Create trajectory goals for both arms
    left_goal = FollowJointTrajectoryGoal()
    right_goal = FollowJointTrajectoryGoal()
    
    # Set joint names
    left_goal.trajectory.joint_names = [
        'yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_3_l',
        'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l',
        'yumi_joint_7_l'
    ]
    right_goal.trajectory.joint_names = [
        'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_3_r',
        'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r',
        'yumi_joint_7_r'
    ]
    
    # Create trajectory points
    left_point = JointTrajectoryPoint()
    left_point.positions = left_positions
    left_point.time_from_start = rospy.Duration(2.0)
    
    right_point = JointTrajectoryPoint()
    right_point.positions = right_positions
    right_point.time_from_start = rospy.Duration(2.0)
    
    # Add points to trajectories
    left_goal.trajectory.points.append(left_point)
    right_goal.trajectory.points.append(right_point)
    
    # Send goals
    left_client.send_goal(left_goal)
    right_client.send_goal(right_goal)
    
    # Wait for results
    left_client.wait_for_result()
    right_client.wait_for_result()

if __name__ == '__main__':
    try:
        set_initial_pose()
    except rospy.ROSInterruptException:
        pass