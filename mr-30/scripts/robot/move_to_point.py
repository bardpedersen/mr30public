#!/usr/bin/env python3

import sys
import numpy as np
import moveit_commander
import geometry_msgs.msg
import tf.transformations

"""
Assume the plant is within a lying down half sylender with radius 0.5 and length 1
"""

class MoveToPoint:
    def __init__(self, group_name="right_arm"):
        self.group_name = group_name
        self.moveit_commander = moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        
    def move_to_point_cartesian(self, point):
        waypoints = []
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = point[0]
        pose_goal.position.y = point[1]
        pose_goal.position.z = point[2]
        pose_goal.orientation.x = point[3]
        pose_goal.orientation.y = point[4]
        pose_goal.orientation.z = point[5]
        pose_goal.orientation.w = point[6]
        waypoints = [pose_goal]

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints, 
            0.01) 
        if fraction == 1.0:
            self.group.execute(plan, wait=True)
            self.group.stop()
            self.group.clear_pose_targets()
            return True
        
        self.group.stop()
        self.group.clear_pose_targets()
        return False
    
    def move_to_point(self, point):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = point[0]
        pose_goal.position.y = point[1]
        pose_goal.position.z = point[2]
        pose_goal.orientation.x = point[3]
        pose_goal.orientation.y = point[4]
        pose_goal.orientation.z = point[5]
        pose_goal.orientation.w = point[6]
        self.group.set_pose_target(pose_goal)
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if plan:
            return True
        return False

    # show all points with orientation in rviz
    def show_points_rviz(self, points):
        import rospy
        import visualization_msgs.msg
        
        # Create marker publisher
        marker_pub = rospy.Publisher("/visualization_marker_array", 
                                   visualization_msgs.msg.MarkerArray, 
                                   queue_size=10,
                                   latch=True)
        
        markers = visualization_msgs.msg.MarkerArray()
        
        for nr, point in enumerate(points):
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = self.robot.get_planning_frame()
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = nr
            marker.type = visualization_msgs.msg.Marker.CUBE
            marker.action = visualization_msgs.msg.Marker.ADD
            
            # Position & orientation
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.x = point[3]
            marker.pose.orientation.y = point[4]
            marker.pose.orientation.z = point[5]
            marker.pose.orientation.w = point[6]
            
            # Scale
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            
            # Color
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            # Set lifetime (0 = forever)
            marker.lifetime = rospy.Duration(0)
            
            markers.markers.append(marker)
        
        # Publish markers
        marker_pub.publish(markers)
        rospy.sleep(0.1)  # Give time for the message to be published

    def show_cylinder_rviz(self, length=1, radius=0.5, target=[0, 0, 0]):
        cylinder_pose = geometry_msgs.msg.PoseStamped()
        cylinder_pose.header.frame_id = self.robot.get_planning_frame()
        cylinder_pose.pose.position.x = length /2 + target[0]
        cylinder_pose.pose.position.y = target[1]
        cylinder_pose.pose.position.z = target[2]
        quaternion = tf.transformations.quaternion_from_euler(0, np.pi/2, 0)
        cylinder_pose.pose.orientation.x = quaternion[0]
        cylinder_pose.pose.orientation.y = quaternion[1]
        cylinder_pose.pose.orientation.z = quaternion[2]
        cylinder_pose.pose.orientation.w = quaternion[3]
        self.scene.add_cylinder("cylinder", cylinder_pose, length, radius)

if __name__ == "__main__":
    move_grid = MoveToPoint("manipulator")
    point = [0.4, -0.2, 0.2, 0.0, 0.0, 0.7071067811865475, 0.7071067811865475]
    move_grid.move_to_point(point)