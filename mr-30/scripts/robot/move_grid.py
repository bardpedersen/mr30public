#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations
from move_to_point import MoveToPoint
from sensor_msgs.msg import PointCloud2

"""
Assume the plant is within a lying down half sylender with radius 0.5 and length 1
"""
class MoveGrid:
    def __init__(self, 
                 radius=0.25, 
                 length=0.5, 
                 num_points_x=4, 
                 num_points_y=3, 
                 target = [0.45, 0.2, 0.3], 
                 cylender_cent = [0.3, 0.15, 0.3], 
                 length_start=0.0,
                 radius_start=np.pi/2 + np.pi/7,
                 radius_stop=np.pi + np.pi/10,
                 group_name="manipulator", 
                 pointcloud_topic="/camera/depth/color/points_pose"):
        self.radius = radius
        self.length = length
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.group_name = group_name
        self.length_start = length_start
        self.length_stop = length
        self.radius_start = radius_start
        self.radius_stop = radius_stop
        self.move_to_point = MoveToPoint(self.group_name)
        self.target = target
        self.cylender_cent = cylender_cent
        self.pointcloud_msg = None
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pointcloud_callback, queue_size=1, buff_size=100000000)
        self.pointcloud_publisher = rospy.Publisher(pointcloud_topic, PointCloud2, queue_size=1)
        
        # create color for loginfo
        self.green_color = "\033[92m"
        self.reset_color = "\033[0m"
        self.red_color = "\033[91m"
        
    def create_points(self):
        x_values = np.linspace(self.length_start, self.length_stop, self.num_points_x)
        theta_values = np.linspace(self.radius_start, self.radius_stop, self.num_points_y)        
        points = []
        for x in x_values:
            for theta in theta_values:
                # Compute the point on the surface relative to the target
                y = self.radius * np.cos(theta)
                z = self.radius * np.sin(theta)
                point = np.array([x, y, z]) + self.cylender_cent
                
                # Compute the direction vector towards the center (target)
                direction_vector = self.target - point
                direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the vector
                
                # Compute the quaternion from the direction vector
                # Assuming the initial orientation is along the x-axis
                initial_vector = np.array([1, 0, 0])
                rotation_axis = np.cross(initial_vector, direction_vector)
                rotation_angle = np.arccos(np.dot(initial_vector, direction_vector))
                quaternion = tf.transformations.quaternion_about_axis(rotation_angle, rotation_axis)
                
                points.append([point[0], point[1], point[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        return points

    def move(self):
        points = self.create_points()
        rospy.loginfo(f"Created points")
        self.move_to_point.show_points_rviz(points)
        #self.move_to_point.show_cylinder_rviz(self.length, self.radius, self.target)
        for point in points:
            success = self.move_to_point.move_to_point_cartesian(point)
            if success:
                rospy.loginfo(f"{self.green_color}Moved to point: {point} successfully{self.reset_color}")
                self.pointcloud_publisher.publish(self.pointcloud_msg)
            else:
                rospy.loginfo("Fail to move cartesian, try to move with joint")
                success = self.move_to_point.move_to_point(point)
                if success:
                    rospy.loginfo(f"{self.green_color}Moved to point: {point} successfully{self.reset_color}")
                    self.pointcloud_publisher.publish(self.pointcloud_msg)
                else:
                    rospy.loginfo(f"{self.red_color}Fail to move to point: {point}{self.reset_color}")
        
            rospy.sleep(2)
        return 
    
    def pointcloud_callback(self, msg):
        self.pointcloud_msg = msg

    def shutdown_hook(self):
        rospy.loginfo("Shutting down move_grid node")

if __name__ == "__main__":
    rospy.init_node('move_sylendrical', anonymous=True)
    move_grid = MoveGrid()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        move_grid.move()
        rate.sleep()