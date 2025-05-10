#!/usr/bin/env python3  

# <vares.lucas@gmail.com, 08Jul2024>
# modified by: <bard.tollef.pedersen@nmbu.no, 20mar2025>

import time
import rospy
import yaml
import threading

import tf2_ros
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped, TransformStamped
from gazebo_msgs.msg import ModelState, ModelStates
from nav_msgs.msg import Path

class MoveCamera:
    def __init__(self, modelName="camera"):
        self.frame_id = "world"
        self.child_id = "base_link"
        self.modelName = modelName

        self.state_msg = ModelState()
        self.state_msg.model_name = modelName
        self.state_msg.reference_frame = self.frame_id    
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        # Initialize transform message
        self.transform_msg = TransformStamped()
        self.transform_msg.header.frame_id = self.frame_id
        self.transform_msg.child_frame_id = self.child_id
        self.transform_msg.transform.rotation.w = 1.0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_broadcaster.sendTransform(self.transform_msg)

        # Add trajectory tracking and visualization (from MoveCamera)
        self.trajectory_points = []

        self.wait_for_model()
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=True)
        self.trajectory_path_pub = rospy.Publisher('/camera_trajectory', Path, queue_size=10)

        self.obstacles = []
        self.camera_size = 0.05  # Adjust based on your camera model size
        self.load_obstacles()
        self.start_tf_broadcast_loop()

    def start_tf_broadcast_loop(self):
        """Start a thread that continuously broadcasts the transform"""
        
        def broadcast_loop():
            rate = rospy.Rate(30)  # 30Hz is typically sufficient for smooth transforms
            rospy.loginfo("Starting continuous TF broadcast loop")
            
            while not rospy.is_shutdown():
                try:
                    # Use current time for each broadcast
                    self.transform_msg.header.stamp = rospy.Time.now()
                    self.tf_broadcaster.sendTransform(self.transform_msg)
                except Exception as e:
                    rospy.logerr(f"Error broadcasting transform: {e}")
                rate.sleep()
        
        # Start broadcasting thread
        self.broadcast_thread = threading.Thread(target=broadcast_loop)
        self.broadcast_thread.daemon = True
        self.broadcast_thread.start()

    def load_obstacles(self):
        """Load obstacle definitions from YAML"""
        try:
            with open('/home/ok/mr30_ws/src/abb_control/config/obstacles.yaml', 'r') as file:
                obstacle_data = yaml.safe_load(file)
                if 'obstacles' in obstacle_data:
                    self.obstacles = obstacle_data['obstacles']
                    rospy.loginfo(f"Loaded {len(self.obstacles)} obstacles")
                else:
                    rospy.logwarn("No obstacles found in YAML file")
        except Exception as e:
            rospy.logerr(f"Failed to load obstacles: {e}")

    def check_collision(self, position):
        """
        Check if the given position would result in a collision
        :param position: [x, y, z] position to check
        :return: True if collision, False if safe
        """
        for obstacle in self.obstacles:
            # Unpack obstacle: [dim_x, dim_y, dim_z, pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w]
            dim = obstacle[:3]
            pos = obstacle[3:6]
            
            # Simple AABB collision check (assumes axis-aligned boxes)
            # Calculate distance from point to box center
            dx = abs(position[0] - pos[0])
            dy = abs(position[1] - pos[1])
            dz = abs(position[2] - pos[2])
            
            # Add camera radius to the obstacle dimensions
            box_x = dim[0]/2 + self.camera_size
            box_y = dim[1]/2 + self.camera_size
            box_z = dim[2]/2 + self.camera_size
            
            # Check if point is inside the expanded box
            if dx <= box_x and dy <= box_y and dz <= box_z:
                return True  # Collision detected
                
        return False  # No collision
    
    def wait_for_model(self):
        """Wait until the model appears in Gazebo's model_states topic"""
        rospy.loginfo(f"Waiting for model '{self.modelName}' to appear in Gazebo...")
        
        model_found = False
        while not model_found and not rospy.is_shutdown():
            try:
                # Get current models from Gazebo
                model_states = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5.0)
                
                # Check if our model is in the list
                if self.modelName in model_states.name:
                    model_found = True
                    rospy.loginfo(f"Model '{self.modelName}' found in Gazebo")
                else:
                    rospy.loginfo(f"Waiting for model '{self.modelName}'... (not found in model_states)")
                    time.sleep(1.0)
            except rospy.ROSException:
                rospy.loginfo("Waiting for /gazebo/model_states topic...")
        
        return model_found

    def move(self, camera_pose):
        """
        Move the camera to the specified pose and update trajectory
        Implements MoveCamera's move functionality
        """
        position = [
            camera_pose.position.x,
            camera_pose.position.y,
            camera_pose.position.z
        ]
        
        # Check for collisions
        if self.check_collision(position):
            rospy.logwarn("Blocked movement: Position would cause collision with obstacle")
            return False

        _time = rospy.Time.now()
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = _time
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose = camera_pose
        
        # Update state message directly
        self.state_msg.pose = camera_pose
        
        # Update transform for TF broadcasting
        self.transform_msg.header.stamp = _time
        self.transform_msg.transform.translation.x = camera_pose.position.x
        self.transform_msg.transform.translation.y = camera_pose.position.y
        self.transform_msg.transform.translation.z = camera_pose.position.z
        self.transform_msg.transform.rotation = camera_pose.orientation
        
        # Apply the model state in Gazebo
        self.set_state(self.state_msg)
        
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = _time
        pose.pose.position.x = camera_pose.position.x
        pose.pose.position.y = camera_pose.position.y
        pose.pose.position.z = camera_pose.position.z
        pose.pose.orientation.x = camera_pose.orientation.x
        pose.pose.orientation.y = camera_pose.orientation.y
        pose.pose.orientation.z = camera_pose.orientation.z
        pose.pose.orientation.w = camera_pose.orientation.w
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(pose)
        
        # Publish path
        self.trajectory_path_pub.publish(self.path_msg)
        return True