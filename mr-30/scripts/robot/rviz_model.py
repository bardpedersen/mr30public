#!/usr/bin/env python3
# filepath: /home/ok/mr30_ws/src/mr-30/scripts/publish_strawberry_marker.py

import rospy
from visualization_msgs.msg import Marker

def publish_model_marker():
    rospy.init_node('strawberry_marker_publisher')
    target = rospy.get_param('~target_model', 'bunny')
    
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "world"  # Use the frame from static_transform_publisher
        marker.header.stamp = rospy.Time.now()
        marker.ns = "model"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD

        if target == 'bunny':
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.9
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.2
            marker.scale.x = 1.2
            marker.scale.y = 1.2
            marker.scale.z = 1.2
            marker.mesh_resource = "file:///home/ok/mr30_ws/src/mr-30/models/bunny.dae"
        elif target == 'box_with_ball':
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.7
            marker.pose.position.y = 0
            marker.pose.position.z = 0.104
            marker.scale.x = 1/1000
            marker.scale.y = 1/1000
            marker.scale.z = 1/1000
            marker.mesh_resource = "file:///home/ok/mr30_ws/src/mr-30/models/box_with_ball.stl"
        else:
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.mesh_resource = "file:///home/ok/mr30_ws/data/plant.stl"

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.mesh_use_embedded_materials = True
        
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_model_marker()
    except rospy.ROSInterruptException:
        pass