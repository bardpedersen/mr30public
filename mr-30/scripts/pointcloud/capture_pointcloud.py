#! /usr/bin/env python3

import rospy
import threading
from sensor_msgs.msg import PointCloud2

class PointCloudCapture:
    def __init__(self):
        self.subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.publisher = rospy.Publisher('/camera/depth/color/points_pose', PointCloud2, queue_size=10)
        self.pointcloud = None
        self.lock = threading.Lock()

    def pointcloud_callback(self, msg):
        with self.lock:
            self.pointcloud = msg

    def publish_pointcloud(self):
        with self.lock:
            if self.pointcloud:
                self.publisher.publish(self.pointcloud)
                rospy.loginfo("Published point cloud to /camera/depth/color/points_pose")

    def run(self):
        rospy.loginfo("Press Enter to capture and publish the point cloud")
        input()  # Wait for Enter key press
        self.publish_pointcloud()

if __name__ == '__main__':
    rospy.init_node('pointcloud_capture_node')
    pointcloud_capture = PointCloudCapture()
    while not rospy.is_shutdown():
        pointcloud_capture.run()