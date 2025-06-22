#!/usr/bin/env python3
import rospy
import csv
import os
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import signal
import sys
from datetime import datetime

class PosePathBuilder:
    def __init__(self):
        rospy.init_node('pose_path_builder')

        # Parameters
        self.pose_topic = rospy.get_param("~pose_topic", "/world_pose")
        self.save_path = rospy.get_param("~save_path", "path.csv")

        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        self.pose_list = []

        # Register shutdown hook
        signal.signal(signal.SIGINT, self.shutdown_handler)

        rospy.loginfo("PosePathBuilder initialized. Subscribing to %s", self.pose_topic)

    def pose_callback(self, msg: PoseStamped):
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(msg)
        self.pose_list.append(msg)
        self.path_pub.publish(self.path_msg)

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("Shutdown detected. Saving %d poses to %s...", len(self.pose_list), self.save_path)

        # 저장 디렉토리 없으면 생성
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            with open(self.save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

                for pose in self.pose_list:
                    p = pose.pose.position
                    o = pose.pose.orientation
                    t = pose.header.stamp.to_sec()
                    writer.writerow([t, p.x, p.y, p.z, o.x, o.y, o.z, o.w])

            rospy.loginfo("Successfully saved path to: %s", self.save_path)
        except Exception as e:
            rospy.logerr("Failed to save CSV: %s", str(e))

        # 종료
        sys.exit(0)

if __name__ == '__main__':
    PosePathBuilder()
    rospy.spin()
