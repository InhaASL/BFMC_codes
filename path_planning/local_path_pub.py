#!/usr/bin/env python
import rospy
import pandas as pd
import tf
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from math import sqrt

class LocalPathPlanner:
    def __init__(self):
        # Load global path
        self.global_path_df = pd.read_csv(rospy.get_param("~path_file"))
        self.global_path = list(self.global_path_df.itertuples(index=False, name=None))
        self.path_index = 0

        # ROS setup
        self.path_pub = rospy.Publisher("local_path", Path, queue_size=10)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

        self.local_path_length = 10  # How many points ahead to plan
        self.replan_triggered = False


    def odom_callback(self, msg):
        position = msg.pose.pose.position
        x, y = position.x, position.y

        # Update nearest path index
        while self.path_index < len(self.global_path):
            gx, gy, _ = self.global_path[self.path_index]
            dist = sqrt((x - gx)**2 + (y - gy)**2)
            if dist < 0.5:
                self.path_index += 1
            else:
                break

        # Replanning condition (example: certain point)
        if not self.replan_triggered and self.path_index > len(self.global_path) // 2:
            rospy.logwarn("Triggering local replanning!")
            self.replan_triggered = True
            self.modify_path()

        self.publish_local_path()

    def modify_path(self):
        # Simulate local replan: reverse remaining path as a dummy example
        remaining = self.global_path[self.path_index:]
        self.global_path = self.global_path[:self.path_index] + list(reversed(remaining))

    def publish_local_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for i in range(self.path_index, min(self.path_index + self.local_path_length, len(self.global_path))):
            x, y, _ = self.global_path[i]
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

if __name__ == "__main__":
    rospy.init_node("path_follower")
    LocalPathPlanner()
    rospy.spin()