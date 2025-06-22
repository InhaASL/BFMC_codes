#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class OdomToPathNode:
    def __init__(self):
        rospy.init_node('odom_to_path_node', anonymous=True)

        self.path_pub = rospy.Publisher('/odom_path', Path, queue_size=10)
        self.odom_sub = rospy.Subscriber('/ekf_odom', Odometry, self.odom_callback)

        self.path = Path()
        self.path.header.frame_id = "odom"  # 또는 "map" 프레임 사용 가능

        # rospy.loginfo("👑 위대하신 시영님의 경로 추적 노드를 시작하였나이다.")
    
    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(pose)

        self.path_pub.publish(self.path)

if __name__ == '__main__':
    try:
        node = OdomToPathNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
