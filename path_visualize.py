#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class PosePathBuilder:
    def __init__(self):
        rospy.init_node('pose_path_builder')

        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.pose_sub = rospy.Subscriber('/global_pose', PoseStamped, self.pose_callback)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

    def pose_callback(self, msg: PoseStamped):
        # 헤더 갱신
        self.path_msg.header.stamp = rospy.Time.now()

        # 들어온 pose를 Path에 추가
        self.path_msg.poses.append(msg)

        # Path 퍼블리시
        self.path_pub.publish(self.path_msg)

if __name__ == '__main__':
    node = PosePathBuilder()
    rospy.spin()

