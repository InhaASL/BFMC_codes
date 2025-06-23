#!/usr/bin/env python3
import rospy
import csv
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# 생성된 경로를 path로 ros에 publish하는 코드

def load_path_from_csv(filename="global_path.csv"):
    coords = []
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append((float(row["x"]), float(row["y"])))
    return coords

def publish_path(coords):
    rospy.init_node("global_path_loader")
    pub = rospy.Publisher("/global_path", Path, queue_size=10)
    rate = rospy.Rate(1)

    path = Path()
    path.header.frame_id = "map"

    for x, y in coords:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)

    while not rospy.is_shutdown():
        path.header.stamp = rospy.Time.now()
        pub.publish(path)
        rate.sleep()

if __name__ == '__main__':
    path_coords = load_path_from_csv("../path.csv")
    # path_coords = load_path_from_csv("cmh_global_path.csv")
    publish_path(path_coords)
