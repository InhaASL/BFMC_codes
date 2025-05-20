#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

class StanleyTracker:
    def __init__(self):
        self.wheelbase = rospy.get_param("~wheelbase", 0.26)
        self.k_gain = rospy.get_param("~stanley_k", 1.0)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.418)
        self.search_window = rospy.get_param("~search_window", 20)
        self.speed = rospy.get_param("~speed", 1.0)
        self.min_speed_ratio = rospy.get_param("~min_speed_ratio", 0.2)

        self.path_np = None  # np.array of shape (N, 2)
        self.path_received = False
        self.search_start_idx = 0

        self.debugging = True

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.path_sub = rospy.Subscriber("/global_path", Path, self.path_callback)
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/tracking_target_marker", Marker, queue_size=1)

        # self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # self.path_sub = rospy.Subscriber("/path", Path, self.path_callback)
        # self.drive_pub = rospy.Publisher("/ackermann_cmd_mux/input/Navigation", AckermannDriveStamped, queue_size=10)

    def path_callback(self, msg):
        self.path_np = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses])
        self.path_received = True
        # self.search_start_idx = 0

    def odom_callback(self, msg):
        if not self.path_received or self.path_np is None or len(self.path_np) == 0:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

        lookahead_idx = self.find_lookahead_index(x, y, lookahead_distance=0.8)
        if lookahead_idx < self.search_start_idx:
            lookahead_idx = self.search_start_idx
        self.search_start_idx = lookahead_idx

        tx, ty = self.path_np[lookahead_idx]
        dx = tx - x
        dy = ty - y

        path_yaw = math.atan2(dy, dx)
        heading_error = self.normalize_angle(path_yaw - yaw)
        cross_track_error = math.sin(heading_error) * math.hypot(dx, dy)
        cross_track_term = math.atan2(self.k_gain * cross_track_error, self.speed)

        steering = heading_error + cross_track_term
        steering = max(min(steering, self.max_steering_angle), -self.max_steering_angle)

        dynamic_speed = self.speed * (1 - (abs(steering) / self.max_steering_angle) * (1 - self.min_speed_ratio))
        dynamic_speed = max(0.1, dynamic_speed)

        self.publish_drive(steering, dynamic_speed)
        if self.debugging:
         self.publish_marker(tx, ty) 

    def find_lookahead_index(self, x, y, lookahead_distance):
        lookahead_distance = max(lookahead_distance - self.speed *0.3 ,0.5)
        start = self.search_start_idx
        end = min(start + self.search_window, len(self.path_np))

        segment = self.path_np[start:end]
        dx = segment[:, 0] - x
        dy = segment[:, 1] - y
        dists = np.sqrt(dx**2 + dy**2)

        for i, d in enumerate(dists):
            if d >= lookahead_distance:
                return start + i

        return len(self.path_np) - 1

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def publish_drive(self, steering_angle, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        self.drive_pub.publish(msg)
        

    def publish_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

if __name__ == "__main__":
    rospy.init_node("stanley_tracker")
    StanleyTracker()
    rospy.spin()
