#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

class StanleyTracker:
    def __init__(self):
        self.wheelbase = rospy.get_param("~wheelbase", 0.26)
        self.k_gain = rospy.get_param("~stanley_k", 1.0)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.418)
        self.search_window = rospy.get_param("~search_window", 40)
        self.speed = rospy.get_param("~speed", 2.0)
        self.min_speed_ratio = rospy.get_param("~min_speed_ratio", 0.2)

        self.path_np = None  # np.array of shape (N, 2)
        self.path_received = False
        self.search_start_idx = 0

        self.debugging = True

        self.x = None
        self.y = None
        self.yaw = None

        self.pose_sub = None
        self.path_sub = None
        self.drive_pub = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/lookahead_marker", Marker, queue_size=1)

    def start(self):
        if self.pose_sub is None:
            self.pose_sub = rospy.Subscriber("/global_pose", PoseStamped, self.pose_callback) 
            self.path_sub = rospy.Subscriber("/global_path", Path, self.path_callback)

    def stop(self):
        if self.pose_sub is not None:
            self.pose_sub.unregister()
            self.pose_sub = None
            self.path_sub.unregister()
            self.path_sub = None

    def path_callback(self, msg):
        self.path_np = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses])
        self.path_received = True

    def pose_callback(self, msg):
        if not self.path_received or self.path_np is None or len(self.path_np) == 0:
            return

        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        orientation = msg.pose.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def process_tracking(self):
        if self.x is None or self.y is None or self.yaw is None:
            return 
        if self.path_np is None or len(self.path_np) == 0:
            return

        self.search_start_idx = self.find_nearest_index(self.x, self.y)

        lookahead_idx = self.find_lookahead_index(self.x, self.y, lookahead_distance=0.8)

        tx, ty = self.path_np[lookahead_idx]
        dx = tx - self.x
        dy = ty - self.y

        path_yaw = math.atan2(dy, dx)
        heading_error = self.normalize_angle(path_yaw - self.yaw)
        cross_track_error = math.sin(heading_error) * math.hypot(dx, dy)
        cross_track_term = math.atan2(self.k_gain * cross_track_error, self.speed)

        steering = heading_error + cross_track_term
        steering = max(min(steering, self.max_steering_angle), -self.max_steering_angle)

        dynamic_speed = self.speed * (1 - (abs(steering) / self.max_steering_angle) * (1 - self.min_speed_ratio))
        dynamic_speed = max(0.1, dynamic_speed)

        self.publish_drive(steering, dynamic_speed)

        if self.debugging:
            self.publish_marker(tx, ty)

    def find_nearest_index(self, x, y):
        if self.path_np is None or len(self.path_np) == 0:
            return 0

        start = self.search_start_idx
        end = min(start + self.search_window, len(self.path_np))
        segment = self.path_np[start:end]

        dx = segment[:, 0] - x
        dy = segment[:, 1] - y
        dists = np.sqrt(dx**2 + dy**2)

        nearest_local_idx = int(np.argmin(dists))
        return start + nearest_local_idx

    def find_lookahead_index(self, x, y, lookahead_distance):
        lookahead_distance = max(lookahead_distance - self.speed * 0.3, 0.5)
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

    def run(self):
        self.start()
        rate = rospy.Rate(20)
        try:
            while not rospy.is_shutdown():
                self.process_tracking()
                rate.sleep()
        finally:
            self.stop()

if __name__ == "__main__":
    rospy.init_node("stanley_tracker")
    StanleyTracker().run()
