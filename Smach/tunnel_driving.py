import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray


class TunnelDriving:
    def __init__(self):
        # ROI 설정
        self.ROI_X_MIN = 0.0
        self.ROI_X_MAX = 0.4
        self.ROI_Y_MIN = -0.4
        self.ROI_Y_MAX = 0.4

        # PID Constants
        self.KP = 3.2
        self.KD = 0.008
        self.KI = 0.001

        # Variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = 0.0
        self.scan_data = None

        self.sub_scan = None
        self.pub_drive = rospy.Publisher("/ackermann_cmd_mux/input/Navigation", AckermannDriveStamped, queue_size=1)
        self.pub_markers = rospy.Publisher("/roi_markers", MarkerArray, queue_size=1)

    def start(self):
        if self.sub_scan is None:
            self.sub_scan = rospy.Subscriber("/scan", LaserScan, self.callback_scan)
            rospy.loginfo("[TunnelDriving] Subscriber started.")

    def stop(self):
        if self.sub_scan is not None:
            self.sub_scan.unregister()
            self.sub_scan = None
            rospy.loginfo("[TunnelDriving] Subscriber stopped.")

    def callback_scan(self, scan):
        self.scan_data = scan

    def process_scan_data(self):
        if self.scan_data is None:
            return "waiting"

        ranges = np.array(self.scan_data.ranges)
        angles = np.linspace(self.scan_data.angle_min + math.pi, self.scan_data.angle_max + math.pi, len(ranges))

        points = np.array([[r * math.cos(angle), r * math.sin(angle)]
                          for r, angle in zip(ranges, angles) if r < self.scan_data.range_max])

        left_points = np.array([point for point in points if self.ROI_Y_MIN <= point[1] <= 0 and self.ROI_X_MIN <= point[0] <= self.ROI_X_MAX])
        right_points = np.array([point for point in points if 0 <= point[1] <= self.ROI_Y_MAX and self.ROI_X_MIN <= point[0] <= self.ROI_X_MAX])

        if len(left_points) == 0 or len(right_points) == 0:
            return "done"

        left_center = np.mean(left_points, axis=0)
        right_center = np.mean(right_points, axis=0)
        center_x = (left_center[0] + right_center[0]) / 2
        center_y = (left_center[1] + right_center[1]) / 2

        error = center_y
        current_time = rospy.Time.now().to_sec()
        delta_time = current_time - self.last_time if self.last_time != 0 else 1e-6
        self.integral += error * delta_time
        derivative = (error - self.prev_error) / delta_time if delta_time > 0 else 0

        steering_angle = -(self.KP * error + self.KD * derivative + self.KI * self.integral)

        if abs(steering_angle) > 20.0 * (math.pi / 180.0):
            speed = 0.3
        elif abs(steering_angle) > 10.0 * (math.pi / 180.0):
            speed = 0.3
        else:
            speed = 0.3

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = -steering_angle
        msg.drive.speed = speed
        self.pub_drive.publish(msg)

        self.prev_error = error
        self.last_time = current_time

        return "running"

    def visualize_markers(self, left_points, right_points):
        marker_array = MarkerArray()

        for i, point in enumerate(left_points):
            left_marker = self.create_marker(i, point[0], point[1], 0.0, 1.0, 0.0)
            left_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(left_marker)

        for i, point in enumerate(right_points, start=len(left_points)):
            right_marker = self.create_marker(i, point[0], point[1], 0.0, 0.0, 1.0)
            right_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(right_marker)

        if len(left_points) > 0 and len(right_points) > 0:
            left_center = np.mean(left_points, axis=0)
            right_center = np.mean(right_points, axis=0)
            center_x = (left_center[0] + right_center[0]) / 2
            center_y = (left_center[1] + right_center[1]) / 2

            center_marker = self.create_marker(len(left_points) + len(right_points), center_x, center_y, 1.0, 1.0, 0.0)
            center_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(center_marker)

        self.pub_markers.publish(marker_array)

    def create_marker(self, marker_id, x, y, r, g, b):
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0
        marker.id = marker_id

        return marker

    def run(self):
        rospy.init_node("Tunnel_Driving", anonymous=True)
        self.start()
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                self.process_scan_data()
                rate.sleep()
        finally:
            self.stop()


if __name__ == "__main__":
    TunnelDriving().run()
