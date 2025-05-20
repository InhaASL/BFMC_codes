#!/usr/bin/env python3
import rospy
import json
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf
import numpy as np
import math
from std_srvs.srv import Trigger

class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')

        # 상태 정의
        self.MISSION_STATES = {
            1: 'path_tracking',
            2: 'lane_follow',
            3: 'parking',
            4: 'stop',
            5: 'mission_start',
            6: 'mission_end'
        }
        self.current_state = 5

        self.scan_data = None
        self.detected_symbols = []
        self.stop_line_detected = False
        self.x = self.y = self.yaw = None

        rospy.Subscriber("/detected_symbol", String, self.detected_symbol_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/stop_line', Bool, self.stop_line_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.mission_flag_pub = rospy.Publisher('/mission_flag', String, queue_size=5)

        rospy.Timer(rospy.Duration(0.1), self.state_machine)

    def detected_symbol_callback(self, msg):
        '''
        Class : {0: 'Car', 1: 'CrossWalk', 2: 'Greenlight', 3: 'HighwayEnd', 
                 4: 'HighwayEntry', 5: 'NoEntry', 6: 'OneWay', 7: 'Parking', 
                 8: 'Pedestrian', 9: 'PriorityRoad', 10: 'Redlight', 11: 'Roundabout', 
                 12: 'Stop', 13: 'Yellowlight', 14: 'object_1', 15: 'object_2'}
        '''
        try:
            self.detected_symbols = json.loads(msg.data)  # YOLO 결과 리스트로 가정
        except json.JSONDecodeError:
            rospy.logwarn("Invalid JSON in /detected_symbol")

    def scan_callback(self, msg):
        if msg is not None:
            self.scan_data = msg

    def stop_line_callback(self, msg):
        if msg is not None:
            self.stop_line_detected = msg.data

    def odom_callback(self, msg):
        if msg is not None:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            orientation = msg.pose.pose.orientation
            _, _, self.yaw = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    def is_label_detect(self, target_labels, area_thresh=1000.0):
        if isinstance(target_labels, str):
            target_labels = [target_labels]

        target_labels = [label.lower() for label in target_labels]

        for symbol in self.detected_symbols:
            label = symbol.get("label", "").lower()
            area = symbol.get("area", 0.0)
            if label in target_labels and area >= area_thresh:
                return label
        return None

    def is_obstacle_scan(self, angle_range_1, angle_range_2, distance_range=(0.1, 0.5)):
        if not self.scan_data:
            return False

        ranges = np.array(self.scan_data.ranges)
        angle_min = self.scan_data.angle_min  # rad
        angle_increment = self.scan_data.angle_increment  # rad
        num_ranges = len(ranges)

        angles = angle_min + angle_increment * np.arange(num_ranges)  

        min_dist, max_dist = distance_range
        valid_distance = (ranges > min_dist) & (ranges < max_dist)

        def angle_mask(angle_range):
            a1, a2 = np.radians(angle_range)
            if a1 < a2:
                return (angles >= a1) & (angles <= a2)
            else:

                return (angles >= a1) | (angles <= a2)

        mask1 = angle_mask(angle_range_1)
        mask2 = angle_mask(angle_range_2)

        sector_mask = (mask1 | mask2) & valid_distance

        return np.any(sector_mask)

    def state_machine(self, event):
        if self.current_state == 1:    # path tracking
            self.handle_path_tracking()
        elif self.current_state == 2:  # lane follow
            self.handle_lane_follow()
        elif self.current_state == 3:  # parking
            self.handle_parking()
        elif self.current_state == 4:  # stop
            self.handle_stop()
        elif self.current_state == 5:  # mission start
            self.handle_mission_start()
        elif self.current_state == 6:  # mission end
            rospy.loginfo("Mission ended. Shutting down perception node.")
            rospy.signal_shutdown("Mission completed.")
            return

    def handle_path_tracking(self):
        label_detected = self.is_label_detect(["object_1", "object_2", "HighwayEntry", "HighwayEnd", "Stop"], 500.0)

        # Obstacle Detect
        if label_detected == "object_1" or label_detected == "object_1" and self.is_obstacle_scan((-160,-180), (0, 20), (0.2, 0.7)):
            self.publish_mission_flag("stop")
            self.current_state = 4

        # Stop Sign Detect
        if label_detected == "Stop" and self.stop_line_detected:
            self.publish_mission_flag("stop")
            self.current_state = 4

        # Highway Check            
        if label_detected == "HighwayEntry":
            rospy.loginfo("Recognize Highway Entrance")
            self.publish_mission_flag("highway_start")
        elif label_detected == "HighwayEnd":
            rospy.loginfo("Recognize Highway Exit")
            self.publish_mission_flag("highway_end")

    def handle_lane_follow(self):
        for symbol in self.detected_symbols:
            label = symbol.get("label", "").lower()
            if label == "stop":
                self.publish_mission_flag("stop")
                self.current_state = 4
                return
            elif label == "highway":
                self.publish_mission_flag("lane_follow")
                self.current_state = 2
                return
            elif label == "parking":
                self.publish_mission_flag("parking")
                self.current_state = 3
                return
            
        # Obstacle Detect
        if self.is_label_detect(["object_1", "object_2"], 1000) and self.is_obstacle_scan((-160,-180), (0, 20), (0.2, 0.7)):
            self.publish_mission_flag("stop")
            self.current_state = 4

    def handle_stop(self):
        if self.scan_data:
            min_distance = min(self.scan_data.ranges)
            if min_distance > 1.0:  # 안전거리 확보됨
                rospy.loginfo("Obstacle cleared, resuming path_tracking")
                self.publish_mission_flag("path_tracking")
                self.current_state = 1

    def handle_parking(self):
        pass

    def handle_mission_start(self):
        if self.is_label_detect("Greenlight", 1000.0):
            self.publish_mission_flag("path_tracking")
            self.current_state = 1
            return

    def publish_mission_flag(self, flag):
        rospy.loginfo(f"Publishing mission flag: {flag}")
        self.mission_flag_pub.publish(String(data=flag))

if __name__ == "__main__":
    try:
        node = PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
