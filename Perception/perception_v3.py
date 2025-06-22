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
        4: 'stop_obstacle',
        5: 'stop_second',
        6: 'stop_traffic',
        7: 'roundabout',
        8: 'tunnel',
        9: 'mission_start',
        10: 'mission_end'}

        self.current_state = 5
        self.previous_state = 1

        self.scan_data = None
        self.detected_symbols = []
        self.stop_line_detected = True ############## True: 사용x, False: 사용o ################
        self.x = self.y = self.yaw = None
        self.roundabout_init_flag = True

        self.handled_label = None

        rospy.Subscriber("/detected_symbol", String, self.detected_symbol_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/stop_line', Bool, self.stop_line_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.mission_flag_pub = rospy.Publisher('/mission_flag', String, queue_size=5)

        rospy.Timer(rospy.Duration(0.1), self.state_machine)

    def detected_symbol_callback(self, msg):
        '''
        Class : {0: 'car', 1: 'crosswalk', 2: 'greenlight', 3: 'highwayend', 
                 4: 'highwayentry', 5: 'noentry', 6: 'object' ,7: 'oneway',
                 7: 'parking', 8: 'priorityroad', 9: 'redlight', 10: 'roundabout', 
                 12: 'stop', 13: 'yellowlight'}
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
        else:
            self.stop_line_detected = False

    def odom_callback(self, msg):
        if msg is not None:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            orientation = msg.pose.pose.orientation
            _, _, self.yaw = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    def get_detected_label(self, target_labels, area_thresh=7000.0):
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        target_labels = [l.lower() for l in target_labels]

        candidates = []
        for symbol in self.detected_symbols:
            label = symbol.get("class", "").lower()
            area  = float(symbol.get("area", 0.0))

            if label in target_labels and area >= area_thresh:
                candidates.append((area, label))

        if not candidates:
            return None

        # 가장 큰 area를 가진 후보 선택
        max_area, max_label = max(candidates, key=lambda x: x[0])

        # 이전에 처리한 레이블과 다를 때만 갱신하고 리턴
        if max_label != self.handled_label:
            self.handled_label = max_label
            return max_label

        return None

    ######################################## 인식 파트 #######################################################

    def is_detected_obstacle(self, angle_range_1=(160, 180), angle_range_2=(-160, -180), distance_range=(0.01,0.7)):
        ### 라이다 범위 확인 필요 ###
        '''
        전방 40도 0.01m~0.7m
        '''

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

                return (angles >= a2) & (angles <= a1)

        mask1 = angle_mask(angle_range_1)
        mask2 = angle_mask(angle_range_2)

        sector_mask = (mask1 | mask2) & valid_distance
        return np.any(sector_mask)
    
    def is_detected_tunnel(self, angle_offsets=[90, 110, 130], wall_distance_thresh=0.6, mode="entry"):
        if not self.scan_data:
            return False
        ranges = np.array(self.scan_data.ranges)
        angle_min = self.scan_data.angle_min
        angle_increment = self.scan_data.angle_increment
        num_points = len(ranges)

        # 좌우 각도 리스트 (rad)
        left_angles = [np.radians(a) for a in angle_offsets]
        right_angles = [np.radians(-a) for a in angle_offsets]

        left_points = []
        right_points = []

        def get_distance(angle_rad):
            idx = int((angle_rad - angle_min) / angle_increment)
            if 0 <= idx < num_points:
                dist = ranges[idx]
                if np.isfinite(dist) and dist < 3.0:
                    return dist
            return None

        for la, ra in zip(left_angles, right_angles):
            dl = get_distance(la)
            dr = get_distance(ra)

            if dl is None or dr is None:
                return False

            left_points.append((dl * np.cos(la), dl * np.sin(la)))
            right_points.append((dr * np.cos(ra), dr * np.sin(ra)))

        distances = [
            np.hypot(lx - rx, ly - ry)
            for (lx, ly), (rx, ry) in zip(left_points, right_points)
        ]
        mean_wall_distance = np.mean(distances)

        if mode == "entry":
            return mean_wall_distance < wall_distance_thresh
        elif mode == "exit":
            return mean_wall_distance > wall_distance_thresh
        else:
            return False
        
    ####################################### 상태 파트 ########################################################


    def state_machine(self, event):
        if self.current_state == 1:    # path tracking
            self.handle_path_tracking()
        elif self.current_state == 2:  # lane follow
            self.handle_lane_follow()
        elif self.current_state == 3:  # parking
            self.handle_parking()
        elif self.current_state == 4:  # stop_obstacle
            self.handle_stop_obstacle()
        elif self.current_state == 5:  # stop_second
            self.handle_stop_second()
        elif self.current_state == 6:  # stop_traffic
            self.handle_stop_traffic()
        elif self.current_state == 7:  # roundabout
            self.handle_roundabout()
        elif self.current_state == 8:  # tunnel
            self.handle_tunnel()
        elif self.current_state == 9:  # mission start
            self.handle_mission_start()
        elif self.current_state == 10:  # mission end
            rospy.loginfo("Mission ended. Shutting down perception node.")
            rospy.signal_shutdown("Mission completed.")
            return
        
    def transition_to(self, new_state):
        self.previous_state = self.current_state
        self.current_state = new_state


    ######################################### 메인 동작 파트  ######################################################
      
    def handle_path_tracking(self):
        label_detected = self.get_detected_label(["object", "highwayentry", "highwayend", "redlight", "stop", "roundabout", 'crosswalk'], 7000.0)
        
        if label_detected != None:
            rospy.loginfo("----------------------")
            rospy.loginfo(label_detected)

        # Obstacle Detect
        if label_detected in ["object"] and self.is_detected_obstacle((160, 180), (-160, -180), (0.01, 0.7)):
            self.publish_mission_flag("stop_obstacle")
            self.transition_to(4)

        # Stop Sign Detect
        if label_detected == "stop" and self.stop_line_detected:
            self.publish_mission_flag("stop_second")
            self.transition_to(5)
        
        # Crosswalk Detect
        if label_detected == "crosswalk" and self.stop_line_detected:
            self.publish_mission_flag("stop_second")
            self.transition_to(5)

        # Red light Detect
        if label_detected == "redlight" and self.stop_line_detected:
            self.publish_mission_flag("stop_traffic")
            self.transition_to(6)

        # Roundabout Detect
        if label_detected == "roundabout" and self.stop_line_detected:
            self.publish_mission_flag("roundabout")
            self.transition_to(7)

        # Tunnel Detect
        if self.is_detected_tunnel(angle_offsets=[160, 135, 110], wall_distance_thresh=1.15, mode="entry"):
            self.publish_mission_flag("tunnel")
            self.transition_to(8)

        # Highway Check            
        if label_detected == "highwayentry":
            rospy.loginfo("Recognize Highway Entrance")
            self.publish_mission_flag("highway_start")
        elif label_detected == "highwayend":
            rospy.loginfo("Recognize Highway Exit")
            self.publish_mission_flag("highway_end")

    def handle_lane_follow(self):
        label_detected = self.get_detected_label(["object", "highwayentry", "highwayend", "redlight", "stop", "roundabout", 'crosswalk'], 7000.0)
        
        if label_detected != None:
            rospy.loginfo("----------------------")
            rospy.loginfo(label_detected)

        # Obstacle Detect
        if label_detected in ["object"] and self.is_detected_obstacle((160, 180), (-160, -180), (0.01, 0.7)):
            self.publish_mission_flag("stop_obstacle")
            self.transition_to(4)

        # Stop Sign Detect
        if label_detected == "stop" and self.stop_line_detected:
            self.publish_mission_flag("stop_second")
            self.transition_to(5)
        
        # Crosswalk Detect
        if label_detected == "crosswalk" and self.stop_line_detected:
            self.publish_mission_flag("stop_second")
            self.transition_to(5)

        # Red light Detect
        if label_detected == "redlight" and self.stop_line_detected:
            self.publish_mission_flag("stop_traffic")
            self.transition_to(6)

        # Roundabout Detect
        if label_detected == "roundabout" and self.stop_line_detected:
            self.publish_mission_flag("roundabout")
            self.transition_to(7)

        # Tunnel Detect
        if self.is_detected_tunnel(angle_offsets=[160, 135, 110], wall_distance_thresh=1.15, mode="entry"):
            rospy.loginfo("Tunnel Entry")
            self.publish_mission_flag("tunnel")
            self.transition_to(8)

        # Highway Check            
        if label_detected == "highwayentry":
            rospy.loginfo("Recognize Highway Entrance")
            self.publish_mission_flag("highway_start")
        elif label_detected == "highwayend":
            rospy.loginfo("Recognize Highway Exit")
            self.publish_mission_flag("highway_end")


######################################### 미션 동작 파트  ######################################################


    def handle_tunnel(self):
        if self.is_detected_tunnel(angle_offsets=[90, 70, 50], wall_distance_thresh=0.8, mode="exit"):
            rospy.loginfo("Tunnel Exit")
            self.publish_mission_flag("path_traking")
            self.transition_to(1)

    def handle_stop_obstacle(self):
        if not self.is_detected_obstacle((160, 180), (-160, -180), (0.01, 0.7)):
            self.publish_mission_flag("path_tracking")
            self.transition_to(1)

    def handle_stop_second(self):
        rospy.sleep(1.0)
        self.publish_mission_flag("path_tracking")
        self.transition_to(self.previous_state)
    
    def handle_stop_traffic(self):
        label_detected = self.get_detected_label("greenlight", 7000.0)

        if label_detected != None:
            rospy.loginfo("----------------------")
            rospy.loginfo(label_detected)

        if label_detected == "greenlight":
            self.publish_mission_flag("path_tracking")
            self.transition_to(1)

    def handle_roundabout(self):
        if self.roundabout_init_flag:
            if self.is_detected_obstacle((160, 180), (-160, -180), (0.01, 0.7)):
                self.roundabout_init_flag = False
                self.publish_mission_flag("path_tracking")

        if self.is_detected_obstacle((160, 180), (-160, -180), (0.01, 0.7)):
            self.publish_mission_flag("stop_second")
            self.transition_to(5)

        if self.stop_line_detected:
            self.publish_mission_flag("path_tracking")
            self.transition_to(1)
            self.roundabout_init_flag = True

    def handle_parking(self):
        pass

    def handle_mission_start(self):
        label_detected = self.get_detected_label("greenlight", 7000.0)
        if label_detected == "greenlight":
            self.publish_mission_flag("path_tracking")
            self.transition_to(1)

    def publish_mission_flag(self, flag):
        rospy.loginfo(f"Publishing mission flag: {flag}")
        self.mission_flag_pub.publish(String(data=flag))

if __name__ == "__main__":
    try:
        node = PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
