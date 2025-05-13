1#!/usr/bin/env python3 
import rospy
import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String
import json


class perception:
    def __init__(self):
        rospy.init_node('perception')
        
        # Subscribers and Publishers
        rospy.Subscriber("/detected_symbol", String, self.detected_symbol_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/stop_line', bool, self.stop_line_callback)
        self.mission_flag_pub = rospy.Publisher('/mission_flag', String, queue_size=5)
        self.debug_publisher1 = rospy.Publisher('/debug_image1', Image, queue_size=10)
        
        # Obstacle and Line Detection Parameters
        self.obstacle_flag = False
        self.area_threshold = 200
        self.line_threshold = 5000
        self.kidszone_threshold = 200
        self.cooldown = 10
        self.last_count_time = 0
        self.mask = None

        self.kidzone_flag = False
        self.cur_kidzone  = False
        self.cone_detected = False
        self.roundabout_flag = False
        self.tunnel_flag = False
        self.parking_flag = False
        self.current_marker_id = None

        self.cur_mission = 0
        self.line_cnt = 0

        self.current_marker_id = None

        rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")

    def detected_symbol_callback(self, msg):
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            rospy.logerr("JSON 파싱 오류: %s", e)
            return

        # 파싱된 리스트를 순회하며 각 검출 결과 출력
        for detection in detections:
            class_name = detection.get("class", "Unknown")
            width = detection.get("area", 0.0)
            rospy.loginfo(f"Detected: {class_name}, Area: {width}")

    def scan_callback(self, data):
        angles = np.linspace(data.angle_min, data.angle_max, len(data.ranges))
        ranges = np.array(data.ranges, dtype=np.float32)

        if self.cur_mission == 0:
            self.cone_check(ranges, angles)
        elif self.cur_mission == 5:
            self.tunnel_check(ranges)
        elif self.cur_mission == 4:
            if rospy.Time.now().to_sec() - self.start_time >= 7.0:
                self.obstacle_check(ranges, angles)
        elif self.cur_mission == 7:
            self.check_parking(ranges, angles)

    def cone_check(self, ranges, angles):
        # Filter data based on angle and range conditions
        angle_min = -math.radians(150)
        angle_max = math.radians(150)
        valid_angles = (angles <= angle_min) | (angles >= angle_max)
        close_objects = ranges <= 1.5
        filtered_ranges = ranges[valid_angles & close_objects]
        filtered_angles = angles[valid_angles & close_objects]
        
        # Convert polar coordinates (range, angle) to Cartesian coordinates (x, y)
        x = filtered_ranges * np.cos(filtered_angles)
        y = filtered_ranges * np.sin(filtered_angles)
        points = np.column_stack((x, y))
        
        if len(points) < 6:
            return
        # Apply DBSCAN for clustering
        db = DBSCAN(eps=0.15, min_samples=2).fit(points)
        labels = db.labels_
        
        # Count clusters with fewer than 8 points
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_points = points[labels == label]
            cluster_size = len(cluster_points)
            
            if cluster_size > 6:
                # Check if the cluster is within 0.5 meters of the reference point
                distances = np.linalg.norm(cluster_points, axis=1)
                # print(cluster_size, distances)
                if np.min(distances) <= 0.8:
                    self.cone_detected = True
                    self.pub_flag = True
                    
                    break  # Exit early since a Lavacorn is detected
    
    def tunnel_check(self, range):
        # Convert target angles to radians
        target_angle_1_rad = np.radians(110)
        target_angle_2_rad = np.radians(-110)

        # Compute indices for the angles
        num_points = len(range)
        angle_increment = 2*np.pi / len(range)
        idx_1 = int((target_angle_1_rad + np.pi) / (angle_increment + 1e-6))
        idx_2 = int((target_angle_2_rad + np.pi) / (angle_increment + 1e-6))

        # Ensure indices are within range
        idx_1 = max(0, min(idx_1, num_points - 1))
        idx_2 = max(0, min(idx_2, num_points - 1))

        # Get distances for these indices
        distance_1 = range[idx_1]
        distance_2 = range[idx_2]

        if distance_1 > 100 or distance_2 > 100:
            return

        # Compute the Euclidean distance between the two points
        distance_between_points = np.sqrt(
            (distance_1 * np.cos(target_angle_1_rad) - distance_2 * np.cos(target_angle_2_rad))**2 +
            (distance_1 * np.sin(target_angle_1_rad) - distance_2 * np.sin(target_angle_2_rad))**2
        )
        # print(distance_between_points)
        # Check if the distance is less than 60cm
        if distance_between_points < 0.6:
            self.tunnel_flag = True
            self.pub_flag = True
        else:
            self.tunnel_flag = False

    def obstacle_check(self, range, angle):
        angle_min = -math.radians(160)
        angle_max = math.radians(160)
        valid_angles = (angle <= angle_min) | (angle >= angle_max)  
        close_objects = range <= 0.7
        desired_ranges = range[valid_angles & close_objects]

    def check_parking(self, range, angle):
        angle_min = -math.radians(91)
        angle_max = -math.radians(89)
        valid_angles = (angle <= angle_min) & (angle >= angle_max)  
        close_objects = range <= 0.33
        desired_ranges = range[valid_angles & close_objects]
        
        if len(desired_ranges) > 0:
            self.parking_flag = True
            self.pub_flag = True
            rospy.loginfo("Parking Zone detected!")

    def image_callback(self, data):
        self.image_ = self.br.compressed_imgmsg_to_cv2(data, 'bgr8')
        self.warp_ = cv2.warpPerspective(self.image_, self.matrix, (self.image_.shape[1], self.image_.shape[0]))
        self.roi_ = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]

        if self.cur_mission == 1:
            if rospy.Time.now().to_sec() - self.start_time >= 12.0:
                self.mask = self.check_kidzone(self.roi_)
        elif self.cur_mission == 2:
            self.mask = self.check_line(self.roi_)
        elif self.cur_mission == 3 and self.roundabout_flag == False:
            self.mask = self.check_roundabouts(self.roi_)
        if self.mask is not None:
            self.debug_publisher1.publish(self.br.cv2_to_imgmsg(self.mask, 'mono8'))
        
    def check_line(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        current_time = rospy.get_time()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.line_threshold and (current_time - self.last_count_time) >= self.cooldown:
                self.line_cnt += 1
                self.last_count_time = current_time
                self.pub_flag = True
                rospy.loginfo("lane detected!")
                break
        return mask

    def check_kidzone(self, img):
        # Color masking and contour detection
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check area of detected contours
        red_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.kidszone_threshold:
                red_detected = True
                break

        # Handle flag transitions
        if red_detected and not self.kidzone_flag:
            self.kidzone_flag = True
            self.pub_flag = True  # Ensure flag triggers a message
            rospy.loginfo("KidZone detected!")
        elif not red_detected and self.kidzone_flag:
            self.kidzone_flag = False
            self.pub_flag = True  # Ensure flag triggers a message
            rospy.loginfo("KidZone ended!")

        return mask
    
        
    def publish_mission_flag(self):
        # Main flag publishing logic
            if self.cur_mission == 0 and self.cone_detected and self.pub_flag:
                self.mission_flag_pub.publish('rubber_cone')
                self.pub_flag = False
                self.cur_mission += 1
                self.start_time = rospy.Time.now().to_sec()
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 1 and self.pub_flag:
                if self.kidzone_flag:
                    self.mission_flag_pub.publish('kids_zone')
                else:
                    self.mission_flag_pub.publish('kids_zone_end')
                    self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 2 and self.line_cnt == 1 and self.pub_flag:
                    self.mission_flag_pub.publish('dynamic_obstacle')
                    self.cur_mission += 1
                    self.pub_flag = False
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 3 and self.roundabout_flag and self.pub_flag:
                self.mission_flag_pub.publish('roundabout')
                self.cur_mission += 1
                self.pub_flag = False
                self.start_time = rospy.Time.now().to_sec()
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 4 and  self.obstacle_flag and self.pub_flag:
                self.mission_flag_pub.publish('gate')
                self.obstacle_flag = False
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 5 and self.tunnel_flag and self.pub_flag:
                self.mission_flag_pub.publish('tunnel')
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 6 and self.pub_flag:
                if self.current_marker_id == 0:
                    self.mission_flag_pub.publish('crossroadA')
                    self.current_marker_id = None
                    self.cur_mission += 1
                    self.pub_flag = False
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
                elif self.current_marker_id == 4:
                    self.mission_flag_pub.publish('crossroadB')
                    self.cur_mission += 1
                    self.current_marker_id = None
                    self.pub_flag = False
                    rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 7 and self.parking_flag and self.pub_flag:
                self.mission_flag_pub.publish('parking')
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
            elif self.cur_mission == 8 and self.line_cnt == 2 and self.pub_flag:
                self.mission_flag_pub.publish('mission_end')
                self.cur_mission += 1
                self.pub_flag = False
                rospy.loginfo(f"CUR_Mission : {self.cur_mission} / Obstacle Flag : {self.obstacle_flag} / Current Line Check : {self.line_cnt} / Current Kids-Zone Flag : {self.kidzone_flag}")
                    

if __name__ == "__main__":
    try:
        node = perception()
        rate = rospy.Rate(10)  # Loop rate for flag publishing
        while not rospy.is_shutdown():
            node.publish_mission_flag()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
