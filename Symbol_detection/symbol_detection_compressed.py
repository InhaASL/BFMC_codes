#!/usr/bin/env python3

from ultralytics import YOLO
from cv_bridge import CvBridge
import logging
import json

import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

class Symbol_detection():
    def __init__(self):
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        # 프레임 지정
        self.frame_interval = 2
        self.frame_count = 0

        rospy.init_node('symbol_detect_node')
        self.image_sub = rospy.Subscriber('/oak/rgb/image_rect_color/compressed', CompressedImage, self.image_callback)
        self.symbol_pub = rospy.Publisher('/detected_symbol', String, queue_size=10)
        # debug 이미지를 publish할 토픽 생성
        self.debug_pub = rospy.Publisher('/debug', Image, queue_size=10)
        self.bridge = CvBridge()

        self.model = YOLO("/home/tony/yolo/yolo/demo_model.pt")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4

    def image_callback(self, msg):
        try:
            # rosbag에서 발행되는 sensor_msgs/Image이면 imgmsg_to_cv2 사용
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("이미지 변환 실패: %s", e)
            return

        self.process_frame(frame)

    def process_frame(self, frame):   
        self.frame_count += 1

        if self.frame_count % self.frame_interval == 0:
            results = self.model.predict(source=frame, show=False, 
                                         conf=self.conf_threshold, iou=self.iou_threshold)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            self.flag_callback(boxes, results)

            # 후처리된 annotated_frame를 ROS 이미지 메시지로 변환 후 debug 토픽으로 publish
            debug_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.debug_pub.publish(debug_msg)
        else:
            pass    

            

    def flag_callback(self, boxes, results):
        detections = []  
        if boxes and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = results[0].names[class_id]
                xyxy = box.xyxy[0]
                width = float(xyxy[2] - xyxy[0])
                height = float(xyxy[3] - xyxy[1])
                area = width * height 
                detection = {
                    "class": class_name,
                    "area": round(area, 2)
                }
                print(f"Detected: {class_name}, Confidence: {confidence:.2f}, Area: {area:.2f}")
                detections.append(detection)
            self.symbol_pub.publish(json.dumps(detections))
        else:
            print("Nothing detected!")
            self.symbol_pub.publish(json.dumps([]))
        print("--------------------------------")

if __name__ == '__main__':
    try:
        symbol_detection = Symbol_detection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
