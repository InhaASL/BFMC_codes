#!/usr/bin/env python3

import math
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import logging

import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

class Symbol_detection():
    def __init__(self):

        # 프레임 지정
        self.frame_interval = 10
        self.frame_count = 0
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        rospy.init_node('symbol_detect_node')
        self.image_sub = rospy.Subscriber('/image_raw', CompressedImage, self.image_callback)
        self.symbol_pub = rospy.Publisher('/detected_symbol', String, queue_size=10)
        self.bridge = CvBridge()

        self.model = YOLO("/home/tony/yolo/best.pt")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45

        

    def image_callback(self, msg):
        try:
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
            cv2.imshow("YOLO Live Detection", annotated_frame)
        else:
            cv2.imshow("YOLO Live Detection", frame)
        cv2.waitKey(1)

    def flag_callback(self, boxes, results):
        detected_symbols = []
        if boxes and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = results[0].names[class_id]
                print(f"Detected: {class_name}, confidence: {confidence:.2f}")
                detected_symbols.append(class_name)
            detected_symbols_str = ", ".join(detected_symbols)
            self.symbol_pub.publish(detected_symbols_str)
        else:
            print("Nothing detected!")
            self.symbol_pub.publish("Nothing detected!")

if __name__ == '__main__':
    try:
        symbol_detection = Symbol_detection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
