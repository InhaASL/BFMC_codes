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
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        rospy.init_node('symbol_detect_node')
        self.symbol_pub = rospy.Publisher('/detected_symbol', String, queue_size=10)

        self.model = YOLO("/home/tony/yolo/best.pt")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45

        self.frame_interval = 10
        self.frame_count = 0

    def process_frame(self, frame):   
        self.frame_count += 1

        # 지정한 간격의 프레임일 때만 검출 수행
        if self.frame_count % self.frame_interval == 0:
            results = self.model.predict(source=frame, show=False, 
                                         conf=self.conf_threshold, iou=self.iou_threshold)
            annotated_frame = results[0].plot()

            boxes = results[0].boxes
            self.flag_callback(boxes, results)
            cv2.imshow("YOLO Live Detection", annotated_frame)
        else:
            cv2.imshow("YOLO Live Detection", frame)

    def flag_callback(self, boxes, results):
        detected_symbols = []
        if boxes and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = results[0].names[class_id]
                print(f"Detected: {class_name} with confidence: {confidence:.2f}")
                detected_symbols.append(class_name)
            detected_symbols_str = ", ".join(detected_symbols)
            self.symbol_pub.publish(detected_symbols_str)
        else:
            print("Nothing detected!")
            self.symbol_pub.publish("Nothing detected!")

if __name__ == '__main__':
    symbol_detection = Symbol_detection()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 받아올 수 없습니다.")
            break

        symbol_detection.process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
