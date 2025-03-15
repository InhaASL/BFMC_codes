#!/usr/bin/env python3

import cv2
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

        rospy.init_node('symbol_detect_node')
        self.image_sub = rospy.Subscriber('/image_raw', CompressedImage, self.image_callback)
        self.symbol_pub = rospy.Publisher('/detected_symbol', String, queue_size=10)
        self.bridge = CvBridge()

        self.model = YOLO("/home/tony/yolo/yolo/demo_model.pt")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45

        # 프레임 지정
        self.frame_interval = 10
        self.frame_count = 0

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
        detections = []  
        if boxes and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = results[0].names[class_id]
                xyxy = box.xyxy[0]
                width = float(xyxy[2] - xyxy[0])

                # 각 검출 결과를 딕셔너리로 저장
                detection = {
                    "class": class_name,
                    "width": round(width, 2)
                }
                print(f"Detected: {class_name}, width: {width:.2f}")
                detections.append(detection)
            self.symbol_pub.publish(json.dumps(detections))
        else:
            print("Nothing detected!")
            self.symbol_pub.publish(json.dumps([]))

if __name__ == '__main__':
    try:
        symbol_detection = Symbol_detection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
