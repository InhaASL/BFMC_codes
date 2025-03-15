#!/usr/bin/env python3

import json
import rospy
from std_msgs.msg import String

def detected_symbol_callback(msg):
    try:
        # 수신한 JSON 문자열을 파이썬 객체(리스트)로 파싱합니다.
        detections = json.loads(msg.data)
    except json.JSONDecodeError as e:
        rospy.logerr("JSON 파싱 오류: %s", e)
        return

    # 파싱된 리스트를 순회하며 각 검출 결과 출력
    for detection in detections:
        class_name = detection.get("class", "Unknown")
        confidence = detection.get("confidence", 0.0)
        width = detection.get("width", 0.0)
        rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Width: {width}")

def listener():
    rospy.init_node('detected_symbol_listener', anonymous=True)
    rospy.Subscriber("/detected_symbol", String, detected_symbol_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
