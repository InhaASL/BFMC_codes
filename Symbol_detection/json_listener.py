#!/usr/bin/env python3

import json
import rospy
from std_msgs.msg import String

def detected_symbol_callback(msg):
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

def listener():
    rospy.init_node('detected_symbol_listener', anonymous=True)
    rospy.Subscriber("/detected_symbol", String, detected_symbol_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
