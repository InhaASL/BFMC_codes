# mission_flag_listener.py
import rospy
from std_msgs.msg import String
from threading import Lock

class MissionFlagListener:
    def __init__(self):
        self._flag = None
        self._lock = Lock()
        self.sub = rospy.Subscriber("/mission_flag", String, self._callback)

    def _callback(self, msg):
        with self._lock:
            self._flag = msg.data

    def get_flag(self):
        with self._lock:
            return self._flag
