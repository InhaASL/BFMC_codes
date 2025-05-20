import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from threading import Lock    

class DrivePublisher:
    def __init__(self):
        self._lock = Lock()
        self._publisher = rospy.Publisher("/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)

    def publish(self, steering_angle, speed):
        with self._lock:
            msg = AckermannDriveStamped()
            msg.drive.steering_angle = steering_angle
            msg.drive.speed = speed
            self._publisher.publish(msg)
