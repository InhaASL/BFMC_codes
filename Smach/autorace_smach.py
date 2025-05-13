#!/usr/bin/env python3
import rospy
import smach
import smach_ros
import autorace_cone
import autorace_laneFollow_2
import autorace_wall
import math
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class MissionStart(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['start_mission'])
        
    def execute(self, userdata):
        rospy.loginfo('State: MISSION START')
        rospy.sleep(3)  # Wait for 3 second
        return 'start_mission'

class LaneFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['static_obstacle', 'dynamic_obstacle', 
                                             'rubber_cone', 'tunnel','roundabout',
                                             'gate','parking', 'mission_end'])
        self.mission_flag_sub = rospy.Subscriber('/mission_flag', String, self.flag_callback)
        self.mission_flag = None  # Default mission flag
        self.line_follow = autorace_laneFollow_2.LaneFollow()

        self.rate = rospy.Rate(20)
        self.deaccel_flag = False

    def flag_callback(self, msg):
        if msg is not None:
            self.mission_flag = msg.data
        else:
            self.mission_flag = None

    def execute(self, userdata):
        rospy.loginfo('State: LANE FOLLOW')
        while not rospy.is_shutdown():
            if self.mission_flag == 'static_obstacle':
                self.mission_flag = None
                return 'static_obstacle'
            elif self.mission_flag == 'dynamic_obstacle':
                self.mission_flag = None
                return 'dynamic_obstacle'
            elif self.mission_flag == 'rubber_cone':
                self.mission_flag = None
                return 'rubber_cone'
            elif self.mission_flag == 'tunnel':
                self.mission_flag = None
                return 'tunnel'
            elif self.mission_flag == 'gate':
                self.mission_flag = None
                return 'gate'
            elif self.mission_flag == 'roundabout':
                self.mission_flag = None
                return 'roundabout'
            elif self.mission_flag == 'kids_zone':
                rospy.loginfo('State: KIDSZONE ENTRANCE')
                self.mission_flag = None
                self.deaccel_flag = True

            elif self.mission_flag == 'kids_zone_end':
                rospy.loginfo('State: KIDSZONE EXIT')
                self.mission_flag = None
                self.deaccel_flag = False

            elif self.mission_flag == 'parking':
                self.mission_flag = None
                return 'parking'
            
            elif self.mission_flag == 'mission_end':
                self.mission_flag = None
                return 'mission_end'
            
            self.line_follow.run(self.deaccel_flag)
            self.rate.sleep()

class StaticObstacleAvoid(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstacle_cleared'])
        self.desired_ranges = []  # 초기값 설정
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.angle_min = -math.radians(173)
        self.angle_max = math.radians(173)
        self.data_updated = False  # 데이터 갱신 플래그

    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]

        # 특정 조건을 만족하는 거리 값들을 필터링하여 저장
        self.desired_ranges = [
            ranges[i] for i, angle in enumerate(angles)
            if (angle <= self.angle_min or angle >= self.angle_max) and ranges[i] <= 1.0
        ]
        self.data_updated = True  # 데이터가 갱신되었음을 표시

    def execute(self, userdata):
        rospy.loginfo('State: STATIC OBSTACLE AVOID')
        rate = rospy.Rate(10)  # 0.1초마다 조건 확인

        while not rospy.is_shutdown():
            if self.data_updated:  # 데이터가 갱신되었을 때만 길이를 확인
                print(len(self.desired_ranges))
                if len(self.desired_ranges) <= 2:
                    return 'obstacle_cleared'
                self.data_updated = False  # 데이터 확인 후 플래그 초기화
            rate.sleep()

class DynamicObstacleAvoid(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstacle_cleared'])

    def execute(self, userdata):
        rospy.loginfo('State: DYNAMIC OBSTACLE AVOID')
        rospy.sleep(3.5)  # Simulate dynamic obstacle avoidance
        return 'obstacle_cleared'

class RubberConeDriving(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.cone_driving = autorace_cone.ConeFollower() 
 
    def execute(self, userdata):
        rospy.loginfo('State: RUBBER CONE DRIVING')
        
        # Rubber cone 추종 로직 시작
        while not rospy.is_shutdown():
            result = self.cone_driving.control_loop() 
            if result == "done":  
                rospy.loginfo("Cone successfully followed. Exiting state.")
                return 'mission_done'
            
            rospy.sleep(0.1)  

class CrossingGate(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.desired_ranges = []  # 초기값 설정
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.angle_min = -math.radians(165)
        self.angle_max = math.radians(165)
        self.data_updated = False  # 데이터 갱신 플래그

    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]

        # 특정 조건을 만족하는 거리 값들을 필터링하여 저장
        self.desired_ranges = [
            ranges[i] for i, angle in enumerate(angles)
            if (angle <= self.angle_min or angle >= self.angle_max) and ranges[i] <= 1.0
        ]
        self.data_updated = True  # 데이터가 갱신되었음을 표시

    def execute(self, userdata):
        rospy.loginfo('State: CROSSING GATE')
        rate = rospy.Rate(10)  # 0.1초마다 조건 확인

        while not rospy.is_shutdown():
            if self.data_updated:  # 데이터가 갱신되었을 때만 길이를 확인
                print(len(self.desired_ranges))
                if len(self.desired_ranges) <= 2:
                    break
                self.data_updated = False  # 데이터 확인 후 플래그 초기화
            rate.sleep()

        rospy.sleep(7)
        return 'mission_done'

class Roundabout(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.desired_ranges = []
        self.data_updated = False  # 데이터 갱신 플래그
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.drive.steering_angle = -0.6
        self.drive_msg.drive.speed = 0.4
        self.rate = rospy.Rate(30)
        rospy.sleep(0.1)
    
    def scan_callback(self, data):
        ranges = data.ranges
        angles = [data.angle_min + i * data.angle_increment for i in range(len(ranges))]
        angle_min = -math.radians(110)
        angle_max = math.radians(150)
        
        # 조건에 맞는 거리 데이터 필터링 및 업데이트
        self.desired_ranges = [
            ranges[i] for i, angle in enumerate(angles)
            if (angle <= angle_min or angle >= angle_max) and ranges[i] <= 1.5
        ]
        self.data_updated = True  # 데이터 갱신 표시

    def execute(self, userdata):
        rospy.loginfo('State: ROUNDABOUT ENTRANCE')
        waiting_rate = rospy.Rate(5)  # 0.2초마다 조건 확인

        # 데이터가 수신될 때까지 대기
        while not rospy.is_shutdown():
            if self.data_updated:  # 데이터가 업데이트되었을 때만 확인
                rospy.loginfo(f'Desired ranges count: {len(self.desired_ranges)}')  # 디버깅 출력
                if len(self.desired_ranges) <= 2:
                    break
                self.data_updated = False  # 플래그 초기화
            waiting_rate.sleep()
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.0:
                break
            self.rate.sleep()
        return 'mission_done'
    
class Tunnel(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['crossroadA','crossroadB'])
        self.mission_flag_sub = rospy.Subscriber('/mission_flag', String, self.flag_callback)
        self.tunnel = autorace_wall.CenterlineFollow()
        self.mission_flag = None

    def flag_callback(self, msg):
        if msg is not None:
            self.mission_flag = msg.data
        else:
            self.mission_flag = None

    def execute(self, userdata):
        rospy.loginfo('State: Tunnel DRIVING')
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            result = self.tunnel.process_scan_data() 
            if result == "done":  
                rospy.loginfo("Tunnel Finished. Waiting Marker...")
                while not rospy.is_shutdown():

                    if self.mission_flag == 'crossroadA':
                        rospy.loginfo("A Marker Detected. Exiting state.")
                        self.mission_flag = None
                        return 'crossroadA'
                    elif self.mission_flag == 'crossroadB':
                        rospy.loginfo("B Marker Detected. Exiting state.")
                        self.mission_flag = None
                        return 'crossroadB'
                    rate.sleep()
            rospy.sleep(0.1)
                
class CrossroadA(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)

    def execute(self, userdata):
        rospy.loginfo('State: MARKER A')
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = 0.4
        drive_msg.drive.speed = 0.4
        rate = rospy.Rate(30)
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.0:
                break
            rate.sleep()
        return 'mission_done'
    
class CrossroadB(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)

    def execute(self, userdata):
        rospy.loginfo('State: MARKER B')
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = -0.4
        drive_msg.drive.speed = 0.4
        rate = rospy.Rate(30)
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.0:
                break
            rate.sleep()
        return 'mission_done'

class Parking(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
        self.drive_msg = AckermannDriveStamped()
           

    def execute(self, userdata):
        rospy.loginfo('State: Parking')
        rospy.sleep(2)
        
        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.speed = -0.2
        rate = rospy.Rate(30)
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 0.6:
                break
        rospy.sleep(1)

        self.drive_msg.drive.steering_angle = 0.7
        self.drive_msg.drive.speed = 0.2
        rate = rospy.Rate(30)
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 2.5:
                break
                
        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.speed = 0.2
        rate = rospy.Rate(30)
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 0.3:
                break
        
        self.drive_msg.drive.steering_angle = -0.7
        self.drive_msg.drive.speed = 0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.3:
                break
                
        rospy.sleep(1)
        
        self.drive_msg.drive.steering_angle = 0.7
        self.drive_msg.drive.speed = -0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.5:
                break
            
        rospy.sleep(10)

        self.drive_msg.drive.steering_angle = -0.9
        self.drive_msg.drive.speed = 0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 0.6:
                break
            
        rospy.sleep(1)
        self.drive_msg.drive.steering_angle = 0.7
        self.drive_msg.drive.speed = -0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 1.5:
                break
        rospy.sleep(1)

        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.speed = 0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 2.0:
                break

        self.drive_msg.drive.steering_angle = 0.7
        self.drive_msg.drive.speed = 0.2
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self.ack_pub.publish(self.drive_msg)
            if rospy.Time.now().to_sec() - start_time >= 2.5:
                break
            
        return 'mission_done'

class MissionEnd(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['terminate'])

    def execute(self, userdata):
        rospy.loginfo('State: MISSION END')
        rospy.sleep(3)  # Simulate end process
        return 'terminate'

def main():
    rospy.init_node('STATE')
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['MISSION_TERMINATED'])

    # Open the container
    with sm:
        # Add states to the state machine
        smach.StateMachine.add('MISSION_START', MissionStart(), transitions={'start_mission': 'LANE_FOLLOW'})
        smach.StateMachine.add('LANE_FOLLOW', LaneFollow(), transitions={'static_obstacle': 'STATIC_OBSTACLE_AVOID',
                                                                         'dynamic_obstacle': 'DYNAMIC_OBSTACLE_AVOID',
                                                                         'rubber_cone': 'RUBBER_CONE_DRIVING',
                                                                         'tunnel': 'TUNNEL_DRIVING',
                                                                         'roundabout': 'ROUND_ABOUT',
                                                                         'gate': 'CROSSING_GATE',
                                                                         'parking': 'PARKING',
                                                                         'mission_end': 'MISSION_END'})
        smach.StateMachine.add('RUBBER_CONE_DRIVING', RubberConeDriving(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('STATIC_OBSTACLE_AVOID', StaticObstacleAvoid(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('DYNAMIC_OBSTACLE_AVOID', DynamicObstacleAvoid(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('CROSSING_GATE', CrossingGate(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('TUNNEL_DRIVING', Tunnel(), transitions={'crossroadA': 'CROSSROAD_A',
                                                                         'crossroadB': 'CROSSROAD_B'})
        smach.StateMachine.add('ROUND_ABOUT', Roundabout(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('CROSSROAD_A', CrossroadA(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('CROSSROAD_B', CrossroadB(), transitions={'mission_done': 'LANE_FOLLOW'})
        smach.StateMachine.add('PARKING', Parking(), transitions={'mission_done': 'MISSION_END'})
        smach.StateMachine.add('MISSION_END', MissionEnd(), transitions={'terminate': 'MISSION_TERMINATED'})
    # # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    # Execute the state machine
    outcome = sm.execute()

    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
