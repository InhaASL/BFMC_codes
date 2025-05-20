#!/usr/bin/env python3
import rospy
import smach
import smach_ros
import path_tracking
import tunnel_driving
import lane_Follow
from flag_listener import MissionFlagListener
from std_msgs.msg import String

class MissionStart(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['start_mission'])
        self.listener = flag_listener

    def execute(self, userdata):
        rospy.loginfo('State: MISSION START - Waiting for mission_start flag')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            flag = self.listener.get_flag()
            if flag == 'mission_start':
                self.listener.clear_flag()
                return 'start_mission'
            rate.sleep()

class LaneFollow(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['path_tracking', 'stop', 'parking', 'tunnel', 'mission_end'])
        self.listener = flag_listener
        self.line_follow = lane_Follow.LaneFollow()
        self.rate = rospy.Rate(20)
        self.deaccel_flag = False

    def execute(self, userdata):
        rospy.loginfo('State: LANE FOLLOW')
        while not rospy.is_shutdown():
            flag = self.listener.get_flag()
            if flag in ['path_tracking', 'stop', 'parking', 'tunnel', 'mission_end']:
                self.listener.clear_flag()
                return flag
            elif flag == 'highway_start':
                rospy.loginfo('State: Highway ENTRANCE')
                self.listener.clear_flag()
                self.deaccel_flag = True
            elif flag == 'highway_end':
                rospy.loginfo('State: Highway EXIT')
                self.listener.clear_flag()
                self.deaccel_flag = False
            self.line_follow.run(self.deaccel_flag)
            self.rate.sleep()

class PathTracking(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['lane_follow', 'stop', 'parking', 'tunnel', 'mission_end'])
        self.listener = flag_listener
        self.path_tracking = path_tracking.StanleyTracker()
        self.rate = rospy.Rate(20)
        self.deaccel_flag = False

    def execute(self, userdata):
        rospy.loginfo('State: PATH TRACKING')
        while not rospy.is_shutdown():
            flag = self.listener.get_flag()
            if flag in ['lane_follow', 'stop', 'parking', 'tunnel', 'mission_end']:
                self.listener.clear_flag()
                return flag
            elif flag == 'highway_start':
                rospy.loginfo('State: Highway ENTRANCE')
                self.listener.clear_flag()
                self.deaccel_flag = True
            elif flag == 'highway_end':
                rospy.loginfo('State: Highway EXIT')
                self.listener.clear_flag()
                self.deaccel_flag = False
            self.path_tracking.run(self.deaccel_flag)
            self.rate.sleep()

class Stop(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['lane_follow', 'path_tracking', 'parking', 'tunnel', 'mission_end'])
        self.listener = flag_listener

    def execute(self, userdata):
        rospy.loginfo('State: STOP')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            flag = self.listener.get_flag()
            if flag in ['lane_follow', 'path_tracking', 'parking', 'tunnel', 'mission_end']:
                self.listener.clear_flag()
                return flag
            rate.sleep()

class Tunnel(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['lane_follow', 'path_tracking', 'parking', 'stop', 'mission_end'])
        self.listener = flag_listener
        self.tunnel = tunnel_driving.TunnelDriving()

    def execute(self, userdata):
        rospy.loginfo('State: Tunnel DRIVING')
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            result = self.tunnel.process_scan_data()
            if result == "done":
                rospy.loginfo("Tunnel Finished. Waiting Marker...")
                while not rospy.is_shutdown():
                    flag = self.listener.get_flag()
                    if flag in ['lane_follow', 'path_tracking', 'parking', 'stop', 'mission_end']:
                        self.listener.clear_flag()
                        return flag
                    rate.sleep()
            rospy.sleep(0.1)

class Parking(smach.State):
    def __init__(self, flag_listener):
        smach.State.__init__(self, outcomes=['lane_follow', 'path_tracking', 'stop', 'tunnel', 'mission_end'])
        self.listener = flag_listener

    def execute(self, userdata):
        rospy.loginfo('State: PARKING')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            flag = self.listener.get_flag()
            if flag in ['lane_follow', 'path_tracking', 'stop', 'tunnel', 'mission_end']:
                self.listener.clear_flag()
                return flag
            rate.sleep()

class MissionEnd(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['terminate'])

    def execute(self, userdata):
        rospy.loginfo('State: MISSION END')
        rospy.sleep(3)
        return 'terminate'

def main():
    rospy.init_node('Smach_node')
    flag_listener = MissionFlagListener()
    sm = smach.StateMachine(outcomes=['MISSION_TERMINATED'])

    with sm:
        smach.StateMachine.add('MISSION_START', MissionStart(flag_listener), transitions={'start_mission': 'LANE_FOLLOW'})
        smach.StateMachine.add('LANE_FOLLOW', LaneFollow(flag_listener), transitions={'path_tracking': 'PATH_TRACKING', 'stop': 'STOP', 'parking': 'PARKING', 'tunnel': 'TUNNEL_DRIVING', 'mission_end': 'MISSION_END'})
        smach.StateMachine.add('PATH_TRACKING', PathTracking(flag_listener), transitions={'lane_follow': 'LANE_FOLLOW', 'stop': 'STOP', 'parking': 'PARKING', 'tunnel': 'TUNNEL_DRIVING', 'mission_end': 'MISSION_END'})
        smach.StateMachine.add('STOP', Stop(flag_listener), transitions={'lane_follow': 'LANE_FOLLOW', 'path_tracking': 'PATH_TRACKING', 'parking': 'PARKING', 'tunnel': 'TUNNEL_DRIVING', 'mission_end': 'MISSION_END'})
        smach.StateMachine.add('TUNNEL_DRIVING', Tunnel(flag_listener), transitions={'lane_follow': 'LANE_FOLLOW', 'path_tracking': 'PATH_TRACKING', 'parking': 'PARKING', 'stop': 'STOP', 'mission_end': 'MISSION_END'})
        smach.StateMachine.add('PARKING', Parking(flag_listener), transitions={'lane_follow': 'LANE_FOLLOW', 'path_tracking': 'PATH_TRACKING', 'stop': 'STOP', 'tunnel': 'TUNNEL_DRIVING', 'mission_end': 'MISSION_END'})
        smach.StateMachine.add('MISSION_END', MissionEnd(), transitions={'terminate': 'MISSION_TERMINATED'})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()