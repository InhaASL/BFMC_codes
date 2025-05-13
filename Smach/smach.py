#!/usr/bin/env python3
import rospy
import smach
import smach_ros
import autorace_cone
import autorace_laneFollow_2
import autorace_wall
import 
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class MissionStart(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['start_mission'])
        
    def execute(self, userdata):
        rospy.loginfo('State: MISSION START')
        rospy.sleep(3)  
        return 'start_mission'

class LaneFollow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['static_obstacle', 'dynamic_obstacle', 'rubber_cone', 'tunnel', 'roundabout', 'gate', 'parking', 'mission_end'])
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

class PathTracking(smach.State):
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

class Stop(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obstacle_cleared'])

    def execute(self, userdata):

        return 'obstacle_cleared'




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
                

class Parking(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['mission_done'])
           

    def execute(self, userdata):
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
        smach.StateMachine.add('PATH_TRACKING', PathTracking(), transitions={'static_obstacle': 'STATIC_OBSTACLE_AVOID',
                                                                         'dynamic_obstacle': 'DYNAMIC_OBSTACLE_AVOID',
                                                                         'rubber_cone': 'RUBBER_CONE_DRIVING',
                                                                         'tunnel': 'TUNNEL_DRIVING',
                                                                         'roundabout': 'ROUND_ABOUT',
                                                                         'gate': 'CROSSING_GATE',
                                                                         'parking': 'PARKING',
                                                                         'mission_end': 'MISSION_END'})

        smach.StateMachine.add('STOP', Stop(), transitions={'obstacle_cleared': 'LANE_FOLLOW'})
        smach.StateMachine.add('TUNNEL_DRIVING', Tunnel(), transitions={'crossroadA': 'CROSSROAD_A',
                                                                         'crossroadB': 'CROSSROAD_B'})
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
