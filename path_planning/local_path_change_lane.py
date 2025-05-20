
#!/usr/bin/env python3
import rospy
import pickle
import networkx as nx
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from math import sqrt

class LaneChangeReplanner:
    def __init__(self):
        # Load graph and edge data
        graph_file = rospy.get_param("~graph_data_file")
        with open(graph_file, "rb") as f:
            data = pickle.load(f)
        self.G = data["graph"]
        self.edges_df = data["edges_df"]
        self.positions = data["positions"]

        self.obstacle_detected = False
        self.current_node = None

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/obstacle_detected", Bool, self.obstacle_callback)
        self.command_pub = rospy.Publisher("/replan_command", String, queue_size=10)
        self.replan_path_pub = rospy.Publisher("/replan_path", Path, queue_size=10)

        rospy.loginfo("Lane Change Replanner ready.")

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data
        if self.obstacle_detected and self.current_node:
            self.evaluate_replan(self.current_node)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_node = self.find_closest_node(x, y)

    def find_closest_node(self, x, y):
        min_dist = float("inf")
        closest = None
        for node_id, (nx, ny) in self.positions.items():
            dist = (x - nx)**2 + (y - ny)**2
            if dist < min_dist:
                min_dist = dist
                closest = node_id
        return closest

    def evaluate_replan(self, node_id):
        neighbors = list(self.G.successors(node_id))
        for nbr in neighbors:
            if self.G[node_id][nbr].get('dotted', False):
                rospy.loginfo(f"Lane change possible: {node_id} → {nbr}")
                path = self.generate_path(node_id, nbr)
                self.publish_path(path)
                self.command_pub.publish("LANE_CHANGE")
                return
        rospy.logwarn(f"No lane change possible from {node_id}. STOP issued.")
        self.command_pub.publish("STOP")

    def generate_path(self, start, target):
        try:
            path_nodes = nx.shortest_path(self.G, source=start, target=target)
            return path_nodes
        except nx.NetworkXNoPath:
            rospy.logwarn("No path found between nodes.")
            return []

    def publish_path(self, node_list):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for node_id in node_list:
            x, y = self.positions[node_id]
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.replan_path_pub.publish(path_msg)

if __name__ == "__main__":
    rospy.init_node("lane_change_replanner")
    LaneChangeReplanner()
    rospy.spin()