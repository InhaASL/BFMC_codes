#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <std_srvs/Trigger.h>
#include <geometry_msgs/PoseStamped.h>

class GlobalOdometryNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // Subscribers
    ros::Subscriber local_odom_sub_;
    ros::Subscriber initial_pose_sub_;
    
    // Publishers
    ros::Publisher global_odom_pub_;
    
    // Services
    ros::ServiceServer reset_pose_service_;
    
    // Transform broadcaster
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    
    // Transform variables
    tf2::Transform global_to_local_transform_;
    tf2::Transform initial_global_pose_;
    tf2::Transform last_local_pose_;
    
    // Flags
    bool is_initialized_;
    bool first_local_odom_received_;
    
    // Frame IDs
    std::string global_frame_id_;
    std::string local_frame_id_;
    std::string base_link_frame_id_;
    
    // Initial pose
    geometry_msgs::Pose initial_pose_;

public:
    GlobalOdometryNode() : pnh_("~"), is_initialized_(false), first_local_odom_received_(false)
    {
        // Get parameters
        pnh_.param<std::string>("global_frame_id", global_frame_id_, "map");
        pnh_.param<std::string>("local_frame_id", local_frame_id_, "odom_local");
        pnh_.param<std::string>("base_link_frame_id", base_link_frame_id_, "base_link");
        
        // Initialize subscribers
        local_odom_sub_ = nh_.subscribe("local_odom", 10, &GlobalOdometryNode::localOdomCallback, this);
        initial_pose_sub_ = nh_.subscribe("initialpose", 1, &GlobalOdometryNode::initialPoseCallback, this);
        
        // Initialize publisher
        global_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("global_odom", 10);
        
        // Initialize service
        reset_pose_service_ = nh_.advertiseService("reset_pose", &GlobalOdometryNode::resetPoseCallback, this);
        
        // Initialize transforms
        global_to_local_transform_.setIdentity();
        initial_global_pose_.setIdentity();
        last_local_pose_.setIdentity();
        
        // Set default initial pose (origin)
        initial_pose_.position.x = 0.0;
        initial_pose_.position.y = 0.0;
        initial_pose_.position.z = 0.0;
        initial_pose_.orientation.x = 0.0;
        initial_pose_.orientation.y = 0.0;
        initial_pose_.orientation.z = 0.0;
        initial_pose_.orientation.w = 1.0;
        
        ROS_INFO("Global Odometry Node initialized");
        ROS_INFO("Waiting for local odometry data on topic: %s", local_odom_sub_.getTopic().c_str());
        ROS_INFO("Publishing global odometry on topic: %s", global_odom_pub_.getTopic().c_str());
        ROS_INFO("Use /initialpose topic or /reset_pose service to set initial pose");
    }
    
    void localOdomCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        if (!first_local_odom_received_)
        {
            // Store the first local pose as reference
            tf2::fromMsg(msg->pose.pose, last_local_pose_);
            first_local_odom_received_ = true;
            
            if (!is_initialized_)
            {
                // If no initial pose was set, use origin as default
                initializePose(initial_pose_);
            }
            
            ROS_INFO("First local odometry received. Global odometry initialized.");
        }
        
        if (!is_initialized_)
        {
            ROS_WARN_THROTTLE(5.0, "Global odometry not initialized. Waiting for initial pose...");
            return;
        }
        
        // Convert local odometry to global coordinate
        tf2::Transform current_local_pose;
        tf2::fromMsg(msg->pose.pose, current_local_pose);
        
        // Calculate relative transform from last local pose to current local pose
        tf2::Transform local_delta = last_local_pose_.inverse() * current_local_pose;
        
        // Apply the delta to global pose
        tf2::Transform current_global_pose = initial_global_pose_ * local_delta;
        
        // Update last local pose
        last_local_pose_ = current_local_pose;
        initial_global_pose_ = current_global_pose;
        
        // Create global odometry message
        nav_msgs::Odometry global_odom;
        global_odom.header.stamp = msg->header.stamp;
        global_odom.header.frame_id = global_frame_id_;
        global_odom.child_frame_id = base_link_frame_id_;
        
        // Set pose
        tf2::toMsg(current_global_pose, global_odom.pose.pose);
        
        // Transform velocity from local to global frame
        tf2::Vector3 local_linear_vel(msg->twist.twist.linear.x, 
                                      msg->twist.twist.linear.y, 
                                      msg->twist.twist.linear.z);
        tf2::Vector3 local_angular_vel(msg->twist.twist.angular.x, 
                                       msg->twist.twist.angular.y, 
                                       msg->twist.twist.angular.z);
        
        // Rotate linear velocity to global frame
        tf2::Vector3 global_linear_vel = current_global_pose.getBasis() * local_linear_vel;
        tf2::Vector3 global_angular_vel = current_global_pose.getBasis() * local_angular_vel;
        
        global_odom.twist.twist.linear.x = global_linear_vel.getX();
        global_odom.twist.twist.linear.y = global_linear_vel.getY();
        global_odom.twist.twist.linear.z = global_linear_vel.getZ();
        global_odom.twist.twist.angular.x = global_angular_vel.getX();
        global_odom.twist.twist.angular.y = global_angular_vel.getY();
        global_odom.twist.twist.angular.z = global_angular_vel.getZ();
        
        // Copy covariance matrices
        global_odom.pose.covariance = msg->pose.covariance;
        global_odom.twist.covariance = msg->twist.covariance;
        
        // Publish global odometry
        global_odom_pub_.publish(global_odom);
        
        // Broadcast transform
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped.header.stamp = msg->header.stamp;
        transform_stamped.header.frame_id = global_frame_id_;
        transform_stamped.child_frame_id = base_link_frame_id_;
        
        transform_stamped.transform.translation.x = current_global_pose.getOrigin().getX();
        transform_stamped.transform.translation.y = current_global_pose.getOrigin().getY();
        transform_stamped.transform.translation.z = current_global_pose.getOrigin().getZ();
        
        tf2::Quaternion q = current_global_pose.getRotation();
        transform_stamped.transform.rotation.x = q.getX();
        transform_stamped.transform.rotation.y = q.getY();
        transform_stamped.transform.rotation.z = q.getZ();
        transform_stamped.transform.rotation.w = q.getW();
        
        tf_broadcaster_.sendTransform(transform_stamped);
    }
    
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
    {
        ROS_INFO("Received initial pose from /initialpose topic");
        initializePose(msg->pose.pose);
    }
    
    bool resetPoseCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
    {
        ROS_INFO("Reset pose service called - resetting to origin");
        
        geometry_msgs::Pose origin_pose;
        origin_pose.position.x = 0.0;
        origin_pose.position.y = 0.0;
        origin_pose.position.z = 0.0;
        origin_pose.orientation.x = 0.0;
        origin_pose.orientation.y = 0.0;
        origin_pose.orientation.z = 0.0;
        origin_pose.orientation.w = 1.0;
        
        initializePose(origin_pose);
        
        res.success = true;
        res.message = "Pose reset to origin successfully";
        return true;
    }
    
    void initializePose(const geometry_msgs::Pose& pose)
    {
        if (!first_local_odom_received_)
        {
            // Store the initial pose for later initialization
            initial_pose_ = pose;
            ROS_INFO("Initial pose stored. Waiting for first local odometry...");
            return;
        }
        
        // Set initial global pose
        tf2::fromMsg(pose, initial_global_pose_);
        
        is_initialized_ = true;
        
        ROS_INFO("Global odometry initialized at position: (%.2f, %.2f, %.2f)", 
                 pose.position.x, pose.position.y, pose.position.z);
        
        // Convert quaternion to RPY for logging
        tf2::Quaternion q;
        tf2::fromMsg(pose.orientation, q);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        ROS_INFO("Initial orientation (RPY): (%.2f, %.2f, %.2f) degrees", 
                 roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "global_odometry_node");
    
    try
    {
        GlobalOdometryNode node;
        ros::spin();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Exception in global_odometry_node: %s", e.what());
        return 1;
    }
    
    return 0;
}