#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose2D.h>
#include <tf/transform_broadcaster.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

class AckermannOdom
{
private:
    ros::NodeHandle nh_;
    ros::Publisher odom_pub_;
    ros::Subscriber drive_sub_;
    tf::TransformBroadcaster tf_broadcaster_;

    double x_, y_, theta_;
    double wheel_base_;
    bool use_tf_;
    ros::Time last_time_;

public:
    AckermannOdom() : x_(0.0), y_(0.0), theta_(0.0)
    {
        nh_.param("wheel_base", wheel_base_, 0.5);   
        nh_.param("use_tf", use_tf_, false);          
        nh_.param("drive_gain", drive_gain, 1.0);
        nh_.param("steering_gain", steering_gain_, 1.0);

        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 50);
        drive_sub_ = nh_.subscribe("ackermann_drive", 10, &AckermannOdom::driveCallback, this);

        last_time_ = ros::Time::now();
    }

    void driveCallback(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg)
    {
        double v = msg->drive.speed;
        double steering_angle = msg->drive.steering_angle;

        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_time_).toSec();
        last_time_ = current_time;

        double delta_theta = (v / wheel_base_) * tan(steering_angle) * dt;
        theta_ += delta_theta * steering_gain_;

        double dx = v * cos(theta_) * dt * drive_gain;
        double dy = v * sin(theta_) * dt * drive_gain;

        x_ += dx;
        y_ += dy;

        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(theta_);

        if (use_tf_)
        {
            geometry_msgs::TransformStamped odom_tf;
            odom_tf.header.stamp = current_time;
            odom_tf.header.frame_id = "odom";
            odom_tf.child_frame_id = "base_link";
            odom_tf.transform.translation.x = x_;
            odom_tf.transform.translation.y = y_;
            odom_tf.transform.translation.z = 0.0;
            odom_tf.transform.rotation = odom_quat;
            tf_broadcaster_.sendTransform(odom_tf);
        }

        nav_msgs::Odometry odom;
        odom.header.stamp = current_time;
        odom.header.frame_id = "odom";

        odom.pose.pose.position.x = x_;
        odom.pose.pose.position.y = y_;
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation = odom_quat;

        odom.child_frame_id = "base_link";
        odom.twist.twist.linear.x = v;
        odom.twist.twist.angular.z = delta_theta / dt;

        odom_pub_.publish(odom);
    }
};
